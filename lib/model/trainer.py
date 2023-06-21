import os
import logging as log
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import trimesh
import mcubes
import wandb
from contextlib import nullcontext

from .ray import exponential_integration, cumsum, sigma_density_function
from ..utils.metrics import psnr

# Warning: you MUST NOT change the resolution of marching cube
RES = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
class Trainer(nn.Module):

    def __init__(self, config, model, pe, log_dir):

        super().__init__()

        self.cfg = config
        self.pos_enc = pe.to(DEVICE)
        self.mlp = model.to(DEVICE)
        self.log_dir = log_dir
        self.log_dict = {}

        self.init_optimizer()
        self.init_log_dict()

    def init_optimizer(self):
        
        trainable_parameters = list(self.mlp.parameters())
        self.optimizer = torch.optim.Adam(trainable_parameters, lr=self.cfg.lr, 
                                    betas=(self.cfg.beta1, self.cfg.beta2),
                                    weight_decay=self.cfg.weight_decay)

    def init_log_dict(self):
        """Custom log dict.
        """
        self.log_dict['total_loss'] = 0.0
        self.log_dict['rgb_loss'] = 0.0
        self.log_dict['total_iter_count'] = 0
        self.log_dict['image_count'] = 0


    def sample_points(self, ray_orig, ray_dir, hierarchical = True, near=1.0, far=3.0, num_points_c=64, num_points_f=128):
        """Sample points along rays. Retruns 3D coordinates of the points.
        TODO: One and extend this function to the hirachical sampling technique 
             used in NeRF or design a more efficient sampling technique for 
             better surface reconstruction.

        Args:
            ray_orig (torch.FloatTensor): Origin of the rays of shape [B, Nr, 3].
            ray_dir (torch.FloatTensor): Direction of the rays of shape [B, Nr, 3].
            near (float): Near plane of the camera.
            far (float): Far plane of the camera.
            num_points (int): Number of points (Np) to sample along the rays.

         Returns:
            points (torch.FloatTensor): 3D coordinates of the points of shape [B, Nr, Np, 3].
            z_vals (torch.FloatTensor): Depth values of the points of shape [B, Nr, Np, 1].
            deltas (torch.FloatTensor): Distance between the points of shape [B, Nr, Np, 1].

        """

        
        B, Nr = ray_orig.shape[:2]  #b = batch size Nr = number of rays per img

        t = torch.linspace(0.0, 1.0, num_points_c, device=ray_orig.device).view(1, 1, -1) + \
            (torch.rand(B, Nr, num_points_c, device=ray_orig.device)/ num_points_c)
        
        

        z_vals = near * (1.0 - t) + far * t
        points = ray_orig[:, :, None, :] + ray_dir[:, :, None, :] * z_vals[..., None]
        deltas = z_vals.diff(dim=-1, prepend=(torch.zeros(B, Nr, 1, device=z_vals.device)+ near))
        coords = points
        depth = z_vals[..., None]
        deltas = deltas[..., None]
        view_dir = ray_dir.unsqueeze(2).expand(coords.shape)
        if not hierarchical:  
            return coords, view_dir, depth, deltas
        
        #in case of hierarchical
        rgb, sigma = self.predict_radience(coords, view_dir) 
        rgb = rgb.detach()
        sigma = sigma.detach() 
        # rgb, sigma returns nan

        if self.cfg.network_type == 'double_mlp':
            sigma = sigma_density_function(sigma, s  = 1e5)

        # 128 ler 64 oldu num_points_c yi 64 e indirince
        # shape of rgb = [2,512,128,3]  -> batch size x number of rays per img x number of samples on a ray x channels
        # shape of sigma = [2,512,128,1]
        # shape of deltas = [2,512,128,1]
        
        sigma_delta_term = -1 * torch.mul(sigma, deltas)
        sigma_delta_cum = torch.cumsum(sigma_delta_term, dim = -2)
        sigma_delta_cum = torch.cat((torch.zeros((B,Nr,1,1), dtype=sigma_delta_cum.dtype, device=sigma_delta_cum.device), sigma_delta_cum), dim=-2)
        transmittance = torch.exp(sigma_delta_cum)
        transmittance = transmittance[:,:,:-1,:]  #last element is redundant
        opacity_term = 1 - torch.exp(sigma_delta_term)
        weights = torch.mul(transmittance, opacity_term)
        weights += 1e-5
        sum_weights = torch.sum(weights, dim = -2).unsqueeze(3)
        weights_normalized = weights / sum_weights
        weights_cdf = torch.cumsum(weights_normalized, dim = -2)   #[2,512,64,1]  each of 2,512 element includes a cdf
        weights_cdf = torch.cat((torch.zeros((B,Nr,1,1), dtype=sigma_delta_cum.dtype, device=sigma_delta_cum.device), weights_cdf), dim=-2)  #basa 0 ekle
        rand_var = torch.rand((B,Nr,num_points_f), device=weights_cdf.device, dtype=weights_cdf.dtype)  #need to sample [2,512,128] samples with these random variables
        weights_cdf_sq = torch.squeeze(weights_cdf, -1)
        inds = torch.searchsorted(weights_cdf_sq, rand_var, right=True)
        # inds has shape (..., N_samples) identifying the bin of each sample.
        below = (inds - 1).clamp(0)
        above = inds.clamp(max=weights_cdf_sq.shape[-1] - 1)
        
        inds_g = torch.stack([below, above], -1).view(
        *below.shape[:-1], below.shape[-1] * 2
        )
        
        #shape z vals = [2,512,64]
        #shape t = [2,512,64]
        cdf_g = torch.gather(weights_cdf_sq, -1, inds_g).view(*below.shape, 2)
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_vals_zero = z_vals[..., :1]
        z_vals_end = z_vals[..., -1:]
        z_vals_mid = torch.cat((z_vals_zero, z_vals_mid, z_vals_end), dim=-1)
        bins_g = torch.gather(z_vals_mid, -1, inds_g).view(*below.shape, 2)
        # cdf_g and bins_g are of shape (..., N_samples, 2) and identify
        # the cdf and the index of the two bin edges surrounding each sample.
        # bins_g shape = [2,512,128,2]

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 0.00001, torch.ones_like(denom), denom)
        t_fine = (rand_var - cdf_g[..., 0]) / denom

        
        z_vals_fine = bins_g[..., 0] + t_fine * (bins_g[..., 1] - bins_g[..., 0])
        z_vals, _ = torch.sort(torch.concat([z_vals, z_vals_fine], dim=-1), dim=-1)
        points = ray_orig[:, :, None, :] + ray_dir[:, :, None, :] * z_vals[..., None]
        deltas = z_vals.diff(dim=-1, prepend=(torch.zeros(B, Nr, 1, device=z_vals.device)+ near))

        coords = points
        depth = z_vals[..., None]
        deltas = deltas[..., None]

        view_dir = ray_dir.unsqueeze(2).expand(coords.shape)
        return coords, view_dir, depth, deltas   

        #return 0


    def predict_radience(self, coords, view_dir):
        """Predict radiance at the given coordinates.
        TODO: You can adjust the network architecture according to your needs. You may also 
        try to use additional raydirections inputs to predict the radiance.

        Args:
            coords (torch.FloatTensor): 3D coordinates of the points of shape [..., 3].

        Returns:
            rgb (torch.FloatTensor): Radiance at the given coordinates of shape [..., 3].
            sigma (torch.FloatTensor): volume density at the given coordinates of shape [..., 1].

        """
        if len(coords.shape) == 2:
            coords = self.pos_enc(coords)
        else:
            input_shape = coords.shape
            coords = self.pos_enc(coords.view(-1, 3)).view(*input_shape[:-1], -1)

        if view_dir is not None:
            input_shape = view_dir.shape
            view_dir = self.pos_enc(view_dir.reshape(-1, 3)).view(*input_shape[:-1], -1)

        if self.cfg.network_type == "skeleton":
            pred = self.mlp(coords)
            rgb = torch.sigmoid(pred[..., :3])
            sigma = torch.relu(pred[..., 3:])

        elif self.cfg.network_type == "updated_skeleton":
            sigma, rgb = self.mlp(coords, view_dir)
            sigma = torch.relu(sigma)
            ##If view_dir is None, we are in mesh construction mode, so rgb is None already
            if rgb is None:
                return rgb, sigma

            rgb = torch.sigmoid(rgb)

        elif self.cfg.network_type == "updated_skeleton2":
            sigma, rgb = self.mlp(coords, view_dir)
            sigma = torch.relu(sigma)
            ##If view_dir is None, we are in mesh construction mode, so rgb is None already
            if rgb is None:
                return rgb, sigma

            rgb = torch.sigmoid(rgb)            

        elif self.cfg.network_type == "double_mlp":

            ##We only use the viewing direction for color rendering, this is passed as None for mesh construction
            if view_dir is None:
                sdf = self.mlp.get_sdf(coords)
                return None, sigma_density_function(sdf, 1e5)

            else:
                sdf, rgb, gradients = self.mlp(coords, view_dir, return_sdf_grad = True)
                self.gradients = gradients
                sigma = sdf
                rgb = torch.sigmoid(rgb)
            
              
        return rgb, sigma

    def volume_render(self, rgb, sigma, depth, deltas):
        """Ray marching to compute the radiance at the given rays.
        TODO: You are free to try out different neural rendering methods.
        
        Args:
            rgb (torch.FloatTensor): Radiance at the sampled points of shape [B, Nr, Np, 3].
            sigma (torch.FloatTensor): Volume density at the sampled points of shape [B, Nr, Np, 1].
            deltas (torch.FloatTensor): Distance between the points of shape [B, Nr, Np, 1].
        
        Returns:
            ray_colors (torch.FloatTensor): Radiance at the given rays of shape [B, Nr, 3].
            weights (torch.FloatTensor): Weights of the given rays of shape [B, Nr, 1].

        """

        if self.cfg.network_type == "double_mlp":
            ###Sigma is sdf in this case 
            sdf = sigma
            inv_s = self.mlp.inv_s(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)   # Single parameter
            # inv_s = inv_s.expand(batch_size * n_samples, 1)

            estimated_next_sdf = sdf + deltas * 0.5
            estimated_prev_sdf = sdf - deltas * 0.5

            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

            transmittance = torch.exp(cumsum( torch.log(1 - alpha), exclusive=True))
            weights = transmittance * alpha

            ray_colors = (weights * rgb).sum(dim=-2)
            ray_dapth = (depth* weights).sum(dim=-2)
            ray_alpha = weights.sum(dim=-2)

            
        else:
            # Sample points along the rays
            
            tau = sigma * deltas
            ray_colors, ray_dapth, ray_alpha = exponential_integration(rgb, tau, depth, exclusive=True)
        #ray alpha becomes nan

        return ray_colors, ray_dapth, ray_alpha


    def forward(self):
        """Forward pass of the network. 
        TODO: Adjust the neural rendering pipeline according to your needs.

        Returns:
            rgb (torch.FloatTensor): Ray codors of shape [B, Nr, 3].

        """
        B, Nr = self.ray_orig.shape[:2]

        # Step 1 : Sample points along the rays
        self.coords, self.view_dir, self.z_vals, self.deltas = self.sample_points(
                                self.ray_orig, self.ray_dir, hierarchical=self.cfg.hierarchical, near=self.cfg.near, far=self.cfg.far,
                                num_points_c=self.cfg.num_pts_per_ray_c, num_points_f=self.cfg.num_pts_per_ray_f)

        # Step 2 : Predict radiance and volume density at the sampled points
        self.rgb, self.sigma = self.predict_radience(self.coords, self.view_dir)
        # Step 3 : Volume rendering to compute the RGB color at the given rays
        self.ray_colors, self.ray_depth, self.ray_alpha = self.volume_render(self.rgb, self.sigma, self.z_vals, self.deltas)

        
        # Step 4 : Compositing with background color
        if self.cfg.bg_color == 'white':
            bg = torch.ones(B, Nr, 3, device=self.ray_colors.device)
            self.rgb = (1 - self.ray_alpha) * bg + self.ray_alpha * self.ray_colors
        else:
            self.rgb = self.ray_alpha * self.ray_colors
        



    def backward(self):
        """Backward pass of the network.
        TODO: You can also desgin your own loss function.
        """

        loss = 0.0
        rgb_loss = torch.abs(self.rgb -  self.img_gts).mean()
           

        loss = rgb_loss # + any other loss terms

        if self.cfg.use_eikonal_loss and self.cfg.network_type == "double_mlp" :
            eikonal_loss = torch.abs(self.sdf_gradient - 1).mean()
            loss += eikonal_loss
            self.log_dict['eikonal_loss'] += eikonal_loss.item()

        self.log_dict['rgb_loss'] += rgb_loss.item()
        self.log_dict['total_loss'] += loss.item()

        loss.backward()

    def step(self, data, epoch):
        """A signle training step.
        """

        # Get rays, and put them on the device
        self.ray_orig = data['rays'][..., :3].to(DEVICE)
        self.ray_dir = data['rays'][..., 3:].to(DEVICE)
        self.img_gts = data['imgs'].to(DEVICE)

        self.optimizer.zero_grad()
        threshold = 600
        threshold2 = 4100
        if epoch != 0 and epoch % threshold == 0 and epoch < threshold2:
            #print("threshold check edildi")
            denom = 2**(np.floor(epoch/threshold))
            lr = self.cfg.lr/denom
            if self.optimizer.param_groups[0]['lr'] != lr:
                #print("optimizer updatelendi")
                self.optimizer.param_groups[0]['lr'] = lr
        #denom = 2**(np.floor(epoch/threshold))
        #self.optimizer.param_groups[0]['lr'] = self.cfg.lr/denom

        #if epoch % 50 == 0:
        #    print("epoch: ",epoch," lr: ",self.optimizer.param_groups[0]['lr'])
            
        self.forward()
        self.backward()
        
        self.optimizer.step()
        self.log_dict['total_iter_count'] += 1
        self.log_dict['image_count'] += self.ray_orig.shape[0]

    def render(self, ray_orig, ray_dir):
        """Render a full image for evaluation.
        """
        B, Nr = ray_orig.shape[:2]
        coords, view_dir, depth, deltas = self.sample_points(ray_orig, ray_dir, hierarchical=self.cfg.hierarchical ,near=self.cfg.near, far=self.cfg.far,
                                num_points_c=self.cfg.num_pts_per_ray_c, num_points_f=self.cfg.num_pts_per_ray_f)
        
        rgb, sigma = self.predict_radience(coords, view_dir)
        #deltas normalde sifira yakinsiyor tam sifir olmuyor, bu sorun mu bilemem
        # bir noktada nan nerden geliyor?
        ray_colors, ray_depth, ray_alpha= self.volume_render(rgb, sigma, depth, deltas)  

        if self.cfg.bg_color == 'white':
            bg = torch.ones(B, Nr, 3, device=ray_colors.device)
            render_img = (1 - ray_alpha) * bg + ray_alpha * ray_colors
        else:
            render_img = ray_alpha * ray_colors
        return render_img, ray_depth, ray_alpha

    def reconstruct_3D(self, save_dir, epoch=0, sigma_threshold = 50., chunk_size=8192):
        """Reconstruct the 3D shape from the volume density.
        """

        # Mesh evaluation
        window_x = torch.linspace(-1., 1., steps=RES, device=DEVICE)
        window_y = torch.linspace(-1., 1., steps=RES, device=DEVICE)
        window_z = torch.linspace(-1., 1., steps=RES, device=DEVICE)
        
        coord = torch.stack(torch.meshgrid(window_x, window_y, window_z)).permute(1, 2, 3, 0).reshape(-1, 3).contiguous()

        _points = torch.split(coord, int(chunk_size), dim=0)
        voxels = []
        for _p in _points:
            _, sigma = self.predict_radience(_p, None) 
            voxels.append(sigma)
        voxels = torch.cat(voxels, dim=0)

        np_sigma = torch.clip(voxels, 0.0).reshape(RES, RES, RES).cpu().numpy()

        vertices, faces = mcubes.marching_cubes(np_sigma, sigma_threshold)
        #vertices = ((vertices - 0.5) / (res/2)) - 1.0
        vertices = (vertices / (RES-1)) * 2.0 - 1.0

        h = trimesh.Trimesh(vertices=vertices, faces=faces)
        h.export(os.path.join(save_dir, '%04d.obj' % (epoch)))


    def log(self, step, epoch):
        """Log the training information.
        """
        log_text = 'STEP {} - EPOCH {}/{}'.format(step, epoch, self.cfg.epochs)
        self.log_dict['total_loss'] /= self.log_dict['total_iter_count']
        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['total_loss'])
        self.log_dict['rgb_loss'] /= self.log_dict['total_iter_count']
        log_text += ' | rgb loss: {:>.3E}'.format(self.log_dict['rgb_loss'])

        log.info(log_text)

        for key, value in self.log_dict.items():
            if 'loss' in key:
                wandb.log({key: value}, step=step)
        self.init_log_dict()

    def validate(self, loader, img_shape, step=0, epoch=0, sigma_threshold = 50., chunk_size=8192, save_img=False):
        """validation function for generating final results.
        """
        torch.cuda.empty_cache() # To avoid CUDA out of memory
        self.eval()

        log.info("Beginning validation...")
        log.info(f"Loaded validation dataset with {len(loader)} images at resolution {img_shape[0]}x{img_shape[1]}")


        self.valid_mesh_dir = os.path.join(self.log_dir, "mesh")
        log.info(f"Saving reconstruction result to {self.valid_mesh_dir}")
        if not os.path.exists(self.valid_mesh_dir):
            os.makedirs(self.valid_mesh_dir)

        if save_img:
            self.valid_img_dir = os.path.join(self.log_dir, "img")
            log.info(f"Saving rendering result to {self.valid_img_dir}")
            if not os.path.exists(self.valid_img_dir):
                os.makedirs(self.valid_img_dir)

        psnr_total = 0.0

        wandb_img = []
        wandb_img_gt = []


        with torch.no_grad() if self.cfg.network_type != 'double_mlp' else nullcontext() as ctx:
        
            # Evaluate 3D reconstruction
            self.reconstruct_3D(self.valid_mesh_dir, epoch=epoch,
                            sigma_threshold=sigma_threshold, chunk_size=chunk_size)

            # Evaluate 2D novel view rendering
            for i, data in enumerate(tqdm(loader)):
                rays = data['rays'].to(DEVICE)         # [1, Nr, 6]
                img_gt = data['imgs'].to(DEVICE)       # [1, Nr, 3]
                mask = data['masks'].repeat(1, 1, 3).to(DEVICE)

                _rays = torch.split(rays, int(chunk_size), dim=1)
                pixels = []
                for _r in _rays:
                    ray_orig = _r[..., :3]          # [1, chunk, 3]
                    ray_dir = _r[..., 3:]           # [1, chunk, 3]
                    ray_rgb, ray_depth, ray_alpha = self.render(ray_orig, ray_dir)
                    pixels.append(ray_rgb)

                pixels = torch.cat(pixels, dim=1)

                psnr_total += psnr(pixels, img_gt)

                img = (pixels).reshape(*img_shape, 3).cpu().numpy() * 255
                gt = (img_gt).reshape(*img_shape, 3).cpu().numpy() * 255
                wandb_img.append(wandb.Image(img))
                wandb_img_gt.append(wandb.Image(gt))

                if save_img:
                    Image.fromarray(gt.astype(np.uint8)).save(
                        os.path.join(self.valid_img_dir, "gt-{:04d}-{:03d}.png".format(epoch, i)) )
                    Image.fromarray(img.astype(np.uint8)).save(
                        os.path.join(self.valid_img_dir, "img-{:04d}-{:03d}.png".format(epoch, i)) )

        wandb.log({"Rendered Images": wandb_img}, step=step)
        wandb.log({"Ground-truth Images": wandb_img_gt}, step=step)
                
        psnr_total /= len(loader)

        log_text = 'EPOCH {}/{}'.format(epoch, self.cfg.epochs)
        log_text += ' {} | {:.2f}'.format(f"PSNR", psnr_total)

        wandb.log({'PSNR': psnr_total, 'Epoch': epoch}, step=step)
        log.info(log_text)
        self.train()

    def save_model(self, epoch):
        """Save the model checkpoint.
        """

        fname = os.path.join(self.log_dir, f'model-{epoch}.pth')
        log.info(f'Saving model checkpoint to: {fname}')
        torch.save(self.mlp, fname)

    