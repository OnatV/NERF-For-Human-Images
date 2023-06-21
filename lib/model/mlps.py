import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from contextlib import nullcontext

class MLP(nn.Module):
    """Super basic but super useful MLP class.
    """
    def __init__(self, 
        input_dim, 
        output_dim, 
        activation = torch.relu,
        bias = True,
        layer = nn.Linear,
        num_layers = 4, 
        hidden_dim = 128, 
        skip       = [2]
    ):
        """Initialize the MLP.

        Args:
            input_dim (int): Input dimension of the MLP.
            output_dim (int): Output dimension of the MLP.
            activation (function): The activation function to use.
            bias (bool): If True, use bias.
            layer (nn.Module): The MLP layer module to use.
            num_layers (int): The number of hidden layers in the MLP.
            hidden_dim (int): The hidden dimension of the MLP.
            skip (List[int]): List of layer indices where the input dimension is concatenated.

        Returns:
            (void): Initializes the class.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim        
        self.activation = activation
        self.bias = bias
        self.layer = layer
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skip = skip
        if self.skip is None:
            self.skip = []
        
        self.make()

    def make(self):
        """Builds the actual MLP.
        """
        layers = []
        for i in range(self.num_layers):
            if i == 0: 
                layers.append(self.layer(self.input_dim, self.hidden_dim, bias=self.bias))
            elif i in self.skip:
                layers.append(self.layer(self.hidden_dim+self.input_dim, self.hidden_dim, bias=self.bias))
            else:
                layers.append(self.layer(self.hidden_dim, self.hidden_dim, bias=self.bias))
        self.layers = nn.ModuleList(layers)
        self.lout = self.layer(self.hidden_dim, self.output_dim, bias=self.bias)

    def forward(self, x, return_h=False):
        """Run the MLP!

        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, ..., input_dim]
            return_h (bool): If True, also returns the last hidden layer.

        Returns:
            (torch.FloatTensor, (optional) torch.FloatTensor):
                - The output tensor of shape [batch, ..., output_dim]
                - The last hidden layer of shape [batch, ..., hidden_dim]
        """
        N = x.shape[0]

        for i, l in enumerate(self.layers):
            if i == 0:
                h = self.activation(l(x))
            elif i in self.skip:
                h = torch.cat([x, h], dim=-1)
                h = self.activation(l(h))
            else:
                h = self.activation(l(h))
        
        out = self.lout(h)
        
        if return_h:
            return out, h
        else:
            return out
        
class MLP2(nn.Module):
    """Updated Super basic but super useful MLP class.
    """
    def __init__(self, 
        input_dim, 
        output_dim = 3, 
        activation = torch.relu,
        bias = True,
        layer = nn.Linear,
        num_layers = 9, 
        hidden_dim = 256, 
        skip       = [4],
        sigma_out_layer = 7,
        sigma_dim = 1,
    ):
        """Initialize the MLP.

        Args:
            input_dim (int): Input dimension of the MLP.
            output_dim (int): Output dimension of the MLP.
            activation (function): The activation function to use.
            bias (bool): If True, use bias.
            layer (nn.Module): The MLP layer module to use.
            num_layers (int): The number of hidden layers in the MLP.
            hidden_dim (int): The hidden dimension of the MLP.
            skip (List[int]): List of layer indices where the input dimension is concatenated.

        Returns:
            (void): Initializes the class.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim        
        self.activation = activation
        self.bias = bias
        self.layer = layer
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skip = skip

        if self.skip is None:
            self.skip = []
        
        self.sigma_out_layer = sigma_out_layer
        self.sigma_dim = sigma_dim
        
        self.make()

    def make(self):
        """Builds the actual MLP.
        """
        layers = []
        for i in range(self.num_layers):
            if i == 0: 
                layers.append(self.layer(self.input_dim, self.hidden_dim, bias=self.bias))
            elif i == (self.sigma_out_layer + 1):
                layers.append(self.layer(self.hidden_dim+ self.input_dim + self.input_dim, self.hidden_dim, bias=self.bias))
            elif i in self.skip:
                layers.append(self.layer(self.hidden_dim+self.input_dim, self.hidden_dim, bias=self.bias))
            else:
                layers.append(self.layer(self.hidden_dim, self.hidden_dim, bias=self.bias))
        self.layers = nn.ModuleList(layers)

        self.rgbout = self.layer(self.hidden_dim, self.output_dim, bias=self.bias)
        self.sigmaout = self.layer(self.hidden_dim, self.sigma_dim, bias=self.bias)

    def forward(self, coords, view_dir, return_h=False):
        """Run the MLP!

        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, ..., input_dim]
            return_h (bool): If True, also returns the last hidden layer.

        Returns:
            (torch.FloatTensor, (optional) torch.FloatTensor):
                - The output tensor of shape [batch, ..., output_dim]
                - The last hidden layer of shape [batch, ..., hidden_dim]
        """

        for i, l in enumerate(self.layers):
            if i == 0:
                h = self.activation(l(coords))
            elif i == (self.sigma_out_layer + 1):
                h = torch.cat([coords, view_dir, h], dim=-1)
                h = self.activation(l(h))
            elif i in self.skip:
                h = torch.cat([coords, h], dim=-1)
                h = self.activation(l(h))
            else:
                h = self.activation(l(h))

            if i == self.sigma_out_layer:
                sigma = self.sigmaout(h)
                if view_dir is None:
                    return sigma , None

        rgb = self.rgbout(h)
        
        if return_h:
            return sigma, rgb, h
        else:
            return sigma, rgb


class ResidualBlock(nn.Module):
    # Residual block with fc layers
    def __init__(self,input_dim=256,hidden_dim=256,output_dim=256,activation = torch.relu):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.l1 = nn.Linear(self.input_dim,self.hidden_dim)
        self.l2 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.l3 = nn.Linear(self.hidden_dim,self.output_dim)

    def forward(self, x):
        residual = x
        out = self.activation(self.l1(x))
        out = self.activation(self.l2(out))
        out = self.l3(out)
        out += residual
        out = self.activation(out)
        return out


class MLP3(nn.Module):
    """Updated Super basic with residual connections instead of skip.
    """
    def __init__(self, 
        input_dim, 
        output_dim = 3, 
        activation = torch.relu,
        bias = True,
        layer = nn.Linear,
        num_layers = 9, 
        hidden_dim = 256, 
        skip       = [4],
        sigma_out_layer = 7,
        sigma_dim = 1,
    ):

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim        
        self.activation = activation
        self.bias = bias
        self.layer = layer
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.sigma_dim = sigma_dim

        self.block = ResidualBlock(input_dim=self.hidden_dim,hidden_dim=self.hidden_dim,output_dim=self.hidden_dim
            ,activation = self.activation)

        self.l1 = self.layer(self.input_dim,self.hidden_dim)
        self.l2 = self.block
        self.l3 = self.layer(self.hidden_dim,self.hidden_dim)
        self.ls = self.layer(self.hidden_dim+self.input_dim,self.hidden_dim)
        self.l4 = self.block
        self.l5 = self.layer(self.hidden_dim,self.hidden_dim)
        self.l6 = self.layer(self.hidden_dim+ self.input_dim + self.input_dim, self.hidden_dim)

        self.rgbout = self.layer(self.hidden_dim, self.output_dim, bias=self.bias)
        self.sigmaout = self.layer(self.hidden_dim, self.sigma_dim, bias=self.bias)
        
    def forward(self, coords, view_dir, return_h=False):
        h = self.activation(self.l1(coords))
        h = self.l2(h)
        h = self.activation(self.l3(h))
        h = torch.cat([coords, h], dim=-1)
        h = self.ls(h)
        h = self.l4(h)
        h = self.activation(self.l5(h))
        sigma = self.sigmaout(h)
        if view_dir is None:
            return sigma , None
        #print("coords: ",coords.shape)
        #print("view_dir: ",view_dir.shape)
        #print("h: ",h.shape)
        h = torch.cat([coords, view_dir, h], dim=-1)
        h = self.activation(self.l6(h))
        rgb = self.rgbout(h)

        if return_h:
            return sigma, rgb, h
        else:
            return sigma, rgb


def get_activation_class(activation_type):
    """Utility function to return an activation function class based on the string description.

    Args:
        activation_type (str): The name for the activation function.
    
    Returns:
        (Function): The activation function to be used. 
    """
    if activation_type == 'relu':
        return torch.relu
    elif activation_type == 'sin':
        return torch.sin
    elif activation_type == 'softplus':
        return torch.nn.functional.softplus
    elif activation_type == 'lrelu':
        return torch.nn.functional.leaky_relu
    else:
        assert False and "activation type does not exist"


class DoubleMLP(nn.Module):

    def __init__(self,  
            input_dim, 
            output_dim = 3, 
            bias = True,
            layer = nn.Linear,
            num_layers = 6, 
            hidden_dim = 128, 
            skip       = [2,5],
            sigma_out_layer = 4,
            sigma_dim = 257):
        
        """Initialize the MLP.
        """
        super().__init__()

        self.input_dim = input_dim ## Coordinates + Positional Encoding
        self.sigma_dim = sigma_dim
        self.hidden_dim = hidden_dim

        # ( Coordinates + Positional Encoding) + (Viewing direction + Positional Encoding = input_dim) + (sigma_dim) + (sigma_gradient_dim = Coordinates + Positional Encoding)
        self.rgb_input_dim = sigma_dim + 3 * input_dim 

        self.sdf = SDFMLP(input_dim=input_dim,  output_dim = self.sigma_dim, num_layers=sigma_out_layer, hidden_dim=self.hidden_dim )
        self.rgb = RGBMLP(input_dim = self.rgb_input_dim, output_dim= output_dim, num_layers=num_layers - sigma_out_layer, hidden_dim=self.hidden_dim)
        self.inv_s   = SingleVarianceNetwork(init_val = 0.3)
    
    def get_sdf(self, coords):
        sdf = self.sdf(coords, return_f=False)
        return sdf

    def forward(self, coords, view_dir, return_sdf_grad =False, return_h=False):
        """Run the MLP!

        Args:
            coords: Point coordinates, with positinal encoding of shape [batch, ..., input_dim]
            view_dir: Viewing directions, with positinal encoding of shape [batch, ..., input_dim]
            
        returns:
            sigma:
            rgb:
        """

        coords.requires_grad_(True)

        sdf_with_features = self.sdf(coords, return_f=True)
        sdf = sdf_with_features[..., 0]
        
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=coords,
            grad_outputs=torch.ones_like(sdf, requires_grad=False, device=coords.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        
        
        rgb_input= torch.cat([coords, view_dir, sdf_with_features, gradients], dim=-1)
        rgb = self.rgb(rgb_input)
        sdf = sdf.unsqueeze(-1)

        if return_sdf_grad:
            return sdf, rgb, gradients

        return sdf, rgb


class SDFMLP(nn.Module):


    def __init__(self, 
        input_dim, 
        output_dim = 257, 
        activation_func = partial(torch.nn.functional.softplus, beta=100),
        bias = True,
        layer = nn.Linear,
        num_layers = 8, 
        hidden_dim = 256, 
        skip       = [5]
    ):
        '''
            Args:
                input_dim (): Input dimension of the MLP. 3 + positional encoding
                output_dim (): Output dimension of the MLP. First dimension is sdf and the remaining are random feature 1 + 256 = 257
                activation (function): The activation function to use.
                bias (bool): If True, use bias.
                layer (nn.Module): The MLP layer module to use.
                num_layers (int): The number of hidden layers in the MLP.
                hidden_dim (int): The hidden dimension of the MLP.
                skip (List[int]): List of layer indices where the input dimension is concatenated.
        '''
    

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim # 1 + 256 = 257
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.activation = activation_func

        self.bias = bias
        self.skip = skip
        self.layer = layer
        self.__make()

    def __make(self):

        layers = []
        for i in range(self.num_layers):
            if i == 0: 
                layers.append(self.layer(self.input_dim, self.hidden_dim, bias=self.bias))
            elif i in self.skip:
                layers.append(self.layer(self.hidden_dim+self.input_dim, self.hidden_dim, bias=self.bias))
            else:
                layers.append(self.layer(self.hidden_dim, self.hidden_dim, bias=self.bias))
        self.layers = nn.ModuleList(layers)
        self.lout = self.layer(self.hidden_dim, self.output_dim, bias=self.bias)
    
    def forward(self, x, return_f=False):
        """
        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, ..., input_dim]
            return_h (bool): If True, also returns the last hidden layer.

        Returns:
            (torch.FloatTensor, (optional) torch.FloatTensor):
                - The output tensor of shape [batch, ..., output_dim]
                - The last hidden layer of shape [batch, ..., hidden_dim]
        """
        N = x.shape[0]

        for i, l in enumerate(self.layers):
            if i == 0:
                h = self.activation(l(x))
            elif i in self.skip:
                h = torch.cat([x, h], dim=-1)
                h = self.activation(l(h))
            else:
                h = self.activation(l(h))
        
        out = self.lout(h)
        
        if return_f:
            return out
        else:
            return out[...,0]
        
class RGBMLP(nn.Module):


    def __init__(self, 
        input_dim, 
        output_dim, 
        activation_func = partial(torch.nn.functional.softplus, beta=100),
        bias = True,
        layer = nn.Linear,
        num_layers = 4, 
        hidden_dim = 256, 
        skip       = [2]
    ):
     
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.activation = activation_func

        self.bias = bias
        self.skip = skip
        self.layer = layer
        self.__make()

    def __make(self):

        layers = []
        for i in range(self.num_layers):
            if i == 0: 
                layers.append(self.layer(self.input_dim, self.hidden_dim, bias=self.bias))
            elif i in self.skip:
                layers.append(self.layer(self.hidden_dim+self.input_dim, self.hidden_dim, bias=self.bias))
            else:
                layers.append(self.layer(self.hidden_dim, self.hidden_dim, bias=self.bias))
        self.layers = nn.ModuleList(layers)
        self.lout = self.layer(self.hidden_dim, self.output_dim, bias=self.bias)
    
    def forward(self, x, return_h=False):
        """
        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, ..., input_dim]
            return_h (bool): If True, also returns the last hidden layer.

        Returns:
            (torch.FloatTensor, (optional) torch.FloatTensor):
                - The output tensor of shape [batch, ..., output_dim]
                - The last hidden layer of shape [batch, ..., hidden_dim]
        """
        N = x.shape[0]

        for i, l in enumerate(self.layers):
            if i == 0:
                h = self.activation(l(x))
            elif i in self.skip:
                h = torch.cat([x, h], dim=-1)
                h = self.activation(l(h))
            else:
                h = self.activation(l(h))
        
        out = self.lout(h) #The activation is suggested in paper
        
        if return_h:
            return out, h
        else:
            return out


##Borrowed from Neus Rendering Paper
class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)