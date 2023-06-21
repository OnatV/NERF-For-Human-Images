To run the code, first install requirements

```
pip install -r requirements.txt
```
```
python train.py --save-root ./final_checkpoints --data-root data/[public|private] --wandb
```
The above code uses the config.yml file in the code root location as default. 

```
python test.py --pretrained-root ./final_checkpoints/{chekpoint_no} --model-name model-4500.pth --data-root data/private
```
