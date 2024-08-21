# SegRep: Mask-Supervised Learning for Segment Representation

SegRep is a Self-Supervised Learning method incorporated with 2 layers of masking method that filters:
  - Image Masking: Mask non-target pixels' value at input tensor level
  - Feature Masking: Mask non-target activations at feature map level

This dual masking strategy can enable extracted features to only consider context from target pixels, omit even the activation originated from masked value.

SegRep can be:  
  - Incorporate in the feature extraction steps in the process of SSL, limiting the model to generalize on target context defined by masks.
  - Incorporate with a trained/pretrained model's feature extraction step to extract segment representation.

## Where do mask come from?  

Mask can be obtain from either segmentation models or manual annotation, or what ever mechanism that label the pixel with a specific tag.  
e.g. Specific organ/tissue/cell within medical images, where they can be segmented with deep learning models or manual annotated with medical professionals.  

## What's the point of SegRep?

In the traditional feature extraction process of deep learning, every pixels within an image contributes to the feature activations. 
This could lead to problems like:
  - Contextual bias or spurious correlations in general imaging, e.g. classification of cows influenced by the green field in the background.
  - Lack of explainability of model's prediction, e.g. a group of pathological images classified as high risk of specific diagnosis, but couldn't explain what structure/tissue/cell contribute to the prediction.

SegRep's purpose is to extract the features of specific objects with arbitrary boundaries defined by the mask, enabling the exteaction of segment representation.

# Dependencies:  

torch == 1.12.0  
torchvision == 0.13.0  
pytorch_lightning == 1.5.4  
numpy == 1.22.3  
PIL.Image == 9.5.0  
lightly == 1.4.1  
lightning-flash == 0.8.1.post0   
joblib == 1.2.0  
tqdm  

# Usage
## segrep_training.py
This script trains a SegRep-SSL model under the framework of pytorch-lightning.  
ResNet 18, TiCo Loss, LARS optimizer is used in this script;   
if you want to use other backbone, loss or optimizer, you'll have to modify the script.  
Use space to seperate multiple input variables

```
python segrep_training.py  
```

| Input Variables           | Description                                                                            |
| ------------------------- | -------------------------------------------------------------------------------------- |
| `--data_path`             | Path to the dataset directory                                                          |
| `--log_dir`               | Directory to save `.ckpt` files                                                        |
| `--no_mask`               | Enable training script to proceed without masking, original SSL so to speak            |
| `--crop_size`             | Input size for cropping transformation                                                 |
| `--batch_size`            | The desired batch size                                                                 |
| `--lars_lr`               | Set the learning rate for LARS optimizer                                               |
| `--epochs`                | The number of epochs for training                                                      |
| `--num_workers`           | Number of workers to use                                                               |
| `--accelerator`           | Type of accelerator being used                                                         |
| `--devices`               | Specific device(s) to use                                                              |
| `--one_device`            | Adjust several settings involved in ddp training                                       |
| `--debug`                 | Enable debug mode to inspect dataset output                                            |
| `--resume_checkpoint`     | Resume training from a previously saved checkpoint                                     |

`--data_path`:  
Specify the directory to your dataset, use wildcard to specify the subdirectories if they exist.
Use wildcard to specify all the file under the directory(subdirectories), format should also be specify, 
e.g.  
```
python segrep_training.py --data_path /work/project/image_dir/*/*.png
```

`--log_dir`:   
Directory you want to save your checkpoint file(.ckpt).  
(Default: `./logs`)
```
python segrep.training.py --log_dir /work/project/log_dir
```

`--no_mask`:  
Add this command if you want to train an original SSL for reference (no masking).
```
python segrep.training.py --no_mask
```

`--crop_size`:  
Specify the size of your input image for random cropping transformation, e.g. 512 for 512x512 size image.  
(Default; `512`)
```
python segrep_training.py --crop_size 256
```

`--batch_size`:  
Specify your designated batch size for your training.  
(Default: `32`)
```
python segrep_training.py --batch_size 16
```

`--lars_lr`:  
Specify your learning rate for LARS optimizer. 
reference: LARS for large batch (Zhu, 2022); lr = 0.3*batch size/256 (Ciga, 2022; Stacke, 2022)  
(Default: `1.2`)
```
python segrep_training.py --lars_lr 1.2
```

`--epochs`:  
Specify your designated maximum training epochs.  
(Default: `23`)  
```
python segrep_training.py --epochs 100
```  

`--num_workers`:  
Specify your designated number of workers.  
(Default: `16`)
```
python segrep_training.py --num_workers 8
```

`--accelerator`:  
Specify the accelerator you want to use; cpu, gpu, tpu, ipu.  
(Default: `gpu`)
```
python segrep_training.py --accelerator gpu
```

`--devices`:  
Specify the devices you want to use. (reference: https://lightning.ai/docs/pytorch/stable/accelerators/gpu_basic.html)  
(Default: `[0, 1, 2]`)
```
python segrep_training.py --devices [0, 1, 2]
```

`--one_device`:  
Add this command if you are training with a single device, this line deal with settings of trainer strategy, batchnorm syncing and gather distributed of loss function.  
If you wish to mannually modify these setting, please modify the code directly.
```
python segrep_training.py --one_device
```


`--grad_accumulate`:  
Accumulate n batches of gradient before optimization step, specify 1 if no gradient accumulation needed.
(Default: `32`)
```
python segrep_training.py --grad_accumulate 16
```

`--debug`:  
Debug function for dataset output, save the output images and mask to confirm if dataset output is working properly.
Automatically exit the code when outputs are saved.
```
python segrep_training.py --debug
```

`--resume_checkpoint`:  
Resume a previous trained state with a specified path to the checkpoint file.
```
python segrep_training.py --resume_checkpoint ./logs/epoch=15.ckpt
```

## segrep_inference.py
This script inference feature representations with trained encoder model.  

```
python segrep_inference.py  
```

| Input Variables           | Description                                                                            |
| ------------------------- | -------------------------------------------------------------------------------------- |
| `--data_path`             | Path to the dataset directory                                                          |
| `--no_mask`               | Enable training script to proceed without masking, original SSL so to speak            |
| `--num_workers`           | Number of workers to use                                                               |
| `--device`                | Specify 1 device to inference                                                          |
| `--checkpoint_dir`        | dir of checkpoint file of trained model                                                |
  
`--data_path`:  
Specify the directory to your dataset, use wildcard to specify the subdirectories if they exist.
Use wildcard to specify all the file under the directory(subdirectories), format should also be specify, 
e.g.  
```
python segrep_inference.py --data_path /work/project/image_dir/*/*.png
```

`--no_mask`:  
Add this command if you want to inference with original encoder(no masking).
```
python segrep.inference.py --no_mask
```

`--num_workers`:  
Specify your designated number of workers.  
(Default: `16`)
```
python segrep_inference.py --num_workers 8
```

`--device`:  
Specify the device you want to use.
(Default: `cuda:0`)
```
python segrep_inference.py --device cuda:0
```

`--checkpoint_dir`:  
Specify the checkpoint directory to load for your encoder.
```
python segrep_inference.py --checkpoint_dir ./logs/epoch=15.ckpt
```
