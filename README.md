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

# Usage (🚧README Under construction🚧)
## segrep_training.py
This script trains a SegRep-SSL model under the framework of pytorch-lightning.
ResNet 18, TiCo Loss, LARS optimizer is used in this script; if you want to use other backbone, loss or optimizer, you'll have to modify the script. 
'''
python segrep_training.py 
'''
| Input Variables| Description                           |
| -------------- | ------------------------------------- |
| --data_path      | Path to the dataset directory    |
| --log_dir      | Directory to save .ckpt file    |
| --no_mask      | Enable training script to proceed without masking, original SSL so to speak    |
| --crop_size      | Path to the dataset directory    |
| --batch_size      | Path to the dataset directory    |
| --epochs      | Path to the dataset directory    |
| --num_workers      |                              |
| --accelerator      | Path to the dataset directory    |
| --devices      | Path to the dataset directory    |
| --strategy      | Path to the dataset directory    |
| --no_sync_batchnorm      | Path to the dataset directory    |
| --no_gather_distributed      |
| --grad_accumulate      |
| --debug      |
| --resume_checkpoint      |
