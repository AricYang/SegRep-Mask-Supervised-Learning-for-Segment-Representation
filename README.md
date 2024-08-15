# SegRep: Mask-Supervised Learning for Segment Representation

SegRep is a Self-Supervised Learning method incorporated with 2 layers of masking method that filters:
  - Image Masking: Mask non-target pixels' value at input tensor level
  - Feature Masking: Mask non-target activations at feature map level

This dual masking strategy can enable extracted features to only consider context from target pixels, omit even the activation originated from masked value.

SegRep's masking strategy can be incorporate with SSL, limiting the model's generalization to focus on target context defined by masks.  

# Where do mask come from?  

Mask can be obtain from either segmentation models or manual annotation, or what ever mechanism that label the pixel with a specific tag.  
e.g. Specific organ/tissue/cell within medical images, where they can be segmented with deep learning models or manual annotated with medical professionals.  


# Implemented modules:  

torch == 1.12.0  
torchvision == 0.13.0  
pytorch_lightning == 1.5.4  
numpy == 1.22.3  
PIL.Image == 9.5.0  
lightly == 1.4.1  
lightning-flash == 0.8.1.post0   
joblib == 1.2.0  
tqdm  
