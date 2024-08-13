#!/usr/bin/env python
# coding: utf-8

import argparse
import torch
from torch import nn
import torchvision
import pytorch_lightning as pl
import pytorch_lightning.utilities
import torchvision.transforms as T
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import glob
from PIL import Image
import copy
import joblib

from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.rotation import random_rotation_transform
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from lightly.loss.tico_loss import TiCoLoss
from lightly.models.modules.heads import TiCoProjectionHead
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import flash
from tqdm import tqdm


class Dataset(BaseDataset):
    
    def __init__(self, 
                 image_dir,
                 preprocessing_color = None,
                 preprocessing_position = None,
                ):
        self.image_dir = image_dir
        self.preprocessing_color = preprocessing_color
        self.preprocessing_position = preprocessing_position
        
    def __len__(self):
        return len(self.image_dir)
    
    def __getitem__(self, idx):
        filepath = self.image_dir[idx]

        # Input image should be in shape of (R, G, B, alpha), where alpha channel stores mask information (target: 255, non-target: 0)
        rgba_img = Image.open(self.image_dir[idx])

        # RGBA image undergoes positional transformation first to align the status of crop, rotation and flip between image and mask
        rgba_img_v1 = self.preprocessing_position(rgba_img)
        
        # Extracts rgb value as image and undergoes color transformation
        image_v1 = rgba_img_v1.convert('RGB')
        image_v1 = self.preprocessing_color(image_v1)
        
        # Extract alpha value as mask and scale from 0-255 to 0-1
        mask_v1 = np.asarray(rgba_img_v1, dtype = 'float32')[:,:,3]/255
        mask_v1 = T.ToTensor()(mask_v1)
        
        # Image Masking(IM): Element-wise multiplication of image and mask, reduce the non-target value in image tensor to 0 and preseve the target value
        image_v1 = image_v1*mask_v1

        # Create 2nd view for contrastive learning
        rgba_img_v2 = self.preprocessing_position(rgba_img)
        image_v2 = rgba_img_v2.convert('RGB')
        image_v2 = self.preprocessing_color(image_v2)
        mask_v2 = np.asarray(rgba_img_v2, dtype = 'float32')[:,:,3]/255
        mask_v2 = T.ToTensor()(mask_v2)
        image_v2 = image_v2*mask_v2

        return ((image_v1, image_v2), (mask_v1, mask_v2)), 0, filepath


class SegRep_TiCo(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()

        # Truncate FC and GAP layer in the network
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.projection_head = TiCoProjectionHead(512, 512, 128)
        self.criterion = TiCoLoss(gather_distributed=True)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        
    
    def forward(self, x, y):
        # Pass image tensor through the network, generate feature maps. Shape: (3, H, W) -> (512, H/(2**5), W/(2**5)) 
        x = self.backbone(x)
        
        # Downsize the mask tensor 5 times to match the size of feature maps. Shape (1, H, W) -> (1, H/(2**5), W/(2**5))
        for i in range(5):
            y = self.avgpool(y)
            
        # Feature Masking(FM): Element-wise multiplication of feature maps and downsized mask, 
        # reduce the non-target activations in feature maps and preseve the target activations.
        # Globally sum each feature map into a single feature value. Shape (512, H/(2**5), W/(2**5)) -> (512, 1, 1)
        x = (x*y).sum([2,3])

        # L2 normalization the vector (Euclidean distance to 1)
        x = F.normalize(x, p=2, dim=1)
        
        return x

    def training_step(self, batch, batch_index):
        ((x0,x1),(y0,y1)) = batch[0]
        
        x0 = self.forward(x0, y0)
        z0 = self.projection_head(x0)
        
        x1 = self.forward(x1, y1)
        z1 = self.projection_head(x1)
        
        loss = self.criterion(z0, z1)
        return loss

    def configure_optimizers(self):
        optimizer = flash.core.optimizers.LARS(self.parameters(), lr=1.2) # LARS for large batch (Zhu, 2022); lr = 0.3*batch size/256 (Ciga, 2022; Stacke, 2022) 
        return optimizer

color_jitter = T.ColorJitter(0.4, 0.4, 0.4, 0.2)
normalize: dict = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

transform_color = torchvision.transforms.Compose([
    T.RandomApply([color_jitter], p=0.5),
    T.RandomGrayscale(p=0.2),
    GaussianBlur(kernel_size=51, sigmas=(0.1,2.0), prob=0.5),
    T.ToTensor(),
    T.Normalize(mean=normalize["mean"], std=normalize["std"])
])

transform_position = torchvision.transforms.Compose([
    T.RandomResizedCrop(size=512, scale=(0.3, 1.0)),
    random_rotation_transform(rr_prob=0.5, rr_degrees = None),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
])


image_dir = sorted(glob.glob('/work/gd14/d14002/segrep/all_epi_patch/*/*.png'))

dataset = Dataset(image_dir, 
                  preprocessing_color = transform_color, 
                  preprocessing_position = transform_position
                 )


seed_everything(42, workers=True)



model = SegRep_TiCo()

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True,
    num_workers=16,
)

trainer = pl.Trainer(max_epochs=349, 
                     accelerator = 'gpu',
                     devices = [0, 1, 2],
                     strategy="ddp_find_unused_parameters_false",
                     default_root_dir=DEFAULT_ROOT_DIR,
                     sync_batchnorm=True,
                     deterministic=True,
                     accumulate_grad_batches=32)

trainer.fit(model=model, 
            train_dataloaders = dataloader,
            #ckpt_path=checkpoint
           )

