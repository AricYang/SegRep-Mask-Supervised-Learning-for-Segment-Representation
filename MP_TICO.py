#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.rotation import random_rotation_transform

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from lightly.loss.tico_loss import TiCoLoss
from lightly.models.modules.heads import TiCoProjectionHead
from pytorch_lightning import seed_everything
import flash
from tqdm import tqdm


# In[ ]:


class Dataset(BaseDataset):
    
    def __init__(self, 
                 image_dir,
                 preprocessing_color = None,
                 preprocessing_position = None,
                ):
        self.image_dir = image_dir
        self.preprocessing_color = preprocessing_color
        self.preprocessing_position = preprocessing_position
        self.totensor = T.ToTensor()
        
    def __len__(self):
        return len(self.image_dir)

    
    def __getitem__(self, idx):
        filepath = self.image_dir[idx]
        
        # load image from directory
        rgba_img = Image.open(self.image_dir[idx])
        
        # Positional augmentations on rgba image
        rgba_img_v1 = self.preprocessing_position(rgba_img)
        
        # extract rgb value from rgba image
        image_v1 = rgba_img_v1.convert('RGB')
        
        # apply color quality augmentations on rgb image
        image_v1 = self.preprocessing_color(image_v1)
        
        # extract alpha value as mask value, normalize it from 0 and 255 to 0 and 1
        mask_v1 = np.asarray(rgba_img_v1, dtype = 'float32')[:,:,3]/255
        
        # convert mask value into tensor
        mask_v1 = self.totensor(mask_v1)
        
        # 2nd augmented view
        rgba_img_v2 = self.preprocessing_position(rgba_img)
        image_v2 = rgba_img_v2.convert('RGB')
        image_v2 = self.preprocessing_color(image_v2)
        mask_v2 = np.asarray(rgba_img_v2, dtype = 'float32')[:,:,3]/255
        mask_v2 = self.totensor(mask_v2)
        
        # tensor masking the rgb tensor
        image_v1 = image_v1*mask_v1
        image_v2 = image_v2*mask_v2

        return ((image_v1, image_v2),(mask_v1, mask_v2)), 0, filepath


# In[ ]:


color_jitter = T.ColorJitter(0.4, 0.4, 0.4, 0.2)
normalize: dict = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

# Adjust crop size to fit your image size
transform_position = torchvision.transforms.Compose([
    T.RandomResizedCrop(size=512, scale=(0.3, 1.0)),
    random_rotation_transform(rr_prob=0.5, rr_degrees = None),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
])

transform_color = torchvision.transforms.Compose([
    T.RandomApply([color_jitter], p=0.5),
    T.RandomGrayscale(p=0.2),
    GaussianBlur(kernel_size=51, sigmas=(0.1,2.0), prob=0.5),
    T.ToTensor(),
    T.Normalize(mean=normalize["mean"], std=normalize["std"])
])


# In[ ]:


# A list of your training image directories, image format is (R,G,B,A) with value range of 0-255
image_dir = sorted(glob.glob('/data/aric/work/2023_02_14/data/all_epi_patch/*/*.png'))


# In[ ]:


dataset = Dataset(image_dir, 
                  preprocessing_color = transform_color, 
                  preprocessing_position = transform_position
                 )


# In[ ]:


# seed for reproducibility
seed_everything(42, workers=True)

class TiCo_MP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        # if initialized with imagenet weight
        # resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        
        # truncate FC and GAP layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.projection_head = TiCoProjectionHead(512, 512, 128)
        self.criterion = TiCoLoss(gather_distributed=True)
        self.l2_norm = F.normalize
        
        # mask pooling layer
        avgpool = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.avgpool_5 = nn.Sequential(avgpool,
                                       avgpool,
                                       avgpool,
                                       avgpool,
                                       avgpool)
        
    def maskpool(self, y):
        y = self.avgpool_5(y)
        y = torch.round(y, decimals = 3)
        return y
    
    def forward(self, x):
        x = self.backbone(x)
        return x

    def training_step(self, batch, batch_index):
        ((x0,x1),(y0,y1)) = batch[0]
        
        y0 = self.maskpool(y0)
        y1 = self.maskpool(y1)
        
        x0 = self.forward(x0)
        # multiply feature map with pooled mask element-wise and sum feature-wise
        x0 = (x0*y0).sum([2,3])
        x0 = self.l2_norm(x0, p=2, dim=1)
        z0 = self.projection_head(x0)
        
        x1 = self.forward(x1)
        x1 = (x1*y1).sum([2,3])
        x1 = self.l2_norm(x1, p=2, dim=1)
        z1 = self.projection_head(x1)
        
        loss = self.criterion(z0, z1)
        return loss

    def configure_optimizers(self):
        optimizer = flash.core.optimizers.LARS(self.parameters(), lr=1.2) # LARS for large batch (Zhu, 2022); lr = 0.3*batch size/256 (Ciga, 2022; Stacke, 2022) 
        return optimizer


# In[ ]:


# load model
model = TiCo_MP()

# to load checkpoint parameters, use the following code, example path of checkpoint = glob.glob('./LOG_DIR/lightning_logs/version_0/checkpoints/*')[0]
# checkpoint will save once every epoch, when a training is interruptted or reached maximum epochs, the next training session's log will save in a new 'version_*' folder
# model = TiCo().load_from_checkpoint(checkpoint)


# In[ ]:


# Define your training settings 

BATCH_SIZE = 32
NUM_WORKERS = 16
EPOCHS = 23
DEVICES = [0,1,2]
LOG_DIR = 'desinated directory to save your log'
GRADIENT_ACCUMULATE = 32


# In[ ]:


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=NUM_WORKERS,
)

trainer = pl.Trainer(max_epochs=EPOCHS, 
                     accelerator ='gpu',
                     devices =DEVICES
                     strategy="ddp_find_unused_parameters_false",
                     default_root_dir=LOG_DIR,
                     sync_batchnorm=True,
                     deterministic=True,
                     accumulate_grad_batches=GRADIENT_ACCUMULATE)


# In[ ]:


trainer.fit(model=model, 
            train_dataloaders = dataloader,
            #ckpt_path=checkpoint
           )
# ckpt_path is for resuming interrupted training

