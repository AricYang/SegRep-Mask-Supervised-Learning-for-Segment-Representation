#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from lightly.loss.tico_loss import TiCoLoss
from lightly.models.modules.heads import TiCoProjectionHead
import flash
from tqdm import tqdm
import joblib


# In[2]:


class Dataset(BaseDataset):
    
    def __init__(self, 
                 image_dir,
                 preprocessing_image = None,
                ):
        self.image_dir = image_dir
        self.preprocessing_image = preprocessing_image
        self.totensor = T.ToTensor()
        
    def __len__(self):
        return len(self.image_dir)

    
    def __getitem__(self, idx):
        filepath = self.image_dir[idx]
        
        rgba_img = Image.open(self.image_dir[idx])
        
        image = rgba_img.convert('RGB')
        image = self.preprocessing_image(image)
        mask = np.asarray(rgba_img, dtype = 'float32')[:,:,3]/255
        mask = self.totensor(mask)
        
        image = image * mask

        return (image, mask), 0, filepath


# In[3]:


normalize: dict = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

transform_image = torchvision.transforms.Compose([
    T.ToTensor(),
    T.Normalize(mean=normalize["mean"], std=normalize["std"])
])

transform_mask = torchvision.transforms.ToTensor()


# In[4]:


# A list of your image directories, image format is (R,G,B,A) with value range of 0-255
image_dir = sorted(glob.glob('/data/aric/work/2023_02_14/data/all_epi_patch/*/*.png'))


# In[5]:


dataset = Dataset(image_dir, 
                  preprocessing_image = transform_image, 
                 )


# In[11]:


class encoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        
        # truncate FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = TiCoProjectionHead(512, 512, 128)
        self.l2_norm = F.normalize

    
        # maskpool for representation
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


# In[12]:


# load your model ckpt file
checkpoint = glob.glob('./LOG_DIR/lightning_logs/version_0/checkpoints/*')[0]
checkpoint


# In[13]:


model = encoder().load_from_checkpoint(checkpoint)
device = torch.device('cuda:0')
model = model.to(device)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=16,
)


# In[ ]:


embeddings = []
model.eval()
with torch.no_grad():
    for (img, mask), _, _ in tqdm(dataloader):
        img = img.to(device)
        mask = mask.to(device)
        
        mask = model.maskpool(mask)
        emb = model.backbone[:-1](img)
        emb = (emb*mask).sum([2,3])
        emb = model.l2_norm(emb, p=2, dim=1)
        embeddings.append(emb.cpu().detach())

    embeddings = torch.cat(embeddings, 0)
    rep = embeddings.numpy()


# In[ ]:


joblib.dump(rep,'./encoded_reps.pkl')


# In[ ]:




