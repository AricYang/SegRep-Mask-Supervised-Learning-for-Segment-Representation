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

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from lightly.loss.tico_loss import TiCoLoss
from lightly.models.modules.heads import TiCoProjectionHead
from tqdm import tqdm
import joblib


def parse_args():
    parser = argparse.ArgumentParser(description='Inferencing script for SegRep')

    parser.add_argument('--data_path', type=str, 
                        default='directory to your image dataset',
                        help='Path to the data directory')
    parser.add_argument('--no_mask', action='store_true',
                        help='Enable training script to proceed without masking, original SSL so to speak')
    parser.add_argument('--checkpoint_dir', type=str, 
                        default='directory of your trained checkpoint',
                        help='Directory to save logs')
    parser.add_argument('--device', type=str, 
                        default='cuda:0',
                        help='inferencing device you are using')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of workers for data loading (default: 16)')
    
    
    return parser.parse_args()

class Dataset(BaseDataset):
    
    def __init__(self, 
                 image_dir,
                 no_mask,
                 preprocessing_image = None,
                ):
        self.image_dir = image_dir
        self.no_mask = no_mask
        self.preprocessing_image = preprocessing_image
        
    def __len__(self):
        return len(self.image_dir)
    
    def __getitem__(self, idx):
        filepath = self.image_dir[idx]
        if self.no_mask:
            rgb_img = Image.open(self.image_dir[idx]).convert('RGB')
            image = self.preprocessing_image(rgb_img)
            return image, 0, filepath
        else:
            rgba_img = Image.open(self.image_dir[idx])
            if (np.asarray(rgba_img)).shape[2] < 4:
                raise ValueError(f"Expect image shape is (H, W, 4), got {np.asarray(rgba_img).shape} instead.")
            image = rgba_img.convert('RGB')
            image = self.preprocessing_image(image)
            mask = np.asarray(rgba_img, dtype = 'float32')[:,:,3]/255
            mask = T.ToTensor()(mask)
            image = image * mask
    
            return (image,mask), 0, filepath



class Encoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.projection_head = TiCoProjectionHead(512, 512, 128)
        self.criterion = TiCoLoss(gather_distributed=True)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)

    
    def segrep_forward(self, x, y):
        x = self.backbone(x)
        for i in range(5):
            y = self.avgpool(y)
        x = (x*y).sum([2,3])
        x = F.normalize(x, p=2, dim=1)
        
        return x

    
    def ori_forward(self, x):    
        x = self.backbone(x)
        x = x.sum([2,3])
        x = F.normalize(x, p=2, dim=1)

        return x


if __name__ == "__main__":
    args = parse_args()
    normalize: dict = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    transform_image = torchvision.transforms.Compose([
        T.ToTensor(),
        T.Normalize(mean=normalize["mean"], std=normalize["std"])
    ])

    # Modify it if subdirectory exists, e.g. sorted(glob.glob(f'{args.data_path}/*/*.png'))
    image_dir = sorted(glob.glob('{args.data_path}'))

    dataset = Dataset(image_dir,
                      no_mask = args.no_mask,
                      preprocessing_image = transform_image, 
                     )

    checkpoint = args.checkpoint_dir

    try:
        model = Encoder().load_from_checkpoint(checkpoint)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        exit(1)
    
    device = torch.device(args.device)
    model = model.to(device)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Inference the representations
    embeddings = []
    model.eval()
    with torch.no_grad():
        if args.no_mask:
            print('Start inferencing with no masking...')
            for img, _, _ in tqdm(dataloader):
                img = img.to(device)
                emb = model.ori_forward(img)
                embeddings.append(emb.cpu().detach())
        else:
            print('Start inferencing with SegRep...')
            for (img, mask), _, _ in tqdm(dataloader):
                img = img.to(device)
                mask = mask.to(device)
                emb = model.segrep_forward(img,mask)
                embeddings.append(emb.cpu().detach())
        embeddings = torch.cat(embeddings, 0)
        rep = embeddings.numpy()

    # Save the representations as .pkl file
    try:
        joblib.dump(rep, 'inferenced_reps.pkl')
        print("Embeddings saved successfully.")
    except Exception as e:
        print(f"Error saving embeddings: {e}")
