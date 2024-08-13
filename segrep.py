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
import random
import os

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

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for SegRep')

    parser.add_argument('--crop_size', type=int, default=512,
                        help='Input pixel size for RandomResizedCrop, e.g. 512 if 512x512')
    parser.add_argument('--data_path', type=str, 
                        default='directory to your image dataset',
                        help='Path to the data directory')
    parser.add_argument('--no_mask', action='store_true',
                        help='Enable training script to proceed without masking, original SSL so to speak')
    parser.add_argument('--log_dir', type=str, 
                        default='designated directory to save your log',
                        help='Directory to save logs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=23,
                        help='Number of epochs (default: 23)')
    parser.add_argument('--devices', nargs='+', type=int, default=[0, 1, 2],
                        help='List of GPU devices (default: [0, 1, 2])')
    parser.add_argument('--grad_accumulate', type=int, default=32,
                        help='Gradient accumulation (default: 32)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of workers for data loading (default: 16)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode to save images from the first dataset item')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to a checkpoint to resume training from (default: None)')
    
    
    return parser.parse_args()



class Dataset(BaseDataset):
    
    def __init__(self, 
                 image_dir,
                 no_mask,
                 preprocessing_color = None,
                 preprocessing_position = None,
                ):
        self.image_dir = image_dir
        self.no_mask = no_mask
        self.preprocessing_color = preprocessing_color
        self.preprocessing_position = preprocessing_position
        
    def __len__(self):
        return len(self.image_dir)
    
    def __getitem__(self, idx):
        filepath = self.image_dir[idx]
        
        if self.no_mask:
            rgb_img = Image.open(self.image_dir[idx]).convert('RGB')
            
            image_v1 = self.preprocessing_position(rgb_img)
            image_v1 = self.preprocessing_color(image_v1)
            image_v2 = self.preprocessing_position(rgb_img)
            image_v2 = self.preprocessing_color(image_v2)

            return (image_v1, image_v2), 0, filepath

        else:    
            # Input image should be in shape of (H, W, 4), where 4 value channels are(R, G, B, alpha)
            # The alpha channel stores mask information (target: 255, non-target: 0)
            rgba_img = Image.open(self.image_dir[idx])
            if np.asarray(rgba_img)).shape[2] < 4:
                raise ValueError(f"Expect image shape to be (H, W, 4), got {np.asarray(rgba_img)).shape} instead.")
    
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



class Ori_TiCo(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        
        # Truncate FC and GAP layer in the network
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.projection_head = TiCoProjectionHead(512, 512, 128)
        self.criterion = TiCoLoss(gather_distributed=True)
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.sum([2,3])
        x = F.normalize(x, p=2, dim=1)
        
        return x

    def training_step(self, batch, batch_index):
        (x0,x1) = batch[0]
        
        x0 = self.forward(x0)
        z0 = self.projection_head(x0)
        
        x1 = self.forward(x1)
        z1 = self.projection_head(x1)
        
        loss = self.criterion(z0, z1)
        return loss

    def configure_optimizers(self):
        optimizer = flash.core.optimizers.LARS(self.parameters(), lr=1.2) # LARS for large batch (Zhu, 2022); lr = 0.3*batch size/256 (Ciga, 2022; Stacke, 2022) 
        return optimizer



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



def unnormalize(tensor, mean, std):
    """
    unnormalize normalized Tensor
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m) 
    return tensor



if __name__ == "__main__":
    args = parse_args()
    color_jitter = T.ColorJitter(0.4, 0.4, 0.4, 0.2)
    normalize: dict = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]} # ImageNet norm

    transform_position = torchvision.transforms.Compose([
        T.RandomResizedCrop(size=args.crop_size, scale=(0.3, 1.0)),
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

    # Set checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.log_dir,
        filename='{epoch}',
        save_top_k=1,  # Save checkpoint every checkpoint
        every_n_epochs=1  # Save every 1 epoch
    )

    resume_checkpoint = None 

    
    # a list of directories to your training image, modify it if subdirectory exists, e.g. sorted(glob.glob(f'{args.data_path}/*/*.png'))
    image_files = sorted(glob.glob(f'{args.data_path}/*.png'))

    
    dataset = Dataset(image_files, 
                      args.no_mask,
                      preprocessing_color=transform_color, 
                      preprocessing_position=transform_position)

    if args.debug:
            
        # get the number of samples
        dataset_length = len(dataset)

        # generate random index 
        random_index = random.randint(0, dataset_length - 1)

        if args.no_mask:
            # get an item in the dataset
            (image_v1, image_v2), _, filepath = dataset[random_index]
            unnormalized_image_v1 = unnormalize(image_v1.clone(), normalize["mean"], normalize['std'])
            unnormalized_image_v2 = unnormalize(image_v2.clone(), normalize["mean"], normalize['std'])
            
            # save images
            torchvision.transforms.ToPILImage()(unnormalized_image_v1).save("image_v1.png")
            torchvision.transforms.ToPILImage()(unnormalized_image_v2).save("image_v2.png")

        else:
            ((image_v1, image_v2), (mask_v1, mask_v2)), _, filepath = dataset[random_index]
            unnormalized_image_v1 = unnormalize(image_v1.clone(), normalize["mean"], normalize['std'])
            unnormalized_image_v2 = unnormalize(image_v2.clone(), normalize["mean"], normalize['std'])
            torchvision.transforms.ToPILImage()(unnormalized_image_v1).save("image_v1.png")
            torchvision.transforms.ToPILImage()((mask_v1 * 255).to(torch.uint8)).save("mask_v1.png")
            torchvision.transforms.ToPILImage()(unnormalized_image_v2).save("image_v2.png")
            torchvision.transforms.ToPILImage()((mask_v2 * 255).to(torch.uint8)).save("mask_v2.png")

        print("input image",filepath)
        print("Debug images saved.")
        exit()         

    
    # set seef for reproducibility
    seed_everything(42, workers=True)
    
    if args.no_mask:
        model = Ori_TiCo()
    else:
        model = SegRep_TiCo()

    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )

    resume_checkpoint = args.resume_checkpoint
    if resume_checkpoint is not None and not os.path.isfile(resume_checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {resume_checkpoint}")

    
    trainer = pl.Trainer(max_epochs=args.epochs, 
                         accelerator = 'gpu',
                         devices = args.devices,
                         strategy="ddp_find_unused_parameters_false",
                         default_root_dir=args.log_dir,
                         sync_batchnorm=True,
                         deterministic=True,
                         callbacks=[checkpoint_callback],
                         resume_from_checkpoint=resume_checkpoint,
                         accumulate_grad_batches=args.grad_accumulate)

    trainer.fit(model=model, 
                train_dataloaders = dataloader,
            )
    trainer.save_checkpoint(os.path.join(args.log_dir, "final_model.ckpt"))
