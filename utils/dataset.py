import os
import zipfile
import requests 
import pickle
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from dotenv import dotenv_values
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms

env_vars = dotenv_values(dotenv_path="./.env")
 
class KvasirDataset(Dataset):
    def __init__(self, 
                 data,
                 mode,
                 image_size= (512, 512),
                 mask_size = (128, 128)):
        self.data = data
        self.mode = mode
        self.image_size = image_size
        self.mask_size = mask_size

        self.base_transforms = transforms.Compose([
            transforms.Resize(self.image_size, transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        self.base_mask_transforms = transforms.Compose([
            transforms.Resize(self.mask_size, transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        if self.mode == 'train':
            self.aug_img_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(20, fill=0),
                transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.95, 1.05), fill=0), 
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)), 
                transforms.RandomGrayscale(p=0.2),
            ])

            self.aug_mask_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(20, fill=0),
                transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.95, 1.05), fill=0), 
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5)
            ])

    def _apply_transforms(self, image, mask):
        image = self.base_transforms(image)
        mask = self.base_mask_transforms(mask)

        if self.mode == 'train':
            seed = random.randint(0, 2**32)
            torch.manual_seed(seed)
            random.seed(seed)
            image = self.aug_img_transforms(image)
            torch.manual_seed(seed)
            random.seed(seed)
            mask = self.aug_mask_transforms(mask)

        image = self.normalize(image)
        mask = (mask > 0.5).float()
        return image, mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            image, mask = item[0], item[1]
            return self._apply_transforms(image, mask)

        except Exception as e:
            print(f"Error loading item {idx}: {str(e)}")