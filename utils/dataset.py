import torch
import random
from dotenv import dotenv_values
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms

env_vars = dotenv_values(dotenv_path="./.env")

class GaussianNoise(object):
    def __init__(self, sigma_min=0.0, sigma_max=0.05, p=0.5):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p = p
    def __call__(self, x):
        if random.random() < self.p:
            sigma = random.uniform(self.sigma_min, self.sigma_max)
            noise = torch.randn_like(x) * sigma
            x = x + noise
            x.clamp_(0, 1)
        return x

class KvasirDataset(Dataset):
    def __init__(self, 
                 data,
                 mode,
                 image_size=(512, 512),
                 mask_size=(128, 128)):
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
            # your current/light training augs
            self.geom_img = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(20, fill=0),
                transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.95, 1.05), fill=0),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            ])
            self.geom_msk = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(20, fill=0),
                transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.95, 1.05), fill=0),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            ])
            self.img_only = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                transforms.RandomGrayscale(p=0.2),
            ])

        elif self.mode == 'oversample':
            # STRONGER augs for duplicated samples
            # (same geom ops/order for image & mask; bigger ranges)
            self.geom_img = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.6),
                transforms.RandomVerticalFlip(p=0.6),
                transforms.RandomRotation(35, fill=0),
                transforms.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.85, 1.15), shear=(-8, 8, -8, 8), fill=0),
                transforms.RandomPerspective(distortion_scale=0.35, p=0.7),
            ])
            self.geom_msk = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.6),
                transforms.RandomVerticalFlip(p=0.6),
                transforms.RandomRotation(35, fill=0),
                transforms.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.85, 1.15), shear=(-8, 8, -8, 8), fill=0),
                transforms.RandomPerspective(distortion_scale=0.35, p=0.7),
            ])
            self.img_only = transforms.Compose([
                transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.05),
                transforms.GaussianBlur(kernel_size=7, sigma=(0.2, 3.0)),
                transforms.RandomGrayscale(p=0.35),
                GaussianNoise(sigma_min=0.0, sigma_max=0.08, p=0.6),
                transforms.RandomErasing(p=0.4, scale=(0.01, 0.05), ratio=(0.3, 3.3), value=0),
            ])
        else:
            # val/test: no augs
            self.geom_img = None
            self.geom_msk = None
            self.img_only = None

    def _apply_transforms(self, image, mask):
        image = self.base_transforms(image)
        mask = self.base_mask_transforms(mask)

        if self.mode in ('train', 'oversample'):
            # lock geometry randomness across image/mask
            seed = random.randint(0, 2**32 - 1)
            torch.manual_seed(seed); random.seed(seed)
            image = self.geom_img(image)
            torch.manual_seed(seed); random.seed(seed)
            mask = self.geom_msk(mask)
            # image-only intensity transforms
            image = self.img_only(image)

        image = self.normalize(image)
        mask = (mask > 0.5).float()
        return image, mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image, mask = item[0], item[1]
        return self._apply_transforms(image, mask)
