import os
import zipfile
import requests 
import pickle
import shutil
import torch
import random
import string
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from dotenv import dotenv_values
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms


def create_fresh_directory(folder_path):
    folder_path = Path(folder_path)
    if folder_path.exists():
        shutil.rmtree(folder_path)
    folder_path.mkdir(parents=True)

def check_kvasir_dataset(dataset_path, subset_folder_path, output_folder_path):
    folder = ["images", "masks", "kavsir_bboxes.json"]
    folder_counter= 0
    
    if dataset_path.exists():
        for file in os.listdir(dataset_path):
            if file in folder:
                folder_counter+=1

        if (folder_counter==len(folder) and 
            len(os.listdir(f"{dataset_path}/{folder[0]}"))==1000 and 
            len(os.listdir(f"{dataset_path}/{folder[1]}"))==1000):
            return 1
        else:
            return 0
    else:
        create_fresh_directory(subset_folder_path)
        create_fresh_directory(output_folder_path)
        return 0

def download_dataset(
    url, 
    data_folder_path, 
    zfname, 
    dataset_name, 
    subset_folder_path, 
    output_folder_path):
    # check if dataset exixts
    data_dir = Path(data_folder_path)
    data_file = data_dir / zfname
    dataset_path = data_dir / dataset_name

    if check_kvasir_dataset(dataset_path, subset_folder_path, output_folder_path):
        print("Directory exists and not empty. Skipping download.")
    else:
        print("Downloading Dataset.")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # download the dataset
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024
        with data_file.open("wb") as file:
            for data in tqdm(response.iter_content(block_size), 
                            total=total_size//block_size,
                            unit="KiB",
                            unit_scale=True):
                file.write(data)

        # extract the zipfile and remove it
        with zipfile.ZipFile(data_file, "r") as zipdata:
            zipdata.extractall(data_dir)
        data_file.unlink()
    
### This needs changing
class SubsetCreator:
    def __init__(self, 
                 dataset_path,
                 subset_folder_path, 
                 n_subsets,
                 images_per_subset):
        self.dataset_path = Path(dataset_path)
        self.n_subsets = n_subsets
        self.images_per_subset = images_per_subset
        self.subset_folder_path = Path(subset_folder_path)
        self.subset_folder_path.mkdir(exist_ok=True)
        
    def create_subsets(self):
        images_dir = self.dataset_path / 'images'
        masks_dir = self.dataset_path / 'masks'
        
        image_paths = sorted(list(images_dir.glob('*.jpg')))
        
        random.shuffle(image_paths)
        
        for i in range(self.n_subsets):
            start_idx = i * self.images_per_subset
            end_idx = start_idx + self.images_per_subset
            subset_paths = image_paths[start_idx:end_idx]
            
            # Create subset data
            subset_data = []
            for img_path in subset_paths:
                mask_path = masks_dir / img_path.name
                if not mask_path.exists():
                    raise ValueError(f"Missing mask for {img_path}")
                    
                image = Image.open(img_path).convert("RGB")
                mask = Image.open(mask_path).convert("L")
                subset_data.append((image, mask))
            
            # Save subset
            subset_file = self.subset_folder_path / f'subset_{string.ascii_lowercase[i]}.pkl'
            with open(subset_file, 'wb') as f:
                pickle.dump(subset_data, f)
                
        return list(self.subset_folder_path.glob('*.pkl'))
    
    
def visualize_dataset(dataset, idx):
    image, mask = dataset[idx]

    image = image.permute(1, 2, 0).numpy() 
    mask = mask.squeeze().numpy()

    # Display
    fig, ax = plt.subplots(1, 2, figsize=(10,7))
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Mask")
    plt.show()