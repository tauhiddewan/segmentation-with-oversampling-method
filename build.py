import os
import ast
import torch
import pickle
import numpy as np
from PIL import Image
from dotenv import dotenv_values
from torchvision import transforms
from sklearn.model_selection import train_test_split
from utils.build import download_dataset, SubsetCreator

def get_mask_area_ratio(mask_data):
    if isinstance(mask_data, torch.Tensor):
        mask = mask_data
    elif isinstance(mask_data, np.ndarray):
        mask = torch.tensor(mask_data)
    elif isinstance(mask_data, Image.Image):
        mask = torch.tensor(np.array(mask_data))
    else:
        raise TypeError(f"Unsupported mask type: {type(mask_data)}")

    foreground = (mask == 1).sum().item()
    return foreground / mask.numel()

def get_binwise_data(data, area_bins, mask_size):
    mask_transform = transforms.Compose([
        transforms.Resize(mask_size, transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    data_bins = [[] for _ in range(len(area_bins)-1)]
    for idx in range(len(data)):
        mask_data = mask_transform(data[idx][1])
        area_ratio = get_mask_area_ratio(mask_data=mask_data)
        for j in range(len(area_bins)-1):
            if area_ratio >= area_bins[j] and area_ratio < area_bins[j+1]:
                data_bins[j].append(data[idx])
    return data_bins

def get_balanced_train_test_split(data, n_bins, mask_size, test_split_size):
    train_dataset, test_dataset = [], []
    area_bins = np.linspace(0, 1, n_bins+1)
    data_bins = get_binwise_data(data, area_bins, mask_size)
        
    for j in range(len(data_bins)):
        if len(data_bins[j]) != 0:
            if len(data_bins[j]) > 1:
                train_data, test_data = train_test_split(data_bins[j], test_size=test_split_size, random_state=42)
                train_dataset.extend(train_data)
                test_dataset.extend(test_data)
    
    return train_dataset, test_dataset


if __name__=="__main__":
    n_subsets=10
    env_vars = dotenv_values(dotenv_path="./.env")
    download_dataset(
        url=env_vars["url"],
        data_folder_path=env_vars["data_folder_path"], 
        zfname=env_vars["dataset_zfname"],
        dataset_name=env_vars["dataset_name"],
        subset_folder_path=env_vars["subset_folder_path"], 
        output_folder_path=env_vars["output_folder_path"])
    
    subset_creator = SubsetCreator(
        dataset_path=f'{env_vars["data_folder_path"]}/{env_vars["dataset_name"]}',
        subset_folder_path = env_vars["subset_folder_path"],
        n_subsets=n_subsets, 
        images_per_subset=100
        )
    
    if len(os.listdir(f'{env_vars["subset_folder_path"]}')) == n_subsets:
        print("subsets found")
        subset_files = os.listdir(f'{env_vars["subset_folder_path"]}')
    else:
        print("Creating subsets")
        subset_files = subset_creator.create_subsets()

    data = []
    for subset_name in subset_files:
        subset_path = f'{env_vars["subset_folder_path"]}/{subset_name}'
        with open(f'{subset_path}', "rb") as f:
            subset_data = pickle.load(f)
        data.extend(subset_data)

    if str(env_vars["balanced_train_test"])=="True":
        train_data, test_data = get_balanced_train_test_split(data=data, 
                                                              n_bins=int(env_vars["n_bins"]), 
                                                              mask_size=ast.literal_eval(env_vars["mask_size"]), 
                                                              test_split_size=float(env_vars["test_split_size"])
                                                              )
    else:
        train_data, test_data = train_test_split(data, 
                                                test_size=float(env_vars["test_split_size"]), 
                                                random_state=42)
        

    
    with open(f'{env_vars["data_folder_path"]}/{env_vars["split_fname"]}', 'wb') as f:
        pickle.dump({'train_data': train_data, 'test_data': test_data}, f)
    

