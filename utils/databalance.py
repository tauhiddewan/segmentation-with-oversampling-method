import ast
import random
import pickle
import numpy as np 
from dotenv import dotenv_values
from torch.utils.data import ConcatDataset
from utils.dataset import KvasirDataset
from torchvision.transforms.functional import to_pil_image

def get_mask_area_ratio(mask):
    foreground = (mask == 1).sum().item() 
    total_pixels = mask.numel() 
    return foreground / total_pixels

def get_binwise_subset_data(subset_dataset):
    area_bins = np.linspace(0, 1, 11)
    data_bins = [[] for ab in area_bins]
    for i in range(len(subset_dataset)):
        area_ratio = get_mask_area_ratio(mask=subset_dataset[i][1])
        for j in range(len(area_bins)-1):
            if area_ratio>=area_bins[j] and area_ratio<area_bins[j+1]:
                data_bins[j].append(subset_dataset[i])
                
    return data_bins

def get_reverse_binwise_scalefactor(subset_binwise_count):
    counts = np.array(subset_binwise_count, dtype=float)
    
    counts = np.clip(counts, 1e-3, None)  # Ensures no log(0) issue
    bins = np.arange(len(counts))
    
    log_counts = np.log(counts)
    slope, intercept = np.polyfit(bins, log_counts, 1)
    A, B = np.exp(intercept), -slope
    
    ## getting reverse weights
    binwise_weights = np.exp(B*bins)
    binwise_weights_norm = binwise_weights / np.mean(binwise_weights)
    effective_counts = counts*binwise_weights_norm
    
    ## rescaling to match original counts
    scale_factor = np.sum(counts) / np.sum(effective_counts)
    effective_counts_scaled = np.round(effective_counts * scale_factor).astype(int)
    
    return effective_counts_scaled

def get_reversed_dataset(subset_binwise_data, subset_binwise_count_reversed):
    expanded_data = []
    for i, factor in enumerate(subset_binwise_count_reversed):
        if factor > 0:
            if len(subset_binwise_data[i]) > 0:
                sampled_data = random.choices(subset_binwise_data[i], k=int(factor))  
                for img, mask in sampled_data:
                    expanded_data.append((to_pil_image(img), to_pil_image(mask)))
    return expanded_data


def balance_training_data(env_vars, train_data):
    train_dataset = KvasirDataset(data=train_data, 
                                  mode="test", 
                                  image_size=ast.literal_eval(env_vars["image_size"]), 
                                  mask_size=ast.literal_eval(env_vars["mask_size"]))

    subset_binwise_data = get_binwise_subset_data(train_dataset)
    subset_binwise_count = [len(data) if (len(data))>0 else 1e-6 for data in subset_binwise_data]
    subset_binwise_count_reversed = get_reverse_binwise_scalefactor(subset_binwise_count)
    expanded_data = get_reversed_dataset(subset_binwise_data, subset_binwise_count_reversed)


    combined_balanced_data = ConcatDataset([train_data, expanded_data])
    return combined_balanced_data
    

if __name__=="__main__":
    env_vars = dotenv_values(dotenv_path="./.env")
    with open("data/subsets/subset_a.pkl", "rb") as data:
        subset_data = pickle.load(data)
        
        
    print(len(balance_training_data(env_vars, subset_data)))