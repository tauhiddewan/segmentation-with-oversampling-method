import os
import sys
import ast
import copy
import json
import torch
import pickle
import numpy as np
import socket
import random
import logging
from PIL import Image
from pathlib import Path
from dotenv import dotenv_values
from collections import Counter
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import gaussian_kde, ks_2samp
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from utils.dataset import KvasirDataset
from utils.model import select_model
from utils.visualization import (plot_training_results, 
                                 plot_test_results, 
                                 get_mask_quality_pct, 
                                 get_dataset, generate_mask, get_sas_modelwise_results, plot_modelwise_comparison)

from utils.metrics import calculate_dice_score, calculate_iou_score
# from ptflops import get_model_complexity_info
env_vars = dotenv_values(dotenv_path="../.env")
logging.getLogger('torch').setLevel(logging.ERROR)
