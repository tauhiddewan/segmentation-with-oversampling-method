import atexit
import logging
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from email import encoders
from dotenv import dotenv_values
from utils.metrics import calculate_dice_score, calculate_iou_score

def create_logger(log_filename, env_vars):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO) 
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f'{env_vars["output_folder_path"]}/{log_filename}.log')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    # Ensure logs are written on program exit, even if interrupted
    def flush_logs():
        for handler in logger.handlers:
            handler.flush()
    atexit.register(flush_logs)
    return logger

def evaluate_masks(pseudo_dataset, real_masks):
    iou_scores = []
    dice_scores = [] 
    pixel_accuracies = []

    for i in range(len(pseudo_dataset)):

        # Calculate metrics
        iou = calculate_iou(pseudo_dataset[i][1], real_masks[i])
        dice = calculate_dice(pseudo_dataset[i][1], real_masks[i])
        accuracy = calculate_pixel_accuracy(pseudo_dataset[i][1], real_masks[i])

        # Store metrics
        iou_scores.append(iou)
        dice_scores.append(dice)
        pixel_accuracies.append(accuracy)

    # Average metrics
    avg_iou = sum(iou_scores) / len(iou_scores)
    avg_dice = sum(dice_scores) / len(dice_scores)
    avg_pixel_accuracies = sum(pixel_accuracies) / len(pixel_accuracies)

    return avg_iou, avg_dice, avg_pixel_accuracies


def get_dice_scores(pseudo_dataset, real_masks):
    dice_scores = []
    for i in range(len(pseudo_dataset)):
        dice = calculate_dice(pseudo_dataset[i][1], real_masks[i])
        dice_scores.append((i, dice))
    return dice_scores

