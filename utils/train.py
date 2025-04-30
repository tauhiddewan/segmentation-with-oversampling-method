import gc
import time
import atexit
import random
import smtplib
import logging
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from dotenv import dotenv_values
from utils.metrics import calculate_dice_score, calculate_iou_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def training_step(dataloader, 
                  model, 
                  model_name,
                  optimizer,
                  criterion,
                  device):
    
    total_loss, total_dice_score, total_iou_score = 0.0, 0.0, 0.0
    
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        images, labels = batch 
        images, labels = images.to(device), labels.to(device)
        
        logits = model(images)
        batch_loss = criterion(logits, labels)
        batch_loss.backward()
        optimizer.step()
        
        batch_dice_score = calculate_dice_score(preds=logits, targets=labels, device=device, model_name=model_name)
        batch_iou_score = calculate_iou_score(preds=logits, targets=labels, device=device, model_name=model_name)
        
        total_loss += batch_loss.item()
        total_dice_score += batch_dice_score.item()
        total_iou_score += batch_iou_score.item()
    return model, optimizer, total_loss/len(dataloader), total_dice_score/len(dataloader), total_iou_score/len(dataloader)   

  
def check_stopping_conditions(epoch_loss_list, 
                              current_loss,
                              current_ma_loss,
                              ma_window, 
                              threshold,
                              min_epochs=100):
    if len(epoch_loss_list)<ma_window or len(epoch_loss_list)<min_epochs:
        return False
    if abs(current_loss) < 1e-8:  # Small constant to avoid division by zero
        return False
    
    relative_diff = abs(current_loss - current_ma_loss) / current_loss
    
    if relative_diff <= threshold:
        return True
    
    return False

        
def training_loop(dataloader, 
                  model, 
                  model_name,
                  optimizer,
                  scheduler,
                  criterion,
                  device,
                  threshold, 
                  train_subset_names,
                  test_subset_names,
                  ma_window,
                  max_epochs, 
                  min_epochs, 
                  best_model_save_path, 
                  logger,
                  use_scheduler = False, 
                  save_model=True):
    model.train()
    epoch_loss_list, ma_loss_list, epoch_dice_score_list, epoch_miou_score_list  = [], [], [], []
    min_loss = float('inf')
    min_loss_score = 0
    best_model, best_optimizer = None, None
    epoch = 0
    
    logger.info(f'Training started...')
    logger.info(f"Choosen Train subsets: {sorted(train_subset_names)}, Test subsets: {test_subset_names}")
    while True:
        model.train()
        epoch += 1
        model, optimizer, mean_epoch_loss, mean_epoch_dice_score, mean_epoch_miou_score = training_step(dataloader=dataloader,
                                                                                                        model=model, 
                                                                                                        model_name=model_name, 
                                                                                                        optimizer=optimizer, 
                                                                                                        criterion=criterion, 
                                                                                                        device=device)
        epoch_loss_list.append(mean_epoch_loss)
        epoch_dice_score_list.append(mean_epoch_dice_score)
        epoch_miou_score_list.append(mean_epoch_miou_score)
        
        ma_loss = np.mean(epoch_loss_list[-ma_window:])
        ma_loss_list.append(ma_loss)
        
        logger.info(f"Epoch [{epoch:04}], Loss: {mean_epoch_loss:.4f}, MA loss: {ma_loss:.4f}, Dice Score: {mean_epoch_dice_score:.4f}, mIoU Score: {mean_epoch_miou_score:.4f}")

        if mean_epoch_loss<min_loss:
            best_model_epoch = epoch
            min_loss_dice_score = mean_epoch_dice_score
            min_loss_miou_score = mean_epoch_miou_score
            min_loss = mean_epoch_loss
            best_model = copy.deepcopy(model)
            best_optimizer = copy.deepcopy(optimizer)
                              
                              
        stopping = check_stopping_conditions(epoch_loss_list=epoch_loss_list,
                                             current_loss=epoch_loss_list[-1],
                                             current_ma_loss=ma_loss,
                                             ma_window=ma_window, 
                                             threshold=threshold, 
                                             min_epochs=min_epochs) 
        
        if stopping:
            logger.warning(f"Metrics converged and loss stabilized. Training stopped!")
            break
        if epoch >= max_epochs:
            logger.warning("Reached maximum number of epochs. Training stopped")
            break
         
        if use_scheduler:
            scheduler.step()
    
    if save_model==True and best_model is not None:
        torch.save({
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': best_optimizer.state_dict()
        }, best_model_save_path)
        logger.info(f"Best model found at Epoch: {best_model_epoch:04}. Loss: {min_loss:.4f}, Dice score: {min_loss_dice_score:.4f}, mIoU score: {min_loss_miou_score:.4f}")
    
    return best_model, epoch_loss_list, ma_loss_list, epoch_dice_score_list, epoch_miou_score_list


def test_loop(model, 
              model_name,
              test_dataloader, 
              criterion, 
              device,
              test_limit):
    # ? TESTING THE DATASET `test_limit` TIMES AND TAKING AN AVARAGE FOR REDUCING RANDOMNESS
    final_test_loss, final_dice_score, final_miou_score = 0.0, 0.0, 0.0
    for _ in range(test_limit):
        model.eval()
        total_loss, total_dice_score, total_miou_score = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in test_dataloader:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                
                dice_score = calculate_dice_score(preds=logits,
                                                  targets=labels, 
                                                  device=device, 
                                                  model_name=model_name)
                iou_score = calculate_iou_score(preds=logits, 
                                                targets=labels,
                                                device=device, 
                                                model_name=model_name)
                
                total_loss += loss.item()
                total_dice_score += dice_score.item()
                total_miou_score += iou_score.item()

        mean_loss = total_loss / len(test_dataloader)
        mean_dice_score = total_dice_score / len(test_dataloader)
        mean_miou_score = total_miou_score / len(test_dataloader)
        final_test_loss += mean_loss
        final_dice_score += mean_dice_score
        final_miou_score += mean_miou_score
    return final_test_loss/test_limit, final_dice_score/test_limit, final_miou_score/test_limit


def get_name(subset_names):
    subset_names.sort()
    name = ""
    for n in subset_names:
        name += n
    return name

def get_remaining_subsets(dataset, training_subset_names):
    rem_subsets, rem_subset_names = [], []
    for key in dataset.keys():
        if key not in training_subset_names:
            rem_subsets.append(dataset[key])
            rem_subset_names.append(key)
    return rem_subsets, rem_subset_names


def tensor_to_pil(tensor):
    if tensor.ndimension() == 3: 
        tensor = tensor.permute(1, 2, 0) 
    array = tensor.numpy()
    if array.dtype != 'uint8':
        array = (array * 255).astype("uint8")
    pil_img = Image.fromarray(array.squeeze())
    return pil_img


def generate_masks(model, 
                   model_name,
                   rem_dataset, 
                   device):
    model.eval() 
    gen_dataset, real_masks, dice_scores = [], [], []
    
    with torch.no_grad(): 
        for i in range(len(rem_dataset)):
            image, label = rem_dataset[i]
            # adding batch dim
            image_batched = image.unsqueeze(0).to(device)
            label_batched = label.unsqueeze(0).to(device)
            
            logits = model(image_batched) 
            dice_score = calculate_dice_score(preds=logits, 
                                              targets=label, 
                                              device=device, 
                                              model_name=model_name)
            # Binary threshold after sigmoid
            if model_name!="unet":
                logits = torch.sigmoid(logits)
            gen_mask_batched = (logits > 0.5).float() 
            
            # removing batch dim
            gen_mask  = gen_mask_batched.squeeze(0).cpu()
            
            gen_dataset.append((image, gen_mask))
            real_masks.append(label)
            dice_scores.append((i, dice_score))
    return gen_dataset, real_masks, dice_scores



def get_lr_scheduler(optimizer, 
                     num_epochs: int, 
                     warmup_epochs: int = 5, 
                     min_lr: float = 1e-6):
    
    warmup = LinearLR(optimizer, 
                     start_factor=0.1,
                     total_iters=warmup_epochs)
    
    cosine = CosineAnnealingLR(optimizer, 
                               T_max=num_epochs - warmup_epochs, 
                               eta_min=min_lr)
    
    return SequentialLR(optimizer, 
                       schedulers=[warmup, cosine], 
                       milestones=[warmup_epochs])

def cleanup_iteration(variables, device, logger, wait_time: int = 5):
    try:
        if 'model' in variables:
            variables['model'].cpu()
        
        keys_to_delete = list(variables.keys())
        
        for var in keys_to_delete:
            del variables[var]
            
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        time.sleep(wait_time)
    
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
