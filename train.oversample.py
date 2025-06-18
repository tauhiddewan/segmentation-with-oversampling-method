import os
import gc
import ast
import time
import json
import copy
import torch
import socket
import pickle
import random
import numpy as np
from pathlib import Path
from PIL import Image
from dotenv import dotenv_values
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from utils.dataset import KvasirDataset
from utils.build import create_fresh_directory
from utils.train import (training_loop, 
                         test_loop,
                         get_lr_scheduler, 
                         cleanup_iteration)
from utils.model import select_model
from utils.loss import select_criterion
from utils.metrics import calculate_dice_score
from utils.misc import create_logger
from utils.notification import send_notification
from utils.build_train_test import get_binwise_data

def get_binswise_ratios(model_name, model, n_bins, env_vars, data, device, score_threshold):
    area_bins = np.linspace(0, 1, n_bins + 1)
    data_bins = get_binwise_data(data=data, 
                                 area_bins=area_bins, 
                                 mask_size=ast.literal_eval(env_vars["mask_size"]))

    img_transform = transforms.Compose([
        transforms.Resize(ast.literal_eval(env_vars["image_size"]), transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    mask_transform = transforms.Compose([
        transforms.Resize(ast.literal_eval(env_vars["mask_size"]), transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    binwise_ratios = []
    oversample_pool = [[] for j in range(len(data_bins))]
    
    for j in range(len(data_bins)):
        if len(data_bins[j])==0:
            binwise_ratios.append(0)
        else:
            ab_count, bl_count = 0, 0
            for image, mask in data_bins[j]:
                image_tensor, mask_tensor = img_transform(image).to(device), mask_transform(mask).to(device)
                batched_image_tensor = image_tensor.unsqueeze(0).to(device)

                preds = model(batched_image_tensor)

                if model_name=="polyp_pvt":
                    preds = preds[0].squeeze(0)
                else:
                    preds = preds.squeeze(0)

                dice_score = calculate_dice_score(preds=preds, targets=mask_tensor, device=device, model_name=model_name)
                score = dice_score.item()
                if score >= score_threshold:
                    ab_count += 1
                else:
                    bl_count += 1
                    oversample_pool[j].append((image, mask))
            binwise_ratios.append(bl_count / (bl_count+ab_count))
 
    return binwise_ratios, oversample_pool

def get_oversample_train_data(model_name, model, n_bins, env_vars, train_data, device, score_threshold, num_oversamples):
    oversampled_train_dataset = []
    binwise_ratios, training_oversample_pool = get_binswise_ratios(
        model_name=model_name,
        model=model, 
        n_bins=n_bins, 
        env_vars=env_vars, 
        data=train_data, 
        device=device, 
        score_threshold=score_threshold)
    
    for j in range(len(binwise_ratios)):
        if binwise_ratios[j] != 0:
            k = min(num_oversamples, len(training_oversample_pool[j]))
            oversampled_train_dataset.extend(random.choices(training_oversample_pool[j], k=k))
        
    return oversampled_train_dataset




def oversample_training(model_name, 
                        model_config,
                        n_bins,
                        score_threshold,
                        env_vars, 
                        logger, 
                        patience, 
                        save_model,
                        load_ckpt,
                        file_dir):
    
    
    #load the full dataset, split it into training and testing set 
    results = {}
    dataset = []
    receivers = ["dxr1368@miami.edu"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = f'{file_dir}/models'
    create_fresh_directory(models_dir)
    
    try: 
        with open(f'{env_vars["data_folder_path"]}/{env_vars["split_fname"]}', 'rb') as f:
            data = pickle.load(f)

        train_data, test_data = data['train_data'], data['test_data']
        train_data_size_original = len(train_data)
        
        train_dataset = KvasirDataset(data=train_data, 
                                      mode="train", 
                                      image_size=ast.literal_eval(env_vars["image_size"]), 
                                      mask_size=ast.literal_eval(env_vars["mask_size"]))
        
        test_dataset = KvasirDataset(data=test_data, 
                                     mode="test", 
                                     image_size=ast.literal_eval(env_vars["image_size"]), 
                                     mask_size=ast.literal_eval(env_vars["mask_size"]))
        
        
        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=int(env_vars["batch_size"]), 
                                      shuffle=True, 
                                      num_workers=4)
        
        test_dataloader = DataLoader(test_dataset, 
                                     batch_size=int(env_vars["batch_size"]), 
                                     shuffle=False, 
                                     num_workers=4)
                                     
        
        model = select_model(model_name=model_name, 
                             model_config=model_config)
        model.to(device)
    
        optimizer = torch.optim.AdamW(model.parameters(), 
                                      lr=float(env_vars["learning_rate"])) 
        
        scheduler = get_lr_scheduler(optimizer=optimizer, 
                                    num_epochs=int(env_vars["max_epochs"]), 
                                    warmup_epochs=5)
        
        criterion = select_criterion(model_name=model_name) 
                
        
        if load_ckpt=="True":
            checkpoint = torch.load(f'{env_vars["output_folder_path"]}/{env_vars["oversample_save_folder_name"]}/best_base_model.{model_name}_{model_config}.{env_vars["variant"]}.pt', weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model, epoch_loss_list, ma_loss_list, epoch_dice_score_list, epoch_miou_score_list = training_loop(
                dataloader=train_dataloader,
                model=model, 
                model_name=model_name,
                train_data_size=train_data_size_original,
                optimizer=optimizer,
                scheduler=scheduler, 
                criterion=criterion,
                device=device,
                threshold=float(env_vars["threshold"]),
                ma_window=int(env_vars["ma_window"]),
                max_epochs=int(env_vars["max_epochs"]), 
                min_epochs=int(env_vars["min_epochs"]),
                best_model_save_path=f'{models_dir}/pre_oversample.{model_name}_{model_config}.{env_vars["variant"]}.pt',
                logger=logger,
                save_model=save_model
                )
                
                
        test_loss, test_dice_score, test_miou_score = test_loop(
            model=model, 
            model_name=model_name,
            test_dataloader=test_dataloader, 
            criterion=criterion, 
            device=device,
            test_limit = int(env_vars["test_limit"]))
                
        
        logger.warning(f'Original Training Dataset size: {len(train_dataset)},  Test score: {test_dice_score:.4f}')
        # if f'original_data' not in results:
        #     results[f'original_data'] = {"size": len(train_data), 
        #                                  "dice_score": test_dice_score, 
        #                                  "iou_score": test_miou_score}
        
        results = {
            "overall": {
                "original_training_size": train_data_size_original,
                "initial_dice_score": test_dice_score,
                "initial_iou_score": test_miou_score
                }
}
        
        x = 1
        best_dice_score = test_dice_score
        
        num_oversamples = int(env_vars["num_oversamples"])
        while patience>0:
            logger.warning(f"Iterative oversampling step: {x} ...")
            
            oversampled_data = get_oversample_train_data(
                model_name=model_name, model=model, 
                n_bins=n_bins, 
                env_vars=env_vars, 
                train_data=train_data, 
                device=device,
                score_threshold=score_threshold,  
                num_oversamples=num_oversamples)
            
            train_data_oversampled = train_data + oversampled_data
            
            train_data_size_oversampled = len(train_data_oversampled)
        
            train_dataset_oversampled = KvasirDataset(
                data=train_data_oversampled, 
                mode="train", 
                image_size=ast.literal_eval(env_vars["image_size"]), 
                mask_size=ast.literal_eval(env_vars["mask_size"]))
            
            train_dataloader_oversampled = DataLoader(
                train_dataset_oversampled, 
                batch_size=int(env_vars["batch_size"]), 
                shuffle=True, 
                num_workers=4)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(env_vars["learning_rate"]))
            scheduler = get_lr_scheduler(
                optimizer=optimizer, 
                num_epochs=int(env_vars["max_epochs"]), 
                warmup_epochs=5)

            criterion = select_criterion(model_name=model_name)
            

            model, epoch_loss_list, ma_loss_list, epoch_dice_score_list, epoch_miou_score_list = training_loop(
                dataloader=train_dataloader_oversampled,
                model=model, 
                model_name=model_name,
                train_data_size=train_data_size_oversampled,
                optimizer=optimizer,
                scheduler=scheduler, 
                criterion=criterion,
                device=device,
                threshold=float(env_vars["threshold"]),
                ma_window=int(env_vars["ma_window"]),
                max_epochs=int(env_vars["max_epochs"]), 
                min_epochs=int(env_vars["min_epochs"]),
                best_model_save_path=f'{models_dir}/post_oversample.iter{x}.{model_name}_{model_config}.{env_vars["variant"]}.pt',
                logger=logger,
                save_model=save_model,
                )
                
                
            ovtest_loss, ovtest_dice_score, ovitest_miou_score = test_loop(
                model=model, 
                model_name=model_name,
                test_dataloader=test_dataloader, 
                criterion=criterion, 
                device=device,
                test_limit = int(env_vars["test_limit"]))

            logger.warning(f"Oversampled Training Dataset size : {len(train_data_oversampled)}, Test score: {ovtest_dice_score:.4f}")
            
            if ovtest_dice_score >= best_dice_score:
                best_dice_score = ovtest_dice_score
                best_model_path = f'{env_vars["output_folder_path"]}/{env_vars["oversample_save_folder_name"]}/best_base_model.{model_name}_{model_config}.{env_vars["variant"]}.pt'
                
                if load_ckpt!="True":
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, best_model_path)
                
                    logger.warning(f"New best model saved at {best_model_path} with Dice Score: {best_dice_score:.4f}")
        
                
            else:
                patience -= 1
                logger.warning(f"Iteration {x}: Test dice score dropped from {best_dice_score:.4f} to {ovtest_dice_score:.4f}. Patience now: {patience}")
                

            iteration_key = f'iter{x}'
            results["iterations"] = results.get("iterations", {})
            results["iterations"][iteration_key] = {
                "training_size": train_data_size_oversampled,
                "dice_score": ovtest_dice_score,
                "iou_score": ovitest_miou_score,
                "num_oversamples": num_oversamples
            }       
            
            x += 1
            num_oversamples += 5    
       
                
        results_fname = "results.json"
        with open(f'{file_dir}/{model_name}_{model_config}.{env_vars["variant"]}.{results_fname}', 'w') as f:
            json.dump(results, f, indent=4)
                
                
        subject = "Training completed!"
        body = f"Hello.\n{socket.gethostname()} has finished the training process.\nPlease check the {results_fname} file attached with this email for details.\n\nThanks\ninfo.training.johnston"
        for receiver in receivers:
            response = send_notification(
                subject=subject, 
                body=body,
                sender_email = str(env_vars["sender_email"]),
                receiver_email = str(receiver),
                smtp_server = str(env_vars["smtp_server"]),
                smtp_port = int(env_vars["smtp_port"]),
                password = str(env_vars["password"]),
                results_fname=results_fname, 
                results_fpath=f'{file_dir}/{model_name}_{model_config}.{env_vars["variant"]}.{results_fname}')
            
            
            logger.info(f'{response} for {receiver}')
    
    except Exception as e: 
        logger.error(e)
        subject = "Training incomplete!"
        body = f"Hello.\n{socket.gethostname()} couldn't finish the training process. Traceback for the error is following: \n{e}.\n\nThanks\ninfo.training.johnston"
        for receiver in receivers:
            response = send_notification(
                subject=subject, 
                body=body,
                sender_email = str(env_vars["sender_email"]),
                receiver_email = str(receiver),
                smtp_server = str(env_vars["smtp_server"]),
                smtp_port = int(env_vars["smtp_port"]),
                password = str(env_vars["password"]),
                results_fname=None, 
                results_fpath=None)
            logger.exception(f'{response} to {receiver}')
    return logger, results
    

if __name__=="__main__":
    env_vars = dotenv_values(dotenv_path="./.env")
    
    Path(env_vars["output_folder_path"]).mkdir(parents=True, exist_ok=True)
    model_name, model_config = str(env_vars["model_name"]), str(env_vars["model_config"])
    logger = create_logger(log_filename=f'train.oversample.threshold_{float(env_vars["score_threshold"])}.{model_name}_{model_config}.{env_vars["variant"]}', env_vars=env_vars)
    

    try:
        interrupted = False
        logger.warning("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< PROCESS STARTED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        logger, results = oversample_training(
            model_name=model_name, 
            model_config=model_config,
            env_vars=env_vars, 
            logger=logger,
            n_bins=int(env_vars["n_bins"]),
            score_threshold=float(env_vars["score_threshold"]),
            patience=int(env_vars["patience"]),
            save_model=bool(env_vars["save_model"]),
            load_ckpt=str(env_vars["load_ckpt"]),
            file_dir=f'{env_vars["output_folder_path"]}/{env_vars["oversample_save_folder_name"]}/threshold_{float(env_vars["score_threshold"])}')

    except KeyboardInterrupt:
        interrupted = True
        logger.warning("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< PROCESS INTERRUPTED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
        
    finally:
        if not interrupted:
            logger.warning("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  PROCESS ENDED  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
        
        for handler in logger.handlers:
            handler.flush()
            
            
    
    