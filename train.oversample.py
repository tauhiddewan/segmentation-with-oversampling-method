import os
import gc
import ast
import json
import copy
import torch
import socket
import pickle
import random
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from dotenv import dotenv_values
from typing import Optional, Any, Mapping
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.dataset import KvasirDataset
from utils.build import create_fresh_directory
from utils.model import select_model
from utils.loss import select_criterion
from utils.metrics import calculate_dice_score
from utils.misc import create_logger
from utils.build_train_test import get_binwise_data
from utils.notification import Notify
from utils.train import (
    training_loop, 
    test_loop,
    get_lr_scheduler, 
    cleanup_iteration
)

class BinManager:
    def __init__(self, n_bins, env_vars):
        self.area_bins = np.linspace(0, 1, n_bins+1)
        self.oversampled_data = []

        self.img_transform = transforms.Compose([
            transforms.Resize(
                ast.literal_eval(env_vars.get("image_size")), 
                transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            ])
        self.msk_transform = transforms.Compose([
            transforms.Resize(
                ast.literal_eval(env_vars.get("mask_size")), 
                transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
            ])


    def process(self, train_data, model, model_name, device, score_threshold, num_oversamples):
        data_bins = [[] for _ in range(len(self.area_bins)-1)]
        for idx, data in enumerate(train_data):
            mask = self.msk_transform(data[1])
            area_ratio = (mask == 1).sum().item() / mask.numel()
            for j in range(len(self.area_bins)-1):
                if area_ratio >= self.area_bins[j] and area_ratio < self.area_bins[j+1]:
                    data_bins[j].append(data)

        data_bin_ratios = []
        oversample_pool = [[] for j in range(len(data_bins))]

        for idx, data_bin in enumerate(data_bins):
            if len(data_bin)==0: data_bin_ratios.append(0)
            else:
                below = 0
                for img, msk in data_bin:
                    img_tensor, msk_tensor = self.img_transform(img), self.msk_transform(msk)
                    img_tensor, msk_tensor = img_tensor.to(device), msk_tensor.to(device)

                    logits = model(img_tensor.unsqueeze(0).to(device))
                    logits = logits[0].squeeze(0) if model_name=="polyp_pvt" else logits.squeeze(0)

                    score = calculate_dice_score(
                        preds=logits, 
                        targets=msk_tensor, 
                        device=device, 
                        model_name=model_name
                    ).item()

                    if score<score_threshold:
                        below += 1
                        oversample_pool[idx].append((img, msk))

                data_bin_ratios.append(below/len(data_bin))


        for j in range(len(data_bin_ratios)):
            if data_bin_ratios[j] != 0 and len(oversample_pool[j]) > 0:
                k = min(num_oversamples, len(oversample_pool[j]))
                self.oversampled_data.extend(random.choices(oversample_pool[j], k=k))

        return self.oversampled_data


class TrainingSession: 
    def __init__(
            self, 
            env_vars: Mapping[str, Any],
            model_name: str, 
            model_config: str,
            logger: logging.Logger,
            file_dir: str | Path
    ):
        self.results = {}       
        self.env_vars = env_vars
        self.model_name = model_name
        self.model_config = model_config
        self.logger = logger
        self.file_dir = str(file_dir)
        self.models_dir = f'{self.file_dir}/models'
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training hyper-params
        self.base_score = None
        self.base_model = None
        self.best_ov_score = None        
        self.best_ov_model = None

        self.load_ckpt = str(self.env_vars.get("load_ckpt", "False")) == "True"
        self.save_model = str(self.env_vars.get("save_model", "True")) == "True"
        self.n_bins = int(self.env_vars.get("n_bins", 20))
        self.test_limit = int(self.env_vars.get("test_limit", 1))  
        self.stop_threshold = float(self.env_vars.get("threshold", 1e-4))
        self.ma_window = int(self.env_vars.get("ma_window", 10))
        self.max_epochs = int(self.env_vars.get("max_epochs", 700))
        self.min_epochs = int(self.env_vars.get("min_epochs", 200))
        self.lr = float(self.env_vars.get("learning_rate", 1e-3))
        self.num_oversamples = int(self.env_vars.get("num_oversamples", 5))
        self.image_size = ast.literal_eval(self.env_vars.get("image_size", "(384, 384)"))
        self.mask_size = ast.literal_eval(self.env_vars.get("mask_size", "(384, 384)"))
        self.batch_size = int(self.env_vars.get("batch_size", 12))
        self.thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.97]

        self.folder_path = f'{self.env_vars.get("output_folder_path")}/{self.env_vars.get("oversample_save_folder_name")}'
        Path(self.folder_path).mkdir(parents=True, exist_ok=True)

        self.best_model_path = (
            f'{self.models_dir}/best_model.{self.model_name}_{self.model_config}.{self.env_vars["variant"]}.pt'
        )

    def __enter__(self):
        with open(f'{self.env_vars["data_folder_path"]}/{self.env_vars["split_fname"]}', "rb") as f:
            data = pickle.load(f)

        self.train_data = data["train_data"]
        self.test_data = data["test_data"]

        self.train_dataset = KvasirDataset(self.train_data, "train", self.image_size, self.mask_size)
        self.test_dataset = KvasirDataset(self.test_data, "test", self.image_size, self.mask_size)
        self.train_dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=4)
        self.test_dataloader = DataLoader(self.test_dataset, self.batch_size, shuffle=False, num_workers=4)

        self.model = select_model(self.model_name, self.model_config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = get_lr_scheduler(self.optimizer, self.max_epochs, warmup_epochs=5) 
        self.criterion = select_criterion(self.model_name)

        self.results = {
            "overall": {
                "original_training_size": len(self.train_dataset),
            },
            "iterations": {}
        }

        return self

    def base_train(self):
        pre_oversample_save_path = (
            f'{self.models_dir}/pre_oversample.{self.model_name}_{self.model_config}.{self.env_vars["variant"]}.pt'
        )

        if self.load_ckpt and Path(self.best_model_path).exists():
            ckpt = torch.load(self.best_model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.logger.warning(f"Loaded checkpoint from {self.best_model_path}")
        else:
            base_model, *_ = training_loop(
                dataloader=self.train_dataloader,
                model=self.model,
                model_name=self.model_name,
                train_data_size=len(self.train_dataset),
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                criterion=self.criterion,
                device=self.device,
                threshold=self.stop_threshold,
                ma_window=self.ma_window,
                max_epochs=self.max_epochs,
                min_epochs=self.min_epochs,
                best_model_save_path=pre_oversample_save_path,
                logger=self.logger,
                save_model=self.save_model
            )

            test_loss, dice, iou = test_loop(
                test_dataloader=self.test_dataloader, 
                model=self.model, 
                model_name=self.model_name, 
                criterion=self.criterion, 
                device=self.device, 
                test_limit=(self.test_limit if self.test_limit > 0 else None)
            )
            
            self.base_model = copy.deepcopy(base_model)
            self.base_score = float(dice)
            self.results["overall"]["initial_dice_score"] = float(dice)
            self.results["overall"]["initial_iou_score"] = float(iou)
            self.logger.warning(
                f'Original Training size: {len(self.train_dataset)}, '
                f'Test Dice: {float(dice):.4f}, IOU: {float(iou):.4f}'
            )

    def oversample_step(self, score_threshold: float):
        self.logger.warning(f"Iterative oversampling step (thr={score_threshold})")
        post_oversample_save_path = (
            f'{self.models_dir}/post_oversample.{self.model_name}_{self.model_config}.{self.env_vars["variant"]}_{score_threshold}.pt'
        )

        # Start from base model for this threshold
        ov_model = copy.deepcopy(self.base_model).to(self.device)
        patience = int(self.env_vars.get("patience", 2))
        patience_left = patience

        per_bin_oversample = self.num_oversamples

        while patience_left > 0:
            self.logger.warning(f"patience used={patience - patience_left + 1} / {patience} >>>")

            bin_manager = BinManager(n_bins=self.n_bins, env_vars=self.env_vars)
            oversampled_data = bin_manager.process(
                train_data=self.train_data, 
                model=ov_model, 
                model_name=self.model_name, 
                device=self.device, 
                score_threshold=score_threshold, 
                num_oversamples=per_bin_oversample
            )

            oversampled_training_data = list(self.train_data) + list(oversampled_data)

            ov_train_dataset = KvasirDataset(oversampled_training_data, "train", self.image_size, self.mask_size)
            ov_train_dataloader = DataLoader(ov_train_dataset, self.batch_size, shuffle=True, num_workers=4)
            
            # Fresh optimizer/scheduler per iteration (as you intended)
            optimizer = torch.optim.AdamW(ov_model.parameters(), lr=self.lr)
            scheduler = get_lr_scheduler(optimizer, self.max_epochs, warmup_epochs=5) 
            criterion = select_criterion(self.model_name)

            ov_model, *_ = training_loop(
                dataloader=ov_train_dataloader,
                model=ov_model,
                model_name=self.model_name,
                train_data_size=len(oversampled_training_data),
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                device=self.device,
                threshold=self.stop_threshold,
                ma_window=self.ma_window,
                max_epochs=self.max_epochs,
                min_epochs=self.min_epochs,
                best_model_save_path=post_oversample_save_path,
                logger=self.logger,
                save_model=self.save_model
            )

            test_loss, ov_dice, ov_iou = test_loop(
                test_dataloader=self.test_dataloader, 
                model=ov_model, 
                model_name=self.model_name, 
                criterion=criterion, 
                device=self.device, 
                test_limit=(self.test_limit if self.test_limit > 0 else None)
            )

            ov_dice = float(ov_dice)
            ov_iou = float(ov_iou)

            self.logger.warning(
                f"Oversampled Training size: {len(oversampled_training_data)}, "
                f"Test Dice: {ov_dice:.3f} (best so far: {f'{self.best_ov_score:.3f}' if self.best_ov_score is not None else 'None'})"
            )
        
            if (self.best_ov_score is None) or (ov_dice > self.best_ov_score):
                self.best_ov_score = ov_dice
                self.best_ov_model = copy.deepcopy(ov_model).to(self.device)

                if self.save_model:
                    torch.save(
                        {
                            'model_state_dict': ov_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                        },
                        self.best_model_path
                    )
                self.logger.warning(
                    f"New best model saved with Dice Score: {self.best_ov_score:.3f}"
                )
                # Reset patience on improvement
                patience_left = patience
            else:
                patience_left -= 1
                self.logger.warning(
                    f"Dice did not improve (current: {ov_dice:.4f}, best: {f'{self.best_ov_score:.3f}' if self.best_ov_score is not None else -1.0}). "
                    f"Patience left: {patience_left}"
                )

            iter_key = f'thr_{score_threshold:.3f}_iter_{patience - patience_left}'
            self.results["iterations"][iter_key] = {
                "training_size": len(oversampled_training_data),
                "dice_score": ov_dice,
                "iou_score": ov_iou,
                "num_oversamples": per_bin_oversample,
                "score_threshold": float(score_threshold),
            }

            per_bin_oversample += 5 

    def run_full_pipeline(self):
        self.base_train()

        for thr in self.thresholds:
            self.oversample_step(score_threshold=thr)

    def dump_results(self):
        results_fname = f'{self.model_name}_{self.model_config}.{self.env_vars["variant"]}.results.json'
        results_fpath = f'{self.file_dir}/{results_fname}'
        Path(self.file_dir).mkdir(parents=True, exist_ok=True)
        with open(results_fpath, 'w') as f:
            json.dump(self.results, f, indent=4)
        return results_fpath  
    
    def __exit__(self, exc_type, exc, tb):
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for h in getattr(self.logger, "handlers", []):
                try: h.flush()
                except: pass
        finally:
            return False

                    

if __name__=="__main__":
    interrupted = False
    env_vars = dotenv_values(dotenv_path="./.env")
    Path(env_vars["output_folder_path"]).mkdir(parents=True, exist_ok=True)
    model_name, model_config = str(env_vars["model_name"]), str(env_vars["model_config"])
    job_name = f"{model_name}_{model_config}.{env_vars['variant']}"
    file_dir = Path(env_vars["output_folder_path"] / Path(env_vars["oversample_save_folder_name"]))

    logger = create_logger(
        log_filename=f'train.oversample.{job_name}', 
        env_vars=env_vars
    )

    notify_params = {
        "job_name": job_name,
        "sender_email": env_vars.get("sender_email"),
        "receiver_emails": ast.literal_eval(env_vars.get("receiver_emails", "[]")),
        "smtp_server": env_vars.get("smtp_server"),
        "smtp_port": int(env_vars.get("smtp_port", "587")),
        "password": env_vars.get("password"),
        "logger": logger,
    }

    with Notify(**notify_params) as notifier:
        try:
            logger.warning("<<<<<<<<<<<<<<<<<<<< PROCESS STARTED >>>>>>>>>>>>>>>>>>>>")
            with TrainingSession(env_vars, model_name, model_config, logger, file_dir) as sess:
                sess.run_full_pipeline()
                results_path = sess.dump_results()
                logger.warning(f"Results written to: {results_path}")
        except KeyboardInterrupt:
            interrupted = True
            logger.warning("<<<<<<<<<<<<<<<<<<<< PROCESS INTERRUPTED >>>>>>>>>>>>>>>>>>\n")
        finally:
            if not interrupted:
                logger.warning("<<<<<<<<<<<<<<<<<<<<  PROCESS ENDED  >>>>>>>>>>>>>>>>>>\n")
            for handler in logger.handlers:
                handler.flush()





