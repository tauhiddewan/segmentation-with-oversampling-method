import os
import gc
import ast
import math
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
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from utils.dataset import KvasirDataset
from utils.build import create_fresh_directory
from utils.model import select_model
from utils.loss import select_criterion
from utils.metrics import calculate_dice_score
from utils.misc import create_logger
from utils.build_train_test import get_binwise_data
from utils.notification import Notify
from utils.tracker import ResultTracker  
from utils.sampler import BinManager
from utils.viz import get_area_vs_dice, plot_butterfly_mask_vs_score

from utils.train import (
    training_loop, 
    test_loop,
    get_lr_scheduler, 
    cleanup_iteration
)

class TrainingSession: 
    def __init__(
            self, 
            env_vars: Mapping[str, Any],
            model_name: str, 
            model_config: str,
            logger: logging.Logger,
            file_dir: str | Path
    ):  
        self.env_vars = env_vars
        self.model_name = model_name
        self.model_config = model_config
        self.logger = logger
        self.file_dir = str(file_dir)
        self.models_dir = f'{self.file_dir}/models'
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training hyper-params
        self.base_model = None     # immutable: pre-oversample snapshot (for plotting)
        self.best_model = None     # evolves: warm-start source for thresholds
        self.best_score = None
        self.save_model = str(self.env_vars.get("save_model", "True")) == "True"
        self.n_bins = int(self.env_vars.get("n_bins", 20))
        self.num_repeat = int(self.env_vars.get("num_repeat", 1))  
        self.stop_threshold = float(self.env_vars.get("threshold", 1e-4))
        self.ma_window = int(self.env_vars.get("ma_window", 10))
        self.max_epochs = int(self.env_vars.get("max_epochs", 700))
        self.min_epochs = int(self.env_vars.get("min_epochs", 200))
        self.lr = float(self.env_vars.get("learning_rate", 1e-3))
        self.num_oversamples = int(self.env_vars.get("num_oversamples", 5))
        self.image_size = ast.literal_eval(self.env_vars.get("image_size", "(384, 384)"))
        self.mask_size = ast.literal_eval(self.env_vars.get("mask_size", "(384, 384)"))
        self.batch_size = int(self.env_vars.get("batch_size", 12))

        self.thresholds = ast.literal_eval(self.env_vars.get("thresholds", "[0.5, 0.4, 0.3, 0.2, 0.1]"))

        self.folder_path = f'{self.env_vars.get("output_folder_path")}/{self.env_vars.get("oversample_save_folder_name")}'
        Path(self.folder_path).mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        with open(f'{self.env_vars["data_folder_path"]}/{self.env_vars["split_fname"]}', "rb") as f:
            data = pickle.load(f)

        self.train_data = data["train_data"]
        self.test_data = data["test_data"]

        self.train_core_data, self.val_data = train_test_split(self.train_data, test_size=0.1, random_state=42)

        self.pre_os_train_dataset = KvasirDataset(self.train_core_data, "train", self.image_size, self.mask_size)
        self.valid_dataset = KvasirDataset(self.val_data, "test", self.image_size, self.mask_size)
        self.test_dataset = KvasirDataset(self.test_data, "test", self.image_size, self.mask_size)

        self.train_dataloader = DataLoader(self.pre_os_train_dataset, self.batch_size, shuffle=True, num_workers=4)
        self.valid_dataloader = DataLoader(self.valid_dataset, self.batch_size, shuffle=False, num_workers=4)
        self.test_dataloader = DataLoader(self.test_dataset, self.batch_size, shuffle=False, num_workers=4)

        self.model = select_model(self.model_name, self.model_config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = get_lr_scheduler(self.optimizer, self.max_epochs, warmup_epochs=5) 
        self.criterion = select_criterion(self.model_name)

        self.tracker = ResultTracker(
            save_dir=self.file_dir,                         
            model_name=self.model_name,
            model_config=self.model_config,
            variant=self.env_vars.get("variant")
        )
        return self
    
    def pre_oversample_train(self):
        pretrained_save_path = (
            f'pre_oversample.{self.model_name}_{self.model_config}.{self.env_vars["variant"]}.'
            f'traindata_{len(self.pre_os_train_dataset)}.pt'
        )

        new_training_save_path = (
            f'{self.models_dir}/pre_oversample.'
            f'{self.model_name}_{self.model_config}.{self.env_vars["variant"]}'
        )

        if Path(pretrained_save_path).is_file():
            self.logger.info(f'Loading pre-oversample model from: {pretrained_save_path}')
            checkpoint = torch.load(pretrained_save_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            pre_os_model = self.model
        else:
            self.logger.info('Training pre-oversample model...')
            pre_os_model, *_ = training_loop(
                dataloader=self.train_dataloader,
                model=self.model,
                model_name=self.model_name,
                train_data_size=len(self.pre_os_train_dataset),
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                criterion=self.criterion,
                device=self.device,
                threshold=self.stop_threshold,
                ma_window=self.ma_window,
                max_epochs=self.max_epochs,
                min_epochs=self.min_epochs,
                best_model_save_path=new_training_save_path,
                logger=self.logger,
                save_model=self.save_model
            )
            self.logger.info(f'Saving pre-oversample model to: {pretrained_save_path}')
            torch.save({'model_state_dict': pre_os_model.state_dict()}, f'{pretrained_save_path}')
        
        # Immutable snapshot for plotting
        self.base_model = copy.deepcopy(pre_os_model).to(self.device)
        # Best-so-far starts as the pre-oversample model
        self.best_model = copy.deepcopy(pre_os_model).to(self.device)

    def post_oversample_train(self, score_threshold: float):
        iter_idx = 0
        best_model_this_thr = None
        best_score_this_thr = None
        best_iter_idx_this_thr = None
        per_bin_oversample = self.num_oversamples
        no_improve_streak = 0
        max_k = int(self.env_vars.get("max_per_bin_oversample", 80))

        self.logger.warning(f"Iterative oversampling step (thr={score_threshold})")
        post_oversample_save_path = (
            f'{self.models_dir}/post_oversample.{self.model_name}_{self.model_config}.{self.env_vars["variant"]}_{score_threshold}.pt'
        )

        # Warm start this threshold from the evolving best model (base_model stays immutable)
        post_os_model = copy.deepcopy(self.best_model).to(self.device)

        patience = int(self.env_vars.get("patience", 3))
        patience_left = patience
        
        running_oversampled_data = []
        max_total_oversamples = int(float(self.env_vars.get("max_total_oversample_pct", 0.5)) * len(self.train_core_data)) ## 50% of training data 
        
        while patience_left > 0:
            iter_idx += 1
            oversampled_data = BinManager(n_bins=self.n_bins, env_vars=self.env_vars).process(
                train_data=self.train_core_data, 
                model=post_os_model, 
                model_name=self.model_name, 
                device=self.device, 
                score_threshold=score_threshold, 
                num_oversamples=per_bin_oversample
            )
            running_oversampled_data.extend(oversampled_data)

            if len(running_oversampled_data) > max_total_oversamples:
                running_oversampled_data = running_oversampled_data[-max_total_oversamples:]  # keep most recent

            oversampled_ds = KvasirDataset(
                data=running_oversampled_data,
                mode='oversample',
                image_size=self.image_size,
                mask_size=self.mask_size
            )

            post_os_train_dataset = ConcatDataset([self.pre_os_train_dataset, oversampled_ds])
            post_os_train_dataloader = DataLoader(post_os_train_dataset, self.batch_size, shuffle=True, num_workers=4)

            self.logger.warning(
                f"[thr={score_threshold:.2f}] iter={iter_idx} | "
                f"new_oversamples={len(oversampled_data)} | "
                f"running_oversamples={len(running_oversampled_data)} | "
                f"concat_train_size={len(post_os_train_dataset)}"
            )
            
            optimizer = torch.optim.AdamW(post_os_model.parameters(), lr=self.lr)
            scheduler = get_lr_scheduler(optimizer, self.max_epochs, warmup_epochs=5) 
            criterion = self.criterion

            post_os_model, *_ = training_loop(
                dataloader=post_os_train_dataloader,
                model=post_os_model,
                model_name=self.model_name,
                train_data_size=len(post_os_train_dataset),
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

            post_os_valid_loss, post_os_valid_dice, post_os_valid_iou = test_loop(
                test_dataloader=self.valid_dataloader, 
                model=post_os_model, 
                model_name=self.model_name, 
                criterion=criterion, 
                device=self.device, 
                num_repeat=(self.num_repeat if self.num_repeat > 0 else None)
            )

            post_os_valid_dice = float(post_os_valid_dice)
            post_os_valid_iou = float(post_os_valid_iou)

            # Save per-iter record under the threshold
            self.tracker.add_post_iter(
                thr=score_threshold,
                iter_idx=iter_idx,
                valid_loss=float(post_os_valid_loss),
                valid_dice=float(post_os_valid_dice),
                valid_iou=float(post_os_valid_iou),
                new_oversamples=int(len(oversampled_data)),
                running_oversamples=int(len(running_oversampled_data)),
                train_size=int(len(post_os_train_dataset)),
            )

            best_str = f"{best_score_this_thr:.3f}" if best_score_this_thr is not None else "None"
            self.logger.warning(
                f"Oversampled Training size: {len(post_os_train_dataset)}, "
                f"Validation Dice: {post_os_valid_dice:.3f} (best this threshold: {best_str})"
            )
        
            if (best_score_this_thr is None) or (post_os_valid_dice > best_score_this_thr):
                best_score_this_thr  = post_os_valid_dice
                best_model_this_thr = copy.deepcopy(post_os_model)
                best_iter_idx_this_thr = iter_idx
                patience_left = patience
                no_improve_streak = 0

            
            else:
                no_improve_streak += 1
                patience_left -= 1

                if no_improve_streak == 2:
                    per_bin_oversample = min(max(1, math.ceil(per_bin_oversample * 1.5)), max_k)
                elif no_improve_streak == 3:
                    # total ~3x relative to original intent; if you want only 2x total, set 2.0/1.33 appropriately
                    per_bin_oversample = min(max(1, math.ceil(per_bin_oversample * 2.0)), max_k)

                best_str = f"{best_score_this_thr:.4f}" if best_score_this_thr is not None else "-1.0"
                self.logger.warning(
                    f"Validation Dice did not improve (current: {post_os_valid_dice:.4f}, best this threshold: {best_str}). "
                    f"Patience left: {patience_left}"
                )
                
            cleanup_iteration(
                variables={'optimizer': optimizer, 'scheduler': scheduler, 'dataloader': post_os_train_dataloader},
                device=self.device,
                logger=self.logger,
                wait_time=2
            )
            
        if best_model_this_thr is not None:
            # Update evolving best; leave base_model untouched for plotting
            self.best_model = copy.deepcopy(best_model_this_thr)

            ## Plotting
            pre_results = get_area_vs_dice(
                data=self.train_core_data,
                model=self.base_model,
                model_name=self.model_name,
                device=self.device,
                image_size=self.image_size,
                mask_size=self.mask_size
                )
            
            post_results = get_area_vs_dice(
                data=self.train_core_data,
                model=best_model_this_thr,
                model_name=self.model_name,
                device=self.device,
                image_size=self.image_size,
                mask_size=self.mask_size
                )
            

            # title_prefix = f"{self.model_name}-{self.model_config} ({self.env_vars.get('variant')}) | "
            save_name = (
                f"butterfly."
                f"{self.model_name}_{self.model_config}."
                f"{self.env_vars.get('variant')}."
                f"thr_{score_threshold:.3f}.pdf"
            )
            savefig_path = f"{self.folder_path}/{save_name}"

            plot_butterfly_mask_vs_score(
                n_bins=self.n_bins,
                threshold=score_threshold,
                pre_results=pre_results,
                post_results=post_results,
                savefig_path=savefig_path
            )

            test_loss, test_dice, test_iou = test_loop(
                test_dataloader=self.test_dataloader, 
                model=best_model_this_thr, 
                model_name=self.model_name, 
                criterion=self.criterion, 
                device=self.device, 
                num_repeat=(self.num_repeat if self.num_repeat > 0 else None)
            )
            test_loss = float(test_loss)
            test_dice = float(test_dice)
            test_iou  = float(test_iou)

            # Save per-threshold TEST metrics
            self.tracker.set_post_threshold_test(
                thr=float(score_threshold),
                test_loss=float(test_loss),
                test_dice=float(test_dice),
                test_iou=float(test_iou),
                best_iter_index=int(best_iter_idx_this_thr) if best_iter_idx_this_thr is not None else None,
                best_val_dice=float(best_score_this_thr) if best_score_this_thr is not None else None,
            )
            
            self.logger.warning(
                f"Threshold={score_threshold:.2f} | Test Dice={test_dice:.4f}, Test IoU={test_iou:.4f} "
                f"| (best_iter={best_iter_idx_this_thr})"
            )

            if (self.best_score is None) or (test_dice > self.best_score + 1e-6):
                self.best_score = test_dice
                self.logger.warning(
                    f"Updated BEST OVERALL (test Dice={self.best_score:.4f}) "
                    f"at thr={score_threshold:.2f} (iter={best_iter_idx_this_thr})"
                )

    def run_full_pipeline(self):
        self.pre_oversample_train()
        for thr in self.thresholds:
            self.post_oversample_train(score_threshold=float(thr))

    def __exit__(self, exc_type, exc, tb):
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for h in getattr(self.logger, "handlers", []):
                try:
                    h.flush()
                except Exception:
                    pass
        finally:
            return False


if __name__=="__main__":
    interrupted = False
    env_vars = dotenv_values(dotenv_path="./.env")
    file_dir = Path(env_vars["output_folder_path"]) / env_vars["oversample_save_folder_name"]
    Path(file_dir).mkdir(parents=True, exist_ok=True)

    model_name, model_config = str(env_vars["model_name"]), str(env_vars["model_config"])
    job_name = f"{model_name}_{model_config}.{env_vars['variant']}"

    logger = create_logger(
        log_filename=f'train.oversample.{job_name}', 
        file_dir=file_dir
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
        except KeyboardInterrupt:
            interrupted = True
            logger.warning("<<<<<<<<<<<<<<<<<<<< PROCESS INTERRUPTED >>>>>>>>>>>>>>>>>>\n")
        finally:
            if not interrupted:
                logger.warning("<<<<<<<<<<<<<<<<<<<<  PROCESS ENDED  >>>>>>>>>>>>>>>>>>\n")
            for handler in logger.handlers:
                handler.flush()
