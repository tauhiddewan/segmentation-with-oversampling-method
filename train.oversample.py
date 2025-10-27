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
from utils.tracker import ResultTracker  # uses set_pre, add_post_iter, set_post_threshold_test

from utils.train import (
    training_loop, 
    test_loop,
    get_lr_scheduler, 
    cleanup_iteration
)

class BinManager:
    def __init__(self, n_bins, env_vars):
        self.area_bins = np.linspace(0, 1, n_bins+1)
        self.env_vars = env_vars

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
        self.oversampled_data = []
        data_bins = [[] for _ in range(len(self.area_bins)-1)]
        for idx, data in enumerate(train_data):
            mask = self.msk_transform(data[1])
            area_ratio = (mask == 1).sum().item() / mask.numel()

            for j in range(len(self.area_bins)-1):
                lo, hi = self.area_bins[j], self.area_bins[j+1]
                if (area_ratio >= lo) and (area_ratio < hi if j < len(self.area_bins)-2 else area_ratio <= hi):
                    data_bins[j].append(data)

        data_bin_ratios = []
        oversample_pool = [[] for _ in range(len(data_bins))]

        for idx, data_bin in enumerate(data_bins):
            if len(data_bin) == 0:
                data_bin_ratios.append(0)
            else:
                below = 0
                for img, msk in data_bin:
                    img_tensor, msk_tensor = self.img_transform(img), self.msk_transform(msk)
                    img_tensor, msk_tensor = img_tensor.to(device), msk_tensor.to(device)

                    was_training = model.training
                    model.eval()
                    with torch.no_grad():
                        logits = model(img_tensor.unsqueeze(0).to(device))
                        logits = logits[0].squeeze(0) if model_name == "polyp_pvt" else logits.squeeze(0)
                    if was_training:
                        model.train()

                    score = calculate_dice_score(
                        preds=logits, 
                        targets=msk_tensor, 
                        device=device, 
                        model_name=model_name
                    ).item()

                    if score < score_threshold:
                        below += 1
                        oversample_pool[idx].append((img, msk))

                data_bin_ratios.append(below / len(data_bin))

        # ---- Difficulty + Evidence weighting with budgeted allocation ----
        hards_per_bin = np.array([len(oversample_pool[j]) for j in range(len(data_bins))], dtype=float)
        imgs_per_bin  = np.array([len(data_bins[j]) for j in range(len(data_bins))], dtype=float)
        eligible = hards_per_bin > 0
        if not eligible.any():
            return self.oversampled_data

        # Jeffreys-smoothed ratio
        p = (hards_per_bin + 0.5) / (imgs_per_bin + 1.0)

        # knobs (safe defaults)
        alpha = float(self.env_vars.get("oversample_alpha", 1.0))   # weight on ratio p
        beta  = float(self.env_vars.get("oversample_beta",  0.5))   # weight on evidence m
        eps_m = float(self.env_vars.get("oversample_eps_m", 0.5))   # small floor

        # weights
        w = np.zeros_like(p)
        w[eligible] = (np.power(p[eligible], alpha) * np.power(hards_per_bin[eligible] + eps_m, beta))

        if np.all(w == 0):
            return self.oversampled_data

        # total budget K for THIS call
        K = max(1, int(num_oversamples * int(eligible.sum())))

        # normalize to per-bin allocations
        W = w.sum()
        frac = w / W
        k_per_bin = np.maximum(1, np.rint(K * frac)).astype(int)

        # optional per-bin cap
        k_max = int(self.env_vars.get("max_per_bin_oversample", 80))
        k_per_bin = np.minimum(k_per_bin, k_max)

        # sample from each bin with/without replacement
        for j in range(len(data_bins)):
            if not eligible[j]:
                continue
            pool = oversample_pool[j]
            if not pool:
                continue

            k = int(k_per_bin[j])
            if k <= len(pool):
                picks = random.sample(pool, k=k)      # without replacement
            else:
                picks = random.choices(pool, k=k)     # WITH replacement

            self.oversampled_data.extend(picks)

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
        self.env_vars = env_vars
        self.model_name = model_name
        self.model_config = model_config
        self.logger = logger
        self.file_dir = str(file_dir)
        self.models_dir = f'{self.file_dir}/models'
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training hyper-params
        self.best_model = None
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
        self.thresholds = [0.6, 0.7, 0.8, 0.85, 0.90, 0.925, 0.94, 0.96, 0.97]

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
        pre_oversample_save_path = (
            f'{self.models_dir}/pre_oversample.{self.model_name}_{self.model_config}.{self.env_vars["variant"]}.pt'
        )
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
            best_model_save_path=pre_oversample_save_path,
            logger=self.logger,
            save_model=self.save_model
        )

        pre_os_test_loss, pre_os_test_dice, pre_os_test_iou = test_loop(
            test_dataloader=self.test_dataloader, 
            model=self.model, 
            model_name=self.model_name, 
            criterion=self.criterion, 
            device=self.device, 
            num_repeat=(self.num_repeat if self.num_repeat > 0 else None)
        )
        
        self.best_model = copy.deepcopy(pre_os_model)
        self.best_score = float(pre_os_test_dice)
        
        self.tracker.set_pre(
            num_training_images=len(self.pre_os_train_dataset),
            test_loss=float(pre_os_test_loss),
            test_dice=float(pre_os_test_dice),
            test_iou=float(pre_os_test_iou),
        )

        self.logger.warning(
            f'Original Training size: {len(self.pre_os_train_dataset)}, '
            f'Test Dice: {float(pre_os_test_dice):.4f}, Test IoU: {float(pre_os_test_iou):.4f}'
        )

    def post_oversample_train(self, score_threshold: float):
        iter_idx = 0
        best_model_this_thr = None
        best_score_this_thr = None
        best_iter_idx_this_thr = None
        per_bin_oversample = self.num_oversamples
        no_improve_streak = 0
        max_k = int(self.env_vars.get("max_per_bin_oversample", 50))

        self.logger.warning(f"Iterative oversampling step (thr={score_threshold})")
        post_oversample_save_path = (
            f'{self.models_dir}/post_oversample.{self.model_name}_{self.model_config}.{self.env_vars["variant"]}_{score_threshold}.pt'
        )

        # Start from base model for this threshold
        post_os_model = copy.deepcopy(self.best_model).to(self.device)
        patience = int(self.env_vars.get("patience", 3))
        patience_left = patience
        
        running_oversampled_data = []
        max_total_oversamples = int(self.env_vars.get("max_total_oversamples", "1500"))

        while patience_left > 0:
            iter_idx += 1
            self.best_model_path = (
                f'{self.models_dir}/best_model_{score_threshold}.{self.model_name}_{self.model_config}.{self.env_vars["variant"]}.pt'
            )

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
                    per_bin_oversample = min(int(per_bin_oversample * 1.5), max_k)

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
            # warm start at next threshold 
            self.best_model = copy.deepcopy(best_model_this_thr)

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

            # Track global best (optional; no JSON write since your schema doesn't include overall best)
            if (self.best_score is None) or (test_dice > self.best_score + 1e-6):
                self.best_score = test_dice
                torch.save({'model_state_dict': self.best_model.state_dict()}, self.best_model_path)
                self.logger.warning(
                    f"Updated BEST OVERALL (test Dice={self.best_score:.4f}) "
                    f"at thr={score_threshold:.2f} (iter={best_iter_idx_this_thr})"
                )

    def run_full_pipeline(self):
        self.pre_oversample_train()
        for thr in self.thresholds:
            self.post_oversample_train(score_threshold=thr)

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
    Path(env_vars["output_folder_path"]).mkdir(parents=True, exist_ok=True)
    model_name, model_config = str(env_vars["model_name"]), str(env_vars["model_config"])
    job_name = f"{model_name}_{model_config}.{env_vars['variant']}"

    file_dir = Path(env_vars["output_folder_path"]) / env_vars["oversample_save_folder_name"]

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
        except KeyboardInterrupt:
            interrupted = True
            logger.warning("<<<<<<<<<<<<<<<<<<<< PROCESS INTERRUPTED >>>>>>>>>>>>>>>>>>\n")
        finally:
            if not interrupted:
                logger.warning("<<<<<<<<<<<<<<<<<<<<  PROCESS ENDED  >>>>>>>>>>>>>>>>>>\n")
            for handler in logger.handlers:
                handler.flush()
