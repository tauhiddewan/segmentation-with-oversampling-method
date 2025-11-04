import ast
import torch
import random
import numpy as np
from torchvision import transforms
from utils.metrics import calculate_dice_score

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