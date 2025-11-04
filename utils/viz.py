import os
import ast
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.metrics import calculate_dice_score

def get_area_vs_dice(data, model, model_name, device, image_size, mask_size):
    img_transform = transforms.Compose([
        transforms.Resize(ast.literal_eval(str(image_size)), transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])
    msk_transform = transforms.Compose([
        transforms.Resize(ast.literal_eval(str(mask_size)), transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    model.eval()
    results = []
    with torch.no_grad():
        for img, msk in data:
            img_t = img_transform(img).to(device)
            msk_t = msk_transform(msk).to(device)
            area_ratio = (msk_t == 1).sum().item() / msk_t.numel()

            preds = model(img_t.unsqueeze(0))
            if isinstance(preds, (tuple, list)):
                preds = preds[0]
            preds = preds.squeeze(0)

            dice = calculate_dice_score(
                preds=preds, targets=msk_t, device=device, model_name=model_name
            ).item()

            results.append((area_ratio, dice))
    return results

def _butterfly_data(results, threshold, n_bins):
    mask_bins = np.linspace(0.0, 1.0, n_bins + 1)
    data_bins = [[] for _ in range(n_bins)]
    for area, score in results:
        for j in range(n_bins):
            lo, hi = mask_bins[j], mask_bins[j + 1]
            if (area >= lo) and (area < hi if j < n_bins - 1 else area <= hi):
                data_bins[j].append(score)
                break
    below_counts = [sum(s < threshold for s in bin_scores) for bin_scores in data_bins]
    above_counts = [sum(s >= threshold for s in bin_scores) for bin_scores in data_bins]
    return mask_bins, below_counts, above_counts

def plot_butterfly_mask_vs_score(n_bins, threshold, pre_results, post_results, savefig_path, title_prefix=""):
    mask_bins_pre, pre_below, pre_above = _butterfly_data(pre_results, threshold, n_bins)
    mask_bins_post, post_below, post_above = _butterfly_data(post_results, threshold, n_bins)

    bar_width = 0.35
    index = np.arange(n_bins)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    # LEFT: PRE
    ax = axes[0]
    ax.bar(index, -np.array(pre_below), bar_width, label=f"< Dice {threshold}")
    ax.bar(index,  np.array(pre_above), bar_width, label=f">= Dice {threshold}")
    ax.axhline(y=0, linestyle="--")
    ax.set_title(f"{title_prefix} Pre-Oversample (t={threshold})")

    for i in range(n_bins):
        ax.text(i, -pre_below[i] - 0.5, str(pre_below[i]), ha='center', va='top', fontsize=8)
        ax.text(i,  pre_above[i] + 0.5, str(pre_above[i]), ha='center', va='bottom', fontsize=8)

    ax.set_xlabel("Mask Area Ratio Bins")
    ax.set_ylabel("Dice Score Counts")
    ax.set_xticks(index)
    ax.set_xticklabels([f"{mask_bins_pre[i]:.2f}-{mask_bins_pre[i+1]:.2f}" for i in range(n_bins)], rotation=45)

    # RIGHT: POST
    ax = axes[1]
    ax.bar(index, -np.array(post_below), bar_width, label=f"< Dice {threshold}")
    ax.bar(index,  np.array(post_above), bar_width, label=f">= Dice {threshold}")
    ax.axhline(y=0, linestyle="--")
    ax.set_title(f"{title_prefix} Post-Oversample (t={threshold})")

    for i in range(n_bins):
        ax.text(i, -post_below[i] - 0.5, str(post_below[i]), ha='center', va='top', fontsize=8)
        ax.text(i,  post_above[i] + 0.5, str(post_above[i]), ha='center', va='bottom', fontsize=8)

    ax.set_xlabel("Mask Area Ratio Bins")
    ax.set_xticks(index)
    ax.set_xticklabels([f"{mask_bins_post[i]:.2f}-{mask_bins_post[i+1]:.2f}" for i in range(n_bins)], rotation=45)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs(os.path.dirname(savefig_path), exist_ok=True)
    fig.savefig(savefig_path, format="pdf", bbox_inches="tight")
    plt.close(fig)




