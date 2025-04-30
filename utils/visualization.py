import os
import ast
import copy
import atexit
import logging
import json
import torch
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from email import encoders
from dotenv import dotenv_values
from utils.model import select_model
from utils.metrics import calculate_dice_score, calculate_iou_score
from utils.dataset import KvasirDataset
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def plot_training_results(data, savefig_path):
    num_subsets = len(data)
    fig, axes = plt.subplots(num_subsets, 2, figsize=(16, 6 * num_subsets))
    
    fig.suptitle("Semi-Automated Segmentaion Training Results", fontsize=20)
    for i, (subset_key, subset) in enumerate(data.items()):
        loss = subset['normal_training_loss_per_epoch'] + subset['semi_auto_training_loss_per_epoch']
        ma_loss = subset['normal_training_ma_loss_per_epoch'] + subset['semi_auto_training_ma_loss_per_epoch']
        score = subset['normal_training_score_per_epoch'] + subset['semi_auto_training_score_per_epoch']
        
        epochs = range(1, len(loss) + 1)
        normal_epochs = len(subset['normal_training_loss_per_epoch'])
        
        training_subset_pct = int(subset_key[14:])
        pseudo_subset_pct = 80 - training_subset_pct
        
        # Plot Loss
        axes[i, 0].plot(epochs[:normal_epochs], loss[:normal_epochs], 'b-', label='Normal Training Loss')
        axes[i, 0].plot(epochs[normal_epochs-1:], loss[normal_epochs-1:], 'b--', label='Semi-Auto Training Loss')
        axes[i, 0].plot(epochs[:normal_epochs], ma_loss[:normal_epochs], 'r-', label='Normal Training MA Loss')
        axes[i, 0].plot(epochs[normal_epochs-1:], ma_loss[normal_epochs-1:], 'r--', label='Semi-Auto Training MA Loss')
        
        axes[i, 0].axvline(x=normal_epochs, color='k', linestyle=':')
        text_x_offset = 0.1
        axes[i, 0].text(normal_epochs + text_x_offset, 0.5, '', rotation=90,
                        verticalalignment='center', transform=axes[i, 0].get_xaxis_transform())
        
        axes[i, 0].set_title(f'Training Loss [{training_subset_pct}% Training + {pseudo_subset_pct}% Pseudo]')
        axes[i, 0].set_xlabel('Epochs')
        axes[i, 0].set_ylabel('Loss')
        axes[i, 0].legend()
        axes[i, 0].grid(True)
        
        # Plot Score
        axes[i, 1].plot(epochs[:normal_epochs], score[:normal_epochs], 'g-', label='Normal Training Score')
        axes[i, 1].plot(epochs[normal_epochs-1:], score[normal_epochs-1:], 'g--', label='Semi-Auto Training Score')
        
        axes[i, 1].axvline(x=normal_epochs, color='k', linestyle=':')
        axes[i, 1].text(normal_epochs + text_x_offset, 0.5, '', rotation=90,
                        verticalalignment='center', transform=axes[i, 1].get_xaxis_transform())
        
        axes[i, 1].set_title(f'Training Score [{training_subset_pct}% Training + {pseudo_subset_pct}% Pseudo]')
        axes[i, 1].set_xlabel('Epochs')
        axes[i, 1].set_ylabel('Score')
        axes[i, 1].legend()
        axes[i, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{savefig_path}/sas_train_results.pdf", format="pdf", bbox_inches="tight")
    plt.close()

def plot_test_results(benchmark_score,
                      metric_name,
                      split_list, 
                      split_label_list,
                      split_score_list_pre, 
                      split_score_list_post,
                      savefig_name):
    plt.figure(figsize=(15,7))

    plt.axhline(y=benchmark_score, color='g', linestyle='-', label=f"Benchmark Score ({benchmark_score*100:.2f}%)")
    plt.axhline(y=0.90 * benchmark_score, color='r', linestyle='--', label=f"90% Benchmark Score ({0.90*benchmark_score*100:.2f}%)")

    plt.scatter(split_list, split_score_list_pre, color='blue', marker='o', s=100, label="Before Adding Generated Masks")
    plt.scatter(split_list, split_score_list_post, color='orange', marker='x', s=100, label="After Adding Generated Masks")
    plt.plot(split_list, split_score_list_pre, color='blue', linestyle='-', linewidth=1)
    plt.plot(split_list, split_score_list_post, color='orange', linestyle='-', linewidth=1)

    plt.xticks(split_list, split_label_list)
    plt.title('Semi-Automated Segmentaion Test Results', fontsize=15)
    plt.xlabel('Real & Generated Dataset Split', fontsize=12)
    plt.ylabel(f'{metric_name} Score', fontsize=12)

    plt.ylim(0.4, .9) #max(benchmark_score, max(split_score_list_pre))+.05
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.legend(loc="upper right")
    plt.tight_layout()

    plt.savefig(f"{savefig_name}", format="pdf")
    plt.close()

def get_mask_quality_pct(data, save_path):
    keys = sorted(list(data.keys()))  
    color_zero = "#E63946"      # Red
    color_avg = "#F4A261"       # Orange
    color_good = "#457B9D"      # Blue
    color_very_good = "#2A9D8F" # Green

    plt.figure(figsize=(14, 10))
    
    zero_counts, avg_counts, good_counts, very_good_counts = [], [], [], []
    
    for idx, key in enumerate(keys):
        sorted_dice_scores = sorted(data[key]["dice_score"], key=lambda x: x[1])
        ids, scores = zip(*sorted_dice_scores)
        scores = [score.cpu().numpy() for score in scores]
        
        zero_count = sum(1 for score in scores if score <= 0.0)
        avg_count = sum(1 for score in scores if 0.0 < score <= 0.5)
        good_count = sum(1 for score in scores if 0.5 < score <= 0.8)
        very_good_count = sum(1 for score in scores if score > 0.8)
        
        zero_counts.append(zero_count)
        avg_counts.append(avg_count)
        good_counts.append(good_count)
        very_good_counts.append(very_good_count)
    
    x = np.arange(len(keys))
    bar_width = 0.8
    
    plt.bar(x, zero_counts, width=bar_width, label='Score ≤ 0.0', color=color_zero)
    plt.bar(x, avg_counts, width=bar_width, bottom=zero_counts, label='0.0 < Score ≤ 0.5', color=color_avg)
    plt.bar(x, good_counts, width=bar_width, 
            bottom=np.array(zero_counts) + np.array(avg_counts), label='0.5 < Score ≤ 0.8', color=color_good)
    plt.bar(x, very_good_counts, width=bar_width, 
            bottom=np.array(zero_counts) + np.array(avg_counts) + np.array(good_counts), label='Score > 0.8', color=color_very_good)
    
    for i, (zero_count, avg_count, good_count, very_good_count) in enumerate(zip(zero_counts, avg_counts, good_counts, very_good_counts)):
        total = zero_count + avg_count + good_count + very_good_count

        zero_pct = (zero_count / total) * 100 if total > 0 else 0
        avg_pct = (avg_count / total) * 100 if total > 0 else 0
        good_pct = (good_count / total) * 100 if total > 0 else 0
        very_good_pct = (very_good_count / total) * 100 if total > 0 else 0

        counts = [zero_count, avg_count, good_count, very_good_count]
        percentages = [zero_pct, avg_pct, good_pct, very_good_pct]
        bottoms = [0, zero_count, zero_count + avg_count, zero_count + avg_count + good_count]

        for j, (count, percentage, bottom) in enumerate(zip(counts, percentages, bottoms)):
            if percentage > 5:  
                label_y = bottom + count / 2
                plt.text(i, label_y, f"{percentage:.1f}%", ha='center', va='center', color="white", fontsize=8)
            elif percentage > 0:
                label_y = bottom + count / 2
                plt.text(i, label_y, f"{percentage:.1f}%", ha='center', va='center', color="white", fontsize=6)



    plt.xticks(x, [str(int(key)*10) for key in keys], rotation=45)  
    plt.yticks(np.arange(0, 701, 100)) 
    plt.ylim(0, 700)
    plt.xlabel('Training Image Size')
    plt.ylabel('Number of Generated Masks')
    plt.title('Distribution of Generated Mask Scores Across Training Sizes')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/gen_mask_quality_dist.pdf', format="pdf", bbox_inches="tight")
    # plt.show()
    plt.close()

def get_dataset(image_size, mask_size):
    dataset = []
    env_vars = dotenv_values(dotenv_path="./.env")
    subset_files = os.listdir(f'{env_vars["subset_folder_path"]}')
    for subset_name in subset_files:
        with open(f'{env_vars["subset_folder_path"]}/{subset_name}', "rb") as f:
            dataset.extend(pickle.load(f))
    full_dataset = KvasirDataset(data=dataset, 
                                mode="test", 
                                image_size=image_size, 
                                mask_size=mask_size)
    return full_dataset

def generate_mask(model, 
                  model_name,
                  data,  
                  device):
    model.eval() 
    model.to(device)
    gen_dataset, real_masks, dice_scores = [], [], []
    with torch.no_grad(): 
            image, label = data
            image_batched = image.unsqueeze(0).to(device)
            label_batched = label.unsqueeze(0).to(device)
            logits = model(image_batched) 
            
            dice_score = calculate_dice_score(preds=logits, targets=label, device=device, model_name=model_name)
            miou_score = calculate_iou_score(preds=logits, targets=label, device=device, model_name=model_name)
            
            if model_name!="unet":
                logits = torch.sigmoid(logits)
                
            gen_mask_batched = (logits > 0.5).float() 
            gen_mask  = gen_mask_batched.squeeze(0).cpu()
            
    return gen_mask, dice_score, miou_score

def get_sas_modelwise_results(data, model_dir, model_name, model_config, device):
    model = select_model(model_name=model_name, model_config=model_config)
    model_files = sorted(os.listdir(model_dir))
    
    results = {}
    for fname in model_files:
        fname_parts = fname.split(".")
        sas_status = fname_parts[0]
        split_key = fname_parts[-2].split("%")[0]
        
        if split_key not in results:
            results[split_key] = {"presas": {}, 
                                  "postsas" : {}}
        
        sas_model = copy.deepcopy(model)
        checkpoint = torch.load(f"{model_dir}/{fname}", weights_only=True)
        sas_model.load_state_dict(checkpoint['model_state_dict'])
        
        gen_mask, dice_score, miou_score = generate_mask(data=data, 
                                                         model=sas_model, 
                                                         model_name=model_name, 
                                                         device=device)
        
        if sas_status=="presas":
            results[split_key]["presas"]["gen_mask"] = gen_mask
            results[split_key]["presas"]["dice_score"] = dice_score
            results[split_key]["presas"]["miou_score"] = miou_score
        else:
            results[split_key]["postsas"]["gen_mask"] = gen_mask
            results[split_key]["postsas"]["dice_score"] = dice_score
            results[split_key]["postsas"]["miou_score"] = miou_score
            
            
    return results

def plot_modelwise_comparison(image_data, results, savefig_name):
    train_stages = list(results.keys())
    train_percentages = [int(stage.split('_')[1]) for stage in train_stages]
    remaining_percentages = train_percentages[::-1]  # Reverse order

    fig, axes = plt.subplots(2, len(train_stages) + 2, figsize=(20, 6))

    axes[0, 0].imshow(image_data[0].permute(1, 2, 0).numpy(), cmap='gray')
    axes[0, 0].set_title('Real Image')
    axes[0, 0].axis('off')
    axes[1, 0].imshow(image_data[0].permute(1, 2, 0).numpy(), cmap='gray')
    axes[1, 0].set_title('Real Image')
    axes[1, 0].axis('off')

    axes[0, 1].imshow(image_data[1].squeeze(), cmap='gray')
    axes[0, 1].set_title('Real Mask')
    axes[0, 1].axis('off')
    axes[1, 1].imshow(image_data[1].squeeze(), cmap='gray')
    axes[1, 1].set_title('Real Mask')
    axes[1, 1].axis('off')

    for i, (stage, train_p, rem_p) in enumerate(zip(train_stages, train_percentages, remaining_percentages)):
        title_suffix = f'{train_p}%:{rem_p}%'

        axes[0, i + 2].imshow(results[stage]['presas']["gen_mask"].squeeze().numpy(), cmap='gray')
        axes[0, i + 2].set_title(f'Pre-SISM - {title_suffix}')
        axes[0, i + 2].axis('off')

        axes[1, i + 2].imshow(results[stage]['postsas']["gen_mask"].squeeze().numpy(), cmap='gray')
        axes[1, i + 2].set_title(f'Post-SISM - {title_suffix}')
        axes[1, i + 2].axis('off')

    plt.tight_layout()
    plt.savefig(f'{savefig_name}', format="pdf", bbox_inches="tight")
    plt.close()



def plot_dice_score_vs_data_percent(data, model_name, model_config, save_path, benchmark_dice_score):
    categories = {
        "presas": "Pre-SISM",
        "postsas": "Post-SISM"
    }
    percentages = ['10%', '20%', '30%', '40%', '50%', '60%', '70%']
    
    bins = np.linspace(0, 1, 11)  
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    
    for ax, (key, title) in zip(axes, categories.items()):

        for i, percentage in enumerate(percentages):
            scores = sorted(np.array(data[key][percentage]))
            
            percent_data = np.linspace(0, 100, len(scores))
        
            if key == "presas":
                line, = ax.plot(scores, percent_data, label=f"Original Data: {percentage}")
            else:
                line, = ax.plot(scores, percent_data, label=f"Original Data: {percentage} + Generated Data: {percentages[-(i+1)]}")
            
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Dice Score", fontsize=12)
        ax.set_ylabel("Percentage of Data (%)", fontsize=12)
        
        ax.grid(True, alpha=0.3)
        ax.set_xticks(bins)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_xticklabels([f"{x:.1f}" for x in bins])
        ax.set_ylim(0, 100)
        
        ax.legend(loc='upper left')
        
    plt.suptitle(f'Percentage Distribution of Dice Scores for {model_name}-{model_config}',
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.legend()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()


def plot_dice_score_entropy(data, model_name, model_config, save_path):
    categories = {"presas": "Pre-SISM", "postsas": "Post-SISM"}
    percentages = ['10%', '20%', '30%', '40%', '50%', '60%', '70%']
    bins = np.arange(0, 1.1, 0.1)

    plt.figure(figsize=(12, 18))

    all_entropies = []
    for key in categories.keys():
        entropy_values = []

        for i in range(len(percentages)):
            scores = np.array(sorted(data[str(key)][percentages[i]]))
            counts, _ = np.histogram(scores, bins=bins)
            total_scores = len(scores)
            probabilities = counts / total_scores
            entropy = -np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0]))
            
            entropy_values.append(entropy)
        
        all_entropies.append(entropy_values)
        
        plt.plot(percentages, entropy_values, marker='o', linestyle='-', label=f"Entropy - {categories[key]}")

    plt.title(f"Entropy Comparison for {model_name}-{model_config}", fontsize=14)
    plt.xlabel("Percentage (%) of Trained Data Used for Model Development", fontsize=12)
    plt.ylabel("Entropy (H)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(percentages)
    plt.legend()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    
    plt.close()
    
    return all_entropies


def multi_model_fit(entropy_values, save_fname, num_data_point):
    def exp_decay(x, a, b):
        return a * np.exp(-b * x)
    
    def exp_decay_offset(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    def power_law(x, a, b):
        return a * (x ** -b)
    
    def logarithmic(x, a, b, c):
        return a - b * np.log(x) + c
    
    def sigmoid(x, a, b, c, d):
        return d + (a - d) / (1 + np.exp(b * (x - c)))

    percentages = np.array([10, 20, 30, 40, 50, 60, 70])
    X = np.arange(1, len(entropy_values) + 1)
    y = np.array(entropy_values)
    X_train = X[:num_data_point]
    y_train = y[:num_data_point]
    X_future = X[num_data_point:]
    y_future = y[num_data_point:]
    
    train_percentages = percentages[:num_data_point]
    future_percentages = percentages[num_data_point:]
    
    models = {
        'Exponential Decay': {
            'func': exp_decay,
            'p0': [y_train[0], 0.5],
            'bounds': ([0, 0], [np.inf, np.inf]),
            'equation': lambda p: f'y = {p[0]:.3f} * e^(-{p[1]:.3f}x)'
        },
        'Exponential with Offset': {
            'func': exp_decay_offset,
            'p0': [y_train[0], 0.5, min(y_train)],
            'bounds': ([0, 0, 0], [np.inf, np.inf, np.inf]),
            'equation': lambda p: f'y = {p[0]:.3f} * e^(-{p[1]:.3f}x) + {p[2]:.3f}'
        },
        'Power Law': {
            'func': power_law,
            'p0': [y_train[0], 0.5],
            'bounds': ([0, 0], [np.inf, np.inf]),
            'equation': lambda p: f'y = {p[0]:.3f} * x^(-{p[1]:.3f})'
        },
        'Logarithmic': {
            'func': logarithmic,
            'p0': [max(y_train), 0.5, min(y_train)],
            'bounds': ([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
            'equation': lambda p: f'y = {p[0]:.3f} - {p[1]:.3f} * ln(x) + {p[2]:.3f}'
        },
        'Sigmoid': {
            'func': sigmoid,
            'p0': [max(y_train), 1, np.mean(X_train), min(y_train)],
            'bounds': ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]),
            'equation': lambda p: f'y = {p[3]:.3f} + ({p[0]:.3f} - {p[3]:.3f})/(1 + e^({p[1]:.3f}(x - {p[2]:.3f})))'
        }
    }
    
    results = {}
    
    for name, model in models.items():
        try:
            popt, _ = curve_fit(model['func'], X_train, y_train, 
                              p0=model['p0'],
                              bounds=model['bounds'],
                              maxfev=10000)
            
            y_pred = model['func'](X_train, *popt)
            y_future_pred = model['func'](X_future, *popt)
            
            r2 = r2_score(y_train, y_pred)
            mse = mean_squared_error(y_train, y_pred)
            future_mse = mean_squared_error(y_future, y_future_pred)
            
            results[name] = {
                'popt': popt,
                'r2': r2,
                'mse': mse,
                'future_mse': future_mse,
                'func': model['func'],
                'equation': model['equation'](popt)
            }
        except RuntimeError as e:
            print(f"Fitting error for {name}: {e}")
            continue
    
    plt.figure(figsize=(15, 10))
    
    plt.scatter(train_percentages, y_train, label="Training Data", color='red', marker='o')
    plt.scatter(future_percentages, y_future, label="Future Data", color='maroon', marker='x')
    
    x_smooth_percent = np.linspace(percentages[0], percentages[-1], 100)
    x_smooth_orig = (x_smooth_percent - percentages[0]) / 10 + 1
    colors = ['blue', 'green', 'purple', 'orange', 'black']
    
    for (name, result), color in zip(results.items(), colors):
        y_smooth = result['func'](x_smooth_orig, *result['popt'])
        plt.plot(x_smooth_percent, y_smooth, '--', color=color,
                label=f'{name}(R² = {result["r2"]:.3f}):\n{result["equation"]}\n')
    
    plt.grid(True, alpha=0.3)
    plt.title('Model Comparison')
    plt.xlabel('% of Original Data Used For Model Development')
    plt.ylabel('Entropy Values')
    plt.xticks(percentages, [f'{p}%' for p in percentages])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    
    plt.savefig(f"outputs/figures/model_comparison_{save_fname}.pdf", 
                bbox_inches='tight')
    plt.close()
    
    summary = {name: {
        'R2': result['r2'],
        'Training MSE': result['mse'],
        'Future MSE': result['future_mse'],
        'Equation': result['equation']
    } for name, result in results.items()}
    
    return summary



def fit_exp_offset(entropy_values, save_fname, num_data_point):
    def exp_offset(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    percentages = np.array([10, 20, 30, 40, 50, 60, 70])
    X = np.arange(1, len(entropy_values) + 1)
    y = np.array(entropy_values)
    X_train = X[:num_data_point]
    y_train = y[:num_data_point]
    X_future = X[num_data_point:]
    y_future = y[num_data_point:] if len(y) > num_data_point else []
    
    train_percentages = percentages[:num_data_point]
    future_percentages = percentages[num_data_point:]
    
    c_guess = min(y)
    a_guess = max(y) - c_guess
    b_guess = 0.5
    
    bounds = (
        [0.1, 0.01, c_guess - 0.5],
        [5.0, 2.0, c_guess + 0.5]
    )
    
    popt, pcov = curve_fit(exp_offset, X_train, y_train, 
                          p0=[a_guess, b_guess, c_guess],
                          bounds=bounds,
                          maxfev=10000)
    
    y_pred = exp_offset(X_train, *popt)
    y_future_pred = exp_offset(X_future, *popt) if len(X_future) > 0 else []
    r2 = r2_score(y_train, y_pred)
    
    plt.figure(figsize=(12, 8))
    
    plt.scatter(train_percentages, y_train, label="Training Data", color='red', marker='o', s=100)
    if len(y_future) > 0:
        plt.scatter(future_percentages, y_future, label="Future Data", color='maroon', marker='x', s=100)
    
    x_smooth_percent = np.linspace(percentages[0], percentages[-1], 100)
    x_smooth_orig = (x_smooth_percent - percentages[0]) / 10 + 1
    y_smooth = exp_offset(x_smooth_orig, *popt)
    
    equation = f'y = {popt[0]:.3f} * e^(-{popt[1]:.3f}x) + {popt[2]:.3f}'
    plt.plot(x_smooth_percent, y_smooth, '--', color='blue',
            label=f'Exponential Fit with Offset (R² = {r2:.3f}):\n{equation}\n')
    
    if len(y_future_pred) > 0:
        plt.scatter(future_percentages, y_future_pred, label="Predicted Future", 
                   color='green', marker='x', s=100)
    
    plt.grid(True, alpha=0.3)
    plt.title('Optimal Training Set Size Estimation')
    plt.xlabel('% of Original Data Used For Model Development')
    plt.ylabel('Entropy Values')
    plt.xticks(percentages, [f'{p}%' for p in percentages])
    plt.legend(fontsize=10)
    
    plt.savefig(f"outputs/figures/exp_offset_{save_fname}.pdf",
                bbox_inches='tight', dpi=300)
    plt.close()
    
    return popt, r2, y_future_pred


def fit_exp_offset_dual(entropy_values1, entropy_values2, save_fname, num_data_point):
    def exp_offset(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    percentages = np.array([10, 20, 30, 40, 50, 60, 70])
    X = np.arange(1, len(entropy_values1) + 1)
    
    def fit_model(entropy_values):
        y = np.array(entropy_values)
        X_train = X[:num_data_point]
        y_train = y[:num_data_point]
        X_future = X[num_data_point:]
        y_future = y[num_data_point:] if len(y) > num_data_point else []
        
        c_guess = min(y)
        a_guess = max(y) - c_guess
        b_guess = 0.5
        
        bounds = ([0.1, 0.01, c_guess - 0.5], [5.0, 2.0, c_guess + 0.5])
        
        popt, _ = curve_fit(exp_offset, X_train, y_train, 
                            p0=[a_guess, b_guess, c_guess],
                            bounds=bounds,
                            maxfev=10000)
        
        y_pred = exp_offset(X_train, *popt)
        y_future_pred = exp_offset(X_future, *popt) if len(X_future) > 0 else []
        r2 = r2_score(y_train, y_pred)
        
        return popt, r2, y_future_pred, X_train, y_train, X_future, y_future
    
    # Fit models for both entropy sets
    popt1, r2_1, y_future_pred1, X_train1, y_train1, X_future1, y_future1 = fit_model(entropy_values1)
    popt2, r2_2, y_future_pred2, X_train2, y_train2, X_future2, y_future2 = fit_model(entropy_values2)

    # Plotting
    plt.figure(figsize=(12, 8))

    # Scatter points for both models
    plt.scatter(percentages[:num_data_point], y_train1, label="Segformer Observed Entropy", color='red', marker='o', s=100)
    plt.scatter(percentages[:num_data_point], y_train2, label="U-Net Observed Entropy", color='blue', marker='*', s=100)
    
    if len(y_future1) > 0:
        plt.scatter(percentages[num_data_point:], y_future1, label="Segformer Future Entropy", color='#8B0000', marker='x', s=100)
        plt.scatter(percentages[num_data_point:], y_future2, label="U-Net Future Entropy", color='#00008B', marker='*', s=100)
    
    # Smooth curve fitting for both models
    x_smooth_percent = np.linspace(percentages[0], percentages[-1], 100)
    x_smooth_orig = (x_smooth_percent - percentages[0]) / 10 + 1
    y_smooth1 = exp_offset(x_smooth_orig, *popt1)
    y_smooth2 = exp_offset(x_smooth_orig, *popt2)

    # Plot fitted curves
    plt.plot(x_smooth_percent, y_smooth1, '--', color='#D62728',
             label=f'Segformer Fit: y = {popt1[0]:.3f} * e^(-{popt1[1]:.3f}x) + {popt1[2]:.3f}')
    plt.plot(x_smooth_percent, y_smooth2, '--', color='#1F77B4',
             label=f'U-Net Fit: y = {popt2[0]:.3f} * e^(-{popt2[1]:.3f}x) + {popt2[2]:.3f}')

    # Predicted future points
    if len(y_future_pred1) > 0:
        plt.scatter(percentages[num_data_point:], y_future_pred1, label="Segformer Predicted Future Entropy", color='#FF7F0E', marker='x', s=100)
        plt.scatter(percentages[num_data_point:], y_future_pred2, label="UNet Predicted Future Entropy", color='#9467BD', marker='*', s=100)
    
    plt.grid(True, alpha=0.3)
    # plt.title('Optimal Training Set Size Estimation With Asymptotic Exponential Decay')
    plt.xlabel('% of Expert-Annotated Masks Used For Model Development')
    plt.ylabel('Entropy Values')
    plt.xticks(percentages, [f'{p}%' for p in percentages])
    plt.legend(fontsize=10)
    
    plt.savefig(f"outputs/figures/exp_offset_{save_fname}.pdf",
                bbox_inches='tight', dpi=300)
    plt.close()

    return popt1, r2_1, y_future_pred1, popt2, r2_2, y_future_pred2
