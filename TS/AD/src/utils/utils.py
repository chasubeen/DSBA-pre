import numpy as np
import torch
from tqdm import tqdm
import time
import logging
import os
import glob
import json
import shutil
import random
import pandas as pd
import matplotlib.pyplot as plt
import sys

def clean_str(x):
    return str(x).replace('[', '').replace(']', '').replace("'", '').replace(',', '').strip()

def save_directory(savedir: str,
                    dataname: str,
                    modelname: str,
                    bank_name: str,
                    target: str,
                    exp_name: str):
    
    if dataname == 'bank':
        # make save directory
        bank_name_str = clean_str(bank_name)
        target_str = clean_str(target)

        savedir = os.path.join(
            savedir,
            f"{modelname}({bank_name_str}_{target_str})",
            exp_name
        )
    else:
        savedir = os.path.join(
            savedir,
            modelname,
            exp_name,
            dataname
        )
    return savedir

def version_build(accelerator, logdir: str, resume: int=None) -> str:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if not os.path.isdir(logdir):
            os.makedirs(logdir)
    if resume is None:
        version = len(os.listdir(logdir))
        load_dir = os.path.join(logdir, f'version{version}')
        save_dir = os.path.join(logdir, f'version{version}')
    else:
        version = resume
        load_dir = os.path.join(logdir, f'version{version}')
        save_dir = os.path.join(logdir, f'version{len(os.listdir(logdir))}')
        
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(save_dir)
    return load_dir, save_dir

class Float32Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)

def check_graph(xs, att, ts, piece=1, threshold=None):
    try:
        ts = pd.to_datetime(ts.astype(str), format='%Y%m%d')
        is_datetime = True
    except:
        is_datetime = False

    l = xs.shape[0]
    chunk = (l + piece - 1) // piece
    fig, axs = plt.subplots(piece, figsize=(16, 4 * piece))

    if piece == 1:
        axs = [axs]

    for i in range(piece):
        L = i * chunk
        R = min((i + 1) * chunk, l) 
        xticks = range(L, R)

        pastel_blue = '#AEC6CF'
        pastel_red = '#FFB6B6'
        pastel_orange = '#FFDAB9'

        axs[i].plot(xticks, xs[L:R], label='Anomaly Score', color=pastel_blue, linewidth=2)

        if threshold is not None:
            axs[i].axhline(y=threshold, color=pastel_red, linestyle='--', linewidth=2, label='Threshold')

        for j in xticks:
            if att[j] == 1 and j + 1 < len(xs):
                axs[i].axvspan(j, j + 1, color=pastel_orange, alpha=0.3)
            elif att[j] == 1 and j + 1 == len(xs):
                axs[i].axvspan(j, j, color=pastel_orange, alpha=0.3)

        tick_locs = np.linspace(L, R - 1, min(6, R - L), dtype=int)
        if is_datetime:
            tick_labels = ts[tick_locs].strftime('%Y-%m-%d')
        else:
            tick_labels = ts[tick_locs].astype(str)

        axs[i].set_xticks(tick_locs)
        axs[i].set_xticklabels(tick_labels)

        axs[i].set_title(f"Segment {i+1}: Index {L} ~ {R - 1}", fontsize=14)
        axs[i].legend()
        axs[i].set_xlabel('Timestamp', fontsize=12)
        axs[i].set_ylabel('Anomaly Score', fontsize=12)
        axs[i].grid(alpha=0.3)

    plt.tight_layout()
    return fig

def load_resume_model(model, loaddir: str, resume: int, pre_training: bool, fine_tuning_method: str):
    new_weights = torch.load(os.path.join(loaddir,'best_model.pt'))
    
    if not pre_training:
        # List of weights to be removed
        weights_to_remove = [
            'TFTpart2.recon_layer.weight',
            'TFTpart2.recon_layer.bias',
            'TFTpart2.mlp.0.weight',
            'TFTpart2.mlp.0.bias'
        ]
        # Remove the specified weights
        for weight in weights_to_remove:
            if weight in new_weights:
                del new_weights[weight]
                
    # load weights
    model.load_state_dict(new_weights, strict=False)

    if not pre_training:
        if fine_tuning_method == 'full':
            for param in model.parameters():
                param.requires_grad = True
            print(f'Fine tuning method: Full')
        elif fine_tuning_method == 'linear_probing':
            train_params = ['final_layers']
            for name, param in model.named_parameters():
                if any(x in name for x in train_params):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print(f'Fine tuning method: Linear probing')
    print('load model from (version {})'.format(resume))