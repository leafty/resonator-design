import sys
sys.path.append('../')
sys.path.append('../src/')

from data_io import get_spiral_data
from daaad.src.learning_model.models.cvae import CondVAEModel
from daaad.src.utils import rec_concat_dict
from torch.utils.data import DataLoader

import pytorch_lightning as pl
import torch
from typing import Union, Dict, List, Tuple
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def train_cvae(num_modes: int = 2, data_fraction: float=1, only_freqs: bool=False, model_name: str='spiral', 
               batch_size: int=128, val_split: float=.1) -> None:
    dataset, _ = get_spiral_data(num_modes=num_modes, data_fraction=data_fraction, only_freqs=only_freqs)
    train_gen, val_gen, test_gen = dataset.get_data_loaders(batch_size, val_split=val_split, test_split=.1)
    callbacks = [
        pl.callbacks.early_stopping.EarlyStopping(monitor='val_loss', patience=12, verbose=True),
    ]
    model = CondVAEModel(dataset=dataset, core_layer_widths=[64, 64], latent_dim=4, loss_weights={'x': 1., 'y': 1., 'kl': 0.1, 'decorrelation': .5})
    model.fit(train_loader=train_gen, val_loader=val_gen, max_epochs=200, callbacks=callbacks)
    model.save(f'../saved_models/{model_name}_{num_modes}modes/')
    dataset.save(f'../saved_models/{model_name}_{num_modes}modes/test_data/')  
    
def sample_circles(model: CondVAEModel, data: Dict, num_samples: int) -> Dict:
    pred_samples = {}
    for isample in range(num_samples):
        pred = {}
        with torch.no_grad():
            pred['x'] = model.decode(data=data)
            pred.update(model.encode(pred['x']))
        pred_samples[isample] = pred
    pred = {'x': {}, 'y': {}}
    for key in pred_samples[0]['x']:
        pred['x'][key] = np.concatenate([pred_samples[isample]['x'][key][:,:,None] for isample in range(num_samples)], axis=2)
    for key in pred_samples[0]['y']:
        pred['y'][key] = np.concatenate([pred_samples[isample]['y'][key][:,:,None] for isample in range(num_samples)], axis=2)
    return pred

def sample_designs(model: CondVAEModel, data_gen: DataLoader, n_gen: int) -> Tuple[Dict, Dict]:
    if isinstance(data_gen, DataLoader):
        data = rec_concat_dict([batch for batch in data_gen])
    else:
        data = data_gen
    pred = sample_circles(model, data, n_gen)
    return pred['y'], pred['x']

def propose_design(model: CondVAEModel, design_dict: Dict, n_samples: int=1000, n_best: int=10) -> pd.DataFrame:
    pred_perf_data, pred_design_data = sample_designs(model, design_dict, n_samples)
    num_requested_designs = list(design_dict.values())[0].shape[0]
    design_error_dict = {}

    for feature in design_dict.keys():
        design_error_dict[feature] = np.abs(pred_perf_data[feature] - design_dict[feature][:,:,None])[:,0]
    design_error_dict['total'] = np.sum([design_error_dict[feature] for feature in design_dict], axis=0)    
    
    columns = [f'{k}_requested' for k in design_dict.keys()] + [f'{k}_predicted' for k in design_dict.keys()] + [k for k in model.dataset.x.keys()] + ['design index']

    apply_log = ["cs_half_width", "cs_half_height", "spiral_turns", "cs_scale"]
    df_list = []
    for i in range(num_requested_designs):
        save_idx = np.argsort(design_error_dict['total'][i])[:n_best]
        df = pd.DataFrame(columns=columns)
        for k in design_dict.keys():
            df[f'{k}_predicted'] = 10 ** pred_perf_data[k][i,0,save_idx]
            df[f'{k}_requested'] = 10 ** design_dict[k][i,0] * np.ones(n_best)
        for k in model.dataset.x.keys():
            if k in apply_log:
                df[k] = 10 ** pred_design_data[k][i,0,save_idx]
            else:
                df[k] = pred_design_data[k][i,0,save_idx]
        df['design index'] = (i+1) * np.ones(n_best)
        df_list.append(df)
        
    suggested_designs = pd.concat(df_list, ignore_index=True)
    return suggested_designs

def get_design_error(model: CondVAEModel, data_gen: DataLoader, n_gen: int=1000, return_pred: bool=False, 
                     linear: bool=False, correction_factors: Dict={'freqs': 1., 'modal_mass': 1.}) -> Tuple[Dict, Dict]:
    if isinstance(data_gen, DataLoader):
        data = rec_concat_dict([batch for batch in data_gen])
    else:
        data = data_gen
    design_error = {}
    pred_perf_data, pred_design_data = sample_designs(model, data, n_gen)
    

    for feature in model.dataset.y.keys():
        if 'MMY' in feature:
            corr_fact = correction_factors['modal_mass']
        else:
            corr_fact = correction_factors['freqs']
        pred_perf_data[feature] += np.log10(corr_fact)
        
    for feature, pred_data in pred_perf_data.items():
        if 'MMY' in feature:
            corr_fact = correction_factors['modal_mass']
        else:
            corr_fact = correction_factors['freqs']
            
        data_tmp = data[feature] + np.log10(corr_fact)
        if linear:
            design_error[feature] = np.abs(10 ** data_tmp[:,:,None] - 10 ** pred_data)
        else:
            design_error[feature] = np.abs(data_tmp[:,:,None] - pred_data)
    if return_pred:
        return design_error, data, pred_perf_data, pred_design_data
    else:
        return design_error, data

def plot_design_errors(model: CondVAEModel, data_gen: DataLoader, linear: bool=False, n_bins: int=10, 
                       n_gen: int=1000, n_best: int=10, 
                       correction_factors: Dict={'modal_mass': 1,'freqs': 1}) -> Tuple[plt.Figure, List, List]:
    ax1_list, ax2_list = [], []
    design_error, data = get_design_error(model, data_gen, n_gen, linear=linear, correction_factors=correction_factors)
    fig = plt.figure(figsize=(20,8))
    for ifeature, feature in enumerate(model.dataset.y.keys()):
        if 'MMY' in feature:
            corr_fact = correction_factors['modal_mass']
        else:
            corr_fact = correction_factors['freqs']
            
        data_tmp = data[feature] + np.log10(corr_fact)
        if linear:
            bins = np.logspace(data_tmp.min(), data_tmp.max(), n_bins + 1, base=10)
        else:
            bins = np.linspace(data_tmp.min(), data_tmp.max(), n_bins + 1)
        delta_bins = bins[1] - bins[0]
        bin_err_mean, bin_err_std = np.empty(n_bins), np.empty(n_bins)
        best_bin_err_mean, best_bin_err_std = np.empty(n_bins), np.empty(n_bins)
        for ibin in range(n_bins):
            if linear:
                idx = np.where((10 ** data_tmp > bins[ibin]) & (10 ** data_tmp <= bins[ibin+1]))[0]
            else:
                idx = np.where((data_tmp > bins[ibin]) & (data_tmp <= bins[ibin+1]))[0]
            bin_err = design_error[feature][idx]
            bin_err_mean[ibin] = np.mean(np.median(bin_err, axis=-1))
            bin_err_std[ibin] = bin_err.std()
            bin_err_sorted = np.sort(design_error[feature][idx], axis=-1)
            best_bin_err_mean[ibin] = np.mean(np.median(bin_err_sorted[:,:,:n_best], axis=-1))
            best_bin_err_std[ibin] = bin_err_sorted[:,:,:n_best].std()

        
        num_design_vars = len(model.dataset.y.keys())
        ax1 = fig.add_subplot(2,num_design_vars, ifeature + 1)
        ax2 = fig.add_subplot(2,num_design_vars, ifeature + 1 + num_design_vars)
        if linear:
            ax2.hist(10 ** data_tmp, density=True, bins=bins, facecolor='k', alpha=0.75)
        else:
            ax2.hist(data[feature], density=True, bins=bins, facecolor='k')
        err1 = ax1.errorbar(bins[:-1] + .5*delta_bins, bin_err_mean, yerr=0*bin_err_std, 
                            marker='o', label=f'average', color='C0', alpha=0.75)
        err2 = ax1.errorbar(bins[:-1] + .5*delta_bins, best_bin_err_mean, yerr=0*best_bin_err_std, 
                            marker='o', label=f'top {n_best}', color='C3', alpha=0.75)
        if ifeature == 0:
            ax1.legend([err1, err2], [err1.get_label(), err2.get_label()])
        if linear:
            ax1.plot([10 ** np.amin(data_tmp), 10 ** np.amax(data_tmp)], [10 ** np.amin(data_tmp), 
                                                                          10 ** np.amax(data_tmp)], 'k--')
            ax1.plot([10 ** np.amin(data_tmp), 10 ** np.amax(data_tmp)], [10 ** np.amin(data_tmp-1), 
                                                                          10 ** np.amax(data_tmp-1)], 'k:')
        ax2.set_xlabel(feature)
        if ifeature == 0:
            ax1.set_ylabel('Abs. Design error')
            ax2.set_ylabel('Training data density', labelpad=10)
        if linear:
            ax1.set_xscale('log')
            ax2.set_xscale('log')
            ax1.set_yscale('log')
        ax2.set_yticks([])
        ax1_list.append(ax1)
        ax2_list.append(ax2)
        
    return fig, ax1_list, ax2_list