import sys
sys.path.append('../')
sys.path.append('../src/')

from daaad.src.models.cae import CondVAEModel
from daaad.src.utils import rec_concat_dict
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
import torch
from typing import Union, Dict, List, Tuple
import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt

def train_cvae(model_name: str, dataset: Dataset, train_gen: DataLoader, val_gen: DataLoader, 
               layer_widths: list=[64, 64], latent_dim: int=4) -> None:
    """Tr

    Args:
        num_modes: _description_. Defaults to 2.
        data_fraction: _description_. Defaults to 1.
        only_freqs: _description_. Defaults to False.
        model_name: _description_. Defaults to 'spiral'.
        batch_size: _description_. Defaults to 128.
        val_split: _description_. Defaults to .1.
    """
    callbacks = [
        pl.callbacks.early_stopping.EarlyStopping(monitor='val_loss', patience=12, verbose=True),
    ]
    model = CondVAEModel(dataset=dataset, layer_widths=layer_widths, latent_dim=latent_dim, 
                         loss_weights={'x': 1., 'y': 1., 'kl': 0.1, 'decorrelation': .5})
    model.fit(train_loader=train_gen, val_loader=val_gen, max_epochs=200, callbacks=callbacks)
    model.save(f'../saved_models/{model_name}/')
    return model
    
def sample_circles(model: CondVAEModel, data: Dict, num_designs: int) -> Dict:
    with torch.no_grad():
        pred = model.decode(data=data, return_lat=False, batch_size=num_designs)
        pred.update(model.encode(pred['x'], batch_size=num_designs))
    return pred

def sample_designs(model: CondVAEModel, data_gen: DataLoader, n_gen: int, correction_factors: dict) -> Tuple[Dict, Dict]:
    if isinstance(data_gen, DataLoader):
        data = rec_concat_dict([batch for batch in data_gen])
    else:
        data = data_gen
        
    pred_perf_data = {}
    for yfeature in model.dataset.y.keys():
        pred_perf_data[yfeature] = np.empty([data[yfeature].shape[0], model.dataset.y[yfeature].get_config()['shape'][0], n_gen])
        
    pred_design_data = {}
    for feature in model.dataset.x.keys():
        pred_design_data[feature] = np.empty([data[yfeature].shape[0], model.dataset.x[feature].get_config()['shape'][0], n_gen])
    keys = list(data.keys())
    num_designs = data[keys[0]].shape[0]
    for igen in range(n_gen):
        pred_attr = sample_circles(model, data, num_designs)
        for feature in model.dataset.y.keys():
            if 'MMY' in feature:
                corr_fact = correction_factors['modal_mass']
            else:
                corr_fact = correction_factors['freqs'] 
            pred_perf_data[feature][:,:,igen] = pred_attr['y'][feature] + np.log10(corr_fact)
    
        for feature in model.dataset.x.keys():
            pred_design_data[feature][:,:,igen] = pred_attr['x'][feature]

    return pred_perf_data, pred_design_data

def propose_design(model: CondVAEModel, design_dict: Dict, n_samples: int=1000, n_best: int=10, apply_log: list=[], 
                   correction_factors: Dict={'freqs': 1., 'modal_mass': 1.}) -> pd.DataFrame:
    pred_perf_data, pred_design_data = sample_designs(model, design_dict, n_samples, correction_factors)
    num_requested_designs = list(design_dict.values())[0].shape[0]
    design_error_dict = {}

    for feature in design_dict.keys():
        if 'MMY' in feature:
            corr_fact = correction_factors['modal_mass']
        else:
            corr_fact = correction_factors['freqs'] 
        design_error_dict[feature] = np.abs(pred_perf_data[feature] - design_dict[feature][:,:,None] - np.log10(corr_fact))[:,0]
    design_error_dict['total'] = np.sum([design_error_dict[feature] for feature in design_dict], axis=0)    
    
    columns = [f'{k}_requested' for k in design_dict.keys()] + [f'{k}_predicted' for k in design_dict.keys()] + [k for k in model.dataset.x.keys()] + ['design index']
    df_list = []
    for i in range(num_requested_designs):
        save_idx = np.argsort(design_error_dict['total'][i])[:n_best]
        df = pd.DataFrame(columns=columns)
        for k in design_dict.keys():
            if 'MMY' in k:
                corr_fact = correction_factors['modal_mass']
            else:
                corr_fact = correction_factors['freqs'] 
            df[f'{k}_predicted'] = 10 ** pred_perf_data[k][i,0,save_idx] 
            df[f'{k}_requested'] = 10 ** design_dict[k][i,0] * np.ones(n_best) * corr_fact
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
    pred_perf_data, pred_design_data = sample_designs(model, data, n_gen, correction_factors=correction_factors)
        
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
    print(design_error[feature].shape)
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
        if linear:
            ax2.set_xlabel(feature[4:])
        else:
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


def plot_proposed_designs(model: CondVAEModel, train_gen: DataLoader, suggested_designs: pd.DataFrame, apply_log: list=[]):
    for iname, name1 in enumerate(model.dataset.x.keys()):
        if name1 in apply_log:
            data1 = np.log10(suggested_designs[name1])
        else:
            data1 = suggested_designs[name1]
        for jname, name2 in enumerate(model.dataset.x.keys()):
            if iname < jname:
                fig = plt.figure()
                if name2 in apply_log:
                    data2 = np.log10(suggested_designs[name2])
                else:
                    data2 = suggested_designs[name2]
                ax = fig.add_subplot(1,1,1)
                
                sn.kdeplot(x=train_gen.dataset.dataset.x[name1].data[:,0], y=train_gen.dataset.dataset.x[name2].data[:,0], ax=ax,color='k', alpha=0.5)
                designs = np.array(np.unique(suggested_designs['design index']), dtype=int)
                for i in designs:
                    ax.scatter(data1[suggested_designs['design index']==i], data2[suggested_designs['design index']==i], 
                               c=f'C{i}', zorder=5, label=f'Design {i}')
                #for i in range(10):
                #    ax.scatter(y_list[i][name1], y_list[i][name2],c='C%d' %i, zorder=9)

                ax.set_ylabel(name2)
                ax.set_xlabel(name1)
                ax.legend()
                plt.tight_layout()
                plt.show()
                
def plot_requested_predicted_designs(suggested_designs: pd.DataFrame, train_gen: DataLoader, num_modes: int, correction_factors: Dict={'modal_mass': 1,'freqs': 1}):
    plt.figure(figsize=(5 * num_modes,5))

    designs = np.array(np.unique(suggested_designs['design index']), dtype=int)

    for imodes in range(num_modes):
        ax = plt.subplot(1,num_modes,imodes+1)
        sn.kdeplot(x=train_gen.dataset.dataset.y[f'log_M{imodes+1}_F'].data[:,0] + np.log10(correction_factors['freqs']),
                y=train_gen.dataset.dataset.y[f'log_M{imodes+1}_MMY'].data[:,0] + np.log10(correction_factors['modal_mass']),ax=ax,color='k', alpha=0.5)
        data1_request, data2_request = np.log10(suggested_designs[f'log_M{imodes+1}_F_requested']), np.log10(suggested_designs[f'log_M{imodes+1}_MMY_requested'])
        data1_pred, data2_pred = np.log10(suggested_designs[f'log_M{imodes+1}_F_predicted']), np.log10(suggested_designs[f'log_M{imodes+1}_MMY_predicted'])
        for idesign in designs:
            ax.scatter(data1_pred[suggested_designs['design index']==idesign], data2_pred[suggested_designs['design index']==idesign], 
                    marker='v', color=f'C{idesign}', s=100)
            ax.scatter(data1_request[suggested_designs['design index']==idesign], data2_request[suggested_designs['design index']==idesign], 
                    marker='o', color=f'C{idesign}', s=100, edgecolors='k')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Modal mass [kg]')
        ax.set_title(f'Mode {imodes+1}')
        xticks = ax.get_xticks()
        xticks = xticks[xticks % 1 == 0]
        xticklabels = ['$10^{%d}$' %x for x in xticks] 
        ax.set_xticks(xticks, xticklabels)
        yticks = ax.get_yticks()
        yticks = yticks[yticks % 1 == 0]
        yticklabels = ['$10^{%d}$' %y for y in yticks] 
        ax.set_yticks(yticks, yticklabels)
        plt.tight_layout()
    