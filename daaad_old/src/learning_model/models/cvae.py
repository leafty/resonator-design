import os
import pickle
import torch
import numpy as np

from typing import Union
from typing import Dict, List, Union
from torch.utils.data import DataLoader
from pytorch_lightning.trainer.supporters import CombinedLoader

from src.learning_model.models.encoders import VEncoder
from src.learning_model.dataset.data_set import DataSet
from src.learning_model.models.cae import CondAEModel, MultiModalCondAEModel
from src.utils import numpy_dict_to_tensor, rec_concat_dict, sum_join_dicts, swarm_plot, torch_dict_to_numpy

class CondVAEModel(CondAEModel):
    def __init__(self, dataset:DataSet, core_layer_widths:List[int], latent_dim:int, x_heads_layer_widths:dict={}, y_heads_layer_widths:dict={}, 
                        loss_weights:dict=None, activation:str='leaky_relu', optimizer:torch.optim.Optimizer=None, pass_y_to_encoder:bool=False, guided:bool=False, name:str='CondAEModel', **kwargs):
        super().__init__(dataset, core_layer_widths, latent_dim, x_heads_layer_widths, y_heads_layer_widths, loss_weights, activation, optimizer, pass_y_to_encoder, guided, name, **kwargs)

        self.encoder = VEncoder({x_key: x_heads[0] for x_key, x_heads in self.x_heads.items()}, 
                               {y_key: y_heads[1] for y_key, y_heads in self.y_heads.items()}, 
                               core_layer_widths, latent_dim, activation)

    def _step(self, batch:dict, batch_idx:int, mode:str):
        pred, losses = super()._step(batch, batch_idx, mode)

        z_mean, z_log_var = pred['z_mean'], pred['z_log_var']
        kl_loss = ((-0.5 * (1 + z_log_var - z_mean**2 - torch.exp(z_log_var))).sum(dim=1)).mean()
        
        if mode in ['train', 'training']:
            losses['train_loss'] += self.loss_weights.get('kl', 1.) * kl_loss
        
        losses[mode + '_kl_loss'] = kl_loss
        return pred, losses

    def decode(self, data:Union[DataLoader, dict], z:np.array=None, transform:bool=True, cond_weights:List[float]=None):
        if isinstance(data, DataLoader):
            data = rec_concat_dict([batch for batch in data])

        if z is None:
            z = data.get('z', torch.normal(mean=0., std=1., size=(len(list(data.values())[0]), self.latent_dim)))
        elif isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()

        return super().decode(data, z, transform, cond_weights)


class MultiModalCondVAEModel(MultiModalCondAEModel):
    def __init__(self, datasets:Dict[str, DataSet], core_layer_widths:List[int], latent_dim:int, name:str='MultiModalCondVAEModel', *args, **kwargs):
        super().__init__(datasets, core_layer_widths, latent_dim, name, *args, **kwargs)

        self.encoder = VEncoder({x_key: x_heads[0] for x_key, x_heads in self.x_heads.items()}, 
                               {y_key: y_heads[1] for y_key, y_heads in self.y_heads.items()}, 
                               self.core_layer_widths, self.latent_dim, self.activation)
        
        self.models = {name: CondVAEModel(ds, self.core_layer_widths, self.latent_dim, name=name + '_model') for name, ds in datasets.items()}
        self._share_weights_across_models()

    def decode(self, data:Union[CombinedLoader, Dict[str, np.array]], z:Dict[str, np.array]=None, transform:bool=True) -> Dict[str, Dict[str, np.array]]:
        if z is None:
            z = data.get('z', torch.normal(mean=0., std=1., size=(len(list(data.values())[0]), self.latent_dim)))
        elif isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()
        return super().decode(data, z, transform)

    def decode_multimodal(self, data:Union[CombinedLoader, Dict[str, np.array]], mode:str, z:Dict[str, np.array]=None, transform:bool=True) -> Dict[str, Dict[str, np.array]]:
        if z is None:
            z = data.get('z', torch.normal(mean=0., std=1., size=(len(list(data.values())[0]), self.latent_dim)))
        elif isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()
        return super().decode_multimodal(data, mode, z, transform)

    @classmethod
    def load_model(cls, path:str):
        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        modes = [mode_path for mode_path in os.listdir(path) if os.path.isdir(os.path.join(path, mode_path))]
        models = {mode: CondVAEModel.load_model(os.path.join(path, mode)) for mode in modes}
        obj = MultiModalCondVAEModel({mode: model.dataset for mode, model in models.items()}, **config)
        obj.models = models
        return obj