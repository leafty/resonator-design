import torch
import numpy as np

from typing import Union
from typing import Dict, List, Union
from torch.utils.data import DataLoader

from dataset.data_set import DataSet
from models.cae import CondAEModel, CondVAEModel
from models.encoders import DiscreteEncoder, Encoder, VEncoder
from utils import one_hot, rec_concat_dict, torch_dict_to_numpy

EPS = 1e-8

class DiscreteCondAEModel(CondAEModel):
    def __init__(self, dataset:DataSet, layer_widths:List[int], cont_latent_dim:int, disc_latent_dims:List[int], x_heads_layer_widths:Dict[str, List[int]]={}, y_heads_layer_widths:Dict[str, List[int]]={}, 
                        loss_weights:Dict[str, float]=None, activation:Union[torch.nn.Module, str]='leaky_relu', optimizer:torch.optim.Optimizer=None, pass_y_to_encoder:bool=False, 
                        name:str='DiscreteCondAEModel', **kwargs):
        super().__init__(dataset, layer_widths, cont_latent_dim + sum(disc_latent_dims), x_heads_layer_widths, y_heads_layer_widths, loss_weights, activation, optimizer, pass_y_to_encoder, name, **kwargs)

        self.encoder = DiscreteEncoder( 
            Encoder({x_key: x_heads[0] for x_key, x_heads in (self.x_heads.items() if not pass_y_to_encoder else (self.x_heads | self.y_heads).items())}, 
                    {y_key: y_heads[1] for y_key, y_heads in self.y_heads.items() if not pass_y_to_encoder} if not pass_y_to_encoder else {}, 
                    layer_widths, cont_latent_dim, activation), 
            disc_latent_dims
        )
        self.cont_latent_dim = cont_latent_dim
        self.disc_latent_dims = disc_latent_dims

    def decode(self, data:Union[DataLoader, Dict[str, np.array]], z_cont:np.array=None, alphas:np.array=None, transform:bool=True, batch_size:int=1, return_lat:bool=False):
        if z_cont is None:
            if isinstance(data, dict) and 'z_cont' in data:
                z_cont = data['z_cont']
            else:
                raise ValueError('The latent embedding `z_cont` must either be passed to the function or be included in `data`.')

        if alphas is None:
            if isinstance(data, dict) and 'alphas' in data:
                alphas = data['alphas']
            else:
                raise ValueError('The discrete latent embedding `alphas` must either be passed to the function or be included in `data`.')

        alphas = [one_hot(alphas[:, i], num_classes=self.disc_latent_dims[i]) for i in range(len(self.disc_latent_dims))]
        z = np.concatenate([z_cont] + alphas, axis=-1)
        return super().decode(data, z, transform, batch_size, return_lat)

    def inspect_discrete_latent(self, data, path:str=None):
        enc = self.encode(data)
        alphas = [np.argmax(alpha_value, axis=-1) for alpha_name, alpha_value in enc.items() if 'alpha_' in alpha_name]

        if isinstance(data, DataLoader):
            data = torch_dict_to_numpy(rec_concat_dict([batch for batch in data]))

        for key, feature in (self.dataset.x | self.dataset.y).items():
            for i, alpha in enumerate(alphas):
                feature.inspect_discrete_latent(
                    alpha, data[key].astype(feature.data_type),
                    title=feature.name + ' vs discrete latent var ' + str(i),
                    path=path,
                )


class DiscreteCondVAEModel(CondVAEModel, DiscreteCondAEModel):
    def __init__(self, dataset:DataSet, layer_widths:List[int], cont_latent_dim:int, disc_latent_dims:List[int], x_heads_layer_widths:Dict[str, List[int]]={}, y_heads_layer_widths:Dict[str, List[int]]={}, 
                        loss_weights:Dict[str, float]=None, activation:Union[torch.nn.Module, str]='leaky_relu', optimizer:torch.optim.Optimizer=None, pass_y_to_encoder:bool=False, 
                        name:str='DiscreteCondVAEModel', **kwargs):
        super().__init__(dataset, layer_widths, cont_latent_dim + sum(disc_latent_dims), x_heads_layer_widths, y_heads_layer_widths, loss_weights, activation, optimizer, pass_y_to_encoder, name, **kwargs)

        self.encoder = DiscreteEncoder(
            VEncoder({x_key: x_heads[0] for x_key, x_heads in (self.x_heads.items() if not pass_y_to_encoder else (self.x_heads | self.y_heads).items())}, 
                    {y_key: y_heads[1] for y_key, y_heads in self.y_heads.items() if not pass_y_to_encoder} if not pass_y_to_encoder else {}, 
                    layer_widths, cont_latent_dim, activation), 
            disc_latent_dims
        )
        self.cont_latent_dim = cont_latent_dim
        self.disc_latent_dims = disc_latent_dims

    def _step(self, batch:Dict[str, np.array], batch_idx:int, mode:str):
        pred, losses = super()._step(batch, batch_idx, mode)

        # get the weight or, if not specified, set it to be 0.1 times the kl loss weight
        disc_kl_loss_weight = self.loss_weights.get('disc_kl', self.loss_weights.get('kl', 0.1) * .1)
        disc_kl_loss = disc_kl_loss_weight * self._kl_multiple_discrete_loss([pred[k] for k in pred.keys() if 'alpha_' in k])
        
        if mode == 'train':
            losses['train_loss'] += disc_kl_loss
        
        losses[mode + '_disc_kl_loss'] = disc_kl_loss
        return pred, losses

    def _kl_multiple_discrete_loss(self, alphas):
        """
        Code by Emilien Dupont from https://github.com/Schlumberger/joint-vae/blob/master/jointvae/training.py

        Calculates the KL divergence between a set of categorical distributions
        and a set of uniform categorical distributions.
        Parameters
        ----------
        alphas : list
            List of the alpha parameters of a categorical (or gumbel-softmax)
            distribution. For example, if the categorical atent distribution of
            the model has dimensions [2, 5, 10] then alphas will contain 3
            torch.Tensor instances with the parameters for each of
            the distributions. Each of these will have shape (N, D).
        """
        # Calculate kl losses for each discrete latent
        kl_losses = [self._kl_discrete_loss(alpha) for alpha in alphas]

        # Total loss is sum of kl loss for each discrete latent
        kl_loss = torch.stack(kl_losses, dim=0).sum()

        return kl_loss

    def _kl_discrete_loss(self, alpha):
        """
        Code by Emilien Dupont from https://github.com/Schlumberger/joint-vae/blob/master/jointvae/training.py

        Calculates the KL divergence between a categorical distribution and a
        uniform categorical distribution.
        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the categorical or gumbel-softmax distribution.
            Shape (N, D)
        """
        disc_dim = int(alpha.size()[-1])
        log_dim = np.log(disc_dim)

        # Calculate negative entropy of each row
        neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
        # Take mean of negative entropy across batch
        mean_neg_entropy = torch.mean(neg_entropy, dim=0)
        # KL loss of alpha with uniform categorical variable
        kl_loss = log_dim + mean_neg_entropy
        return kl_loss

    def decode(self, data:Union[DataLoader, Dict[str, np.array]], z_cont:np.array=None, alphas:np.array=None, transform:bool=True, batch_size:int=1, return_lat:bool=False):
        if z_cont is None:
            if isinstance(data, DataLoader):
                num_samples = len(data.dataset)
            else:
                num_samples = len(list(data.values())[0])
            z_cont = np.random.normal(size=(num_samples, self.cont_latent_dim))

        if alphas is None:
            # sample random class indices
            alphas = [np.random.randint(self.disc_latent_dims[i], size=(len(z_cont), 1)) for i in range(len(self.disc_latent_dims))]
        else:
            alphas = [alphas[:, i] for i in range(len(alphas[0]))]

        alphas = [one_hot(alphas[i], num_classes=self.disc_latent_dims[i]) for i in range(len(self.disc_latent_dims))]
        z = np.concatenate([z_cont] + alphas, axis=-1)
        return super().decode(data, z, transform, batch_size, return_lat)