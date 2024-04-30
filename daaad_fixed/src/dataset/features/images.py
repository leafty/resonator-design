import numpy as np
import torch.nn as nn

from typing import Dict, List
from utils import imscatter_many, imshow_many
from dataset.features.matrix import DataMatrix
from models.heads import InHeadConv2D, OutHeadConv2D
from dataset.data_transformers import DataTransformer


class DataImage(DataMatrix):
    '''
    A class representing images.
    '''

    @staticmethod
    def compare(images:Dict[str, np.array], title:str, axis=None, path:str=None, evaluation:bool=False, max_samples:int=3):
        '''
        Compare several distributions against each other by plotting images from each distribution in an own column.
        Adds a columns with the squared difference between two distributions if in evaluation mode.

        Parameters
        ----------
        distributions : Dict[np.array]
            Contains several distributions, where the keys are the distribution names 
            and values np.array containing samples from the distributions
            e.g. {"Actual values": np.array, "Predicted values": np.array}
        title : str
            Title of the resulting plot(s)
        axis : plt.axis
            The axis where the plots should be places. If None, a new axis is created
        path : str
            Where the plots should be saved. If None, plots are not saved
        evaluation : bool
            Whether there are 2 distributions that should be evaluated against each other
        '''
        keys = list(images.keys())
        assert len(keys) == 2

        if evaluation:
            images['Squared Difference'] = (images[keys[0]] - images[keys[1]])**2
            keys = list(images.keys())
        
        imgs = []
        for i in range(min(len(images[keys[0]]), max_samples) * len(keys)):
            imgs.append(images[keys[i % len(keys)]][i // len(keys)])
        imshow_many(imgs, title=title, subtitles=keys, row_width=len(keys), colorbars=True, axis=axis, path=path)


    @staticmethod
    def augment(data:np.array, std:float=0.01) -> np.array:
        '''
        Adds random noise to data for augmentation.

        Parameters
        ----------
        data : np.array
            The data to be augmented
        std : float
            The standard deviation of the noise
        '''

        return data + np.random.normal(size=data.shape, scale=std)


    def __init__(self, name:str, scaling_type:str='norm_m1to1', data_transformer:DataTransformer=None, **kwargs):
        '''
        Parameters
        ----------
        data : np.array
            The data associated with this feature
        name : str
            The name of this feature
        scaling_type : str
            How the data should be transformed, one of "standard" for standardisation, 
            "norm_0to1" for scaling between 0 and 1 or "norm_m1to1" for scaling between -1 and 1
        '''
        super().__init__(name, scaling_type, data_transformer, **kwargs)
        self.type = 'image'


    def inspect(self, num_images:int=3, axis=None, path:str=None, **kwargs):
        '''
        Visualises the data associated with this feature in a density plot for inspection.

        Parameters
        ----------
        axis : plt.axis
            The axis where the plot should be places. If None, a new axis is created.
        path : str
            Where the plot should be saved. If None, plot is not saved.
        '''
        if self.data is None:
            raise Exception('Data is None, call "set_data" in ' + self.name + ' before calling "inspect".')

        ixs = np.random.randint(0, len(self.data), (num_images,))
        imshow_many(images=self.data[ixs], title=self.name, row_width=num_images, axis=axis, path=path, **kwargs)


    def inspect_latent(self, latent_dim_0:np.array, latent_dim_1:np.array, images:np.array, **kwargs):
        '''
        Creates scatter plot of latent space and assigns colors to points according to "values"

        Parameters
        ----------
        latent_dim_0, latent_dim_1 : np.array
            Coordinates of each point in the latent space
        values : np.array
            Value of each point in the latent space (used for coloring)
        '''
        imscatter_many('Latent dimension 0', latent_dim_0, 'Latent dimension 1', latent_dim_1, images, 
                        title=kwargs.get('title', None), zoom=(latent_dim_0**2 + latent_dim_1**2)/2, axis=kwargs.get('axis', None), path=kwargs.get('path', None))


    def inspect_discrete_latent(self, class_ixs:np.array, images:np.array, **kwargs):
        '''
        Creates scatter plot of latent space and assigns colors to points according to "values"

        Parameters
        ----------
        class_ixs : np.array
            Class to which the samples have been attributed in the latent space.
        values : np.array
            Value of each point in the latent space (used for coloring)
        '''
        sorted_ixs = np.argsort(class_ixs.flatten())
        imshow_many(
            images=images[sorted_ixs], 
            subtitles=class_ixs[sorted_ixs].astype(str), 
            title=kwargs.get('title', self.name + ' vs discrete latent variable'),
            kwargs=kwargs
        )


    def get_heads(self, head_layer_widths:List[int], last_core_layer_width:int, activation:nn.Module) -> tuple:
        '''
        Returns a fully-connected head for encoding and decoding this feature

        Parameters
        ----------
        head_layer_widths : List[int]
            List of integers where the length denotes number of layers and values the width of the layers in the head
        last_core_layer_width : int
            Dimensionality of autoencoder output to be decoded by this feature head
        activation : nn.Module
            Activation function to be used in this head
        '''
        if self.scaling_type in ['minmax', 'norm_0to1']:
            out_activation = 'sigmoid'
        elif self.scaling_type in ['norm_m1to1']:
            out_activation = 'tanh'
        else:
            out_activation = None

        in_head = InHeadConv2D(
            self.shape, 
            head_layer_widths, 
            activation, 
            attn_block_indices=[len(head_layer_widths) // 2] if len(head_layer_widths) < 0 else [],
        )

        out_head = OutHeadConv2D(
            last_core_layer_width, 
            self.shape, 
            head_layer_widths[::-1], 
            activation, 
            out_activation=out_activation, 
            attn_block_indices=[len(head_layer_widths) // 2] if len(head_layer_widths) < 0 else [],
        )

        return in_head, out_head