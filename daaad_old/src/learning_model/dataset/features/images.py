import numpy as np
import torch.nn as nn

from typing import List
from src.utils import imshow_many
from src.learning_model.models.heads import InHeadConv2D, OutHeadConv2D
from src.learning_model.dataset.features.matrix import DataMatrix


class DataImage(DataMatrix):
    '''
    A class representing continuous or real data.
    '''

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


    @staticmethod
    def get_heads(head_layer_widths:List[int], last_core_layer_width:int, activation:nn.Module) -> tuple:
        return InHeadConv2D(1, head_layer_widths, activation), OutHeadConv2D(last_core_layer_width, head_layer_widths[::-1] + [1], activation)


    def __init__(self, name:str, scaling_type:str='minmax', **kwargs):
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
        self.type = 'image'
        super().__init__(name, scaling_type, **kwargs)

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
        imshow_many(self.data[ixs], self.name, num_images, axis, path, **kwargs)

    def get_config(self):
        '''
        Returns the config necessary for building same instance. 
        Useful for storing / loading the dataset.
        '''
        return {
            'scaling_type': self.scaling_type, 
            'mean_value': self.mean_value,
            'std_value': self.std_value,
            **super().get_config()
        }
