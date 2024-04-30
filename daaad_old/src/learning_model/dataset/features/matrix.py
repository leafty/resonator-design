import numpy as np
import torch.nn as nn
from typing import List

from src.learning_model.dataset.features.real import DataReal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.learning_model.models.heads import InHeadConv2D, OutHeadConv2D
from src.learning_model.dataset.features.data_object import DataTransformer

class DataMatrix(DataReal):
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
        # TODO: Add support for matrices with more than 2 dimensions.
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
        super().__init__(name, scaling_type, **kwargs)
        self.type = 'matrix'

    def _init_transformer(self, data:np.array) -> DataTransformer:
        if self.scaling_type in ['standard', 'standardize', 'norm_standard']:
            self.transformer = StandardScaler()
            self.transformer.mean_ = data.mean()
            self.transformer.scale_ = data.std()**2
        elif self.scaling_type in ['minmax', 'normalize', 'norm_0to1']:
            self.transformer = MinMaxScaler(feature_range=(0, 1))
            self.transformer.scale_ = 1 / (data.max() - data.min())
            self.transformer.min_ = -self.transformer.scale_ * data.min()
        elif self.scaling_type in ['norm_m1to1']:
            self.transformer = MinMaxScaler(feature_range=(-1, 1))
            self.transformer.scale_ = 2 / (data.max() - data.min())
            self.transformer.min_ =  -1 - self.transformer.scale_ * data.min()
        return self.transformer

    def inspect(self, axis=None, path:str=None, **kwargs):
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

        # TODO
        raise NotImplementedError()

    @staticmethod
    def validate_data(data:np.array) -> np.array:
        if len(data.shape) == 2:
            return data[:, :, np.newaxis]
        elif len(data.shape) == 3:
            return data
        else:
            raise Exception('Data should have three dimensions, but has ' + str(len(data.shape)))