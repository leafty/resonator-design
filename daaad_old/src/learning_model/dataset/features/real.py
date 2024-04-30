from typing import List
import numpy as np
import torch.nn as nn

from src.utils import density_plot
from src.learning_model.models.heads import InHeadFC, OutHeadFC
from src.learning_model.dataset.features.data_object import DataObject, DataTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataReal(DataObject):
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
        in_head = InHeadFC(1, head_layer_widths, activation)
        return in_head, OutHeadFC(last_core_layer_width, in_head)


    def __init__(self, name:str, scaling_type:str='standard', **kwargs):
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

        self.type = 'real'
        self.scaling_type = scaling_type
        super(DataReal, self).__init__(name, **kwargs)

    def _init_transformer(self, data:np.array) -> DataTransformer:
        if self.scaling_type in ['standard', 'standardize', 'norm_standard']:
            self.transformer = StandardScaler()
        elif self.scaling_type in ['minmax', 'normalize', 'norm_0to1']:
            self.transformer = MinMaxScaler(feature_range=(0, 1))
        elif self.scaling_type in ['norm_m1to1']:
            self.transformer = MinMaxScaler(feature_range=(-1, 1))
        else:
            raise ValueError(f'Scaling type {self.scaling_type} for feature ' + str(self.name) + \
                ' not in list of valid augmentations: "standard", "minmax", "norm_0to1" or "norm_m1to1".')
        self.transformer.fit(data)
        return self.transformer

    def set_data(self, data: np.array):
        super().set_data(data)
        self._init_transformer(self.data)
        self.mean_value, self.std_value  = self.data.mean(), self.data.std()

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

        density_plot({self.name: self.data}, self.name, axis, path, **kwargs)

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