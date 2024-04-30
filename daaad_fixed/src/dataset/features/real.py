from typing import List
from dataset.data_transformers import DataTransformerSequence
import numpy as np
import torch.nn as nn

from utils import density_plot
from models.heads import InHeadFC, OutHeadFC
from dataset.features.data_object import DataObject, DataTransformer
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


    def __init__(self, name:str, scaling_type:str='standard', data_transformer:DataTransformer=None, **kwargs):
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

        self.scaling_type = scaling_type
        super().__init__(name, data_transformer, **kwargs)
        self.type = 'real'
        if data_transformer is None:
            self.transformer = DataTransformer.deserialize(scaling_type)

    def _init_transformer(self, data:np.array) -> DataTransformer:
        """
        Initializes the transformer in charge of transforming the data before it is passed to the machine learning model.

        Parameters
        ----------
        data : np.array
            The data to be transformed
            
        Returns
        -------
        DataTransformer
            The transformer used to pre-process the data.
        """
        self.transformer.fit(data.reshape(-1, 1))
        return self.transformer

    def set_data(self, data: np.array):
        """
        Sets the data attribute and initializes the transformer and shape.

        Parameters
        ----------
        data : np.array
            The data to be set as the attribute.
        """
        super().set_data(data)
        self.mean_value, self.std_value = self.data.mean(), self.data.std()
        self.max_value, self.min_value = self.data.max(), self.data.min()

    def append_data(self, data: np.array, update_transformer: bool = True):
        """
        Appends new data to the existing attribute and updates the transformer and shape if specified.
        
        Parameters
        ----------
        data : np.array
            The data to be appended to the existing attribute.
        update_transformer : bool
            Whether to update the transformer and shape after appending the data.
        """
        super().append_data(data, update_transformer)
        self.mean_value, self.std_value = self.data.mean(), self.data.std()
        self.max_value, self.min_value = self.data.max(), self.data.min()

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

    def get_heads(self, head_layer_widths:List[int], last_core_layer_width:int, activation:nn.Module, **kwargs) -> tuple:
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

        return InHeadFC(1, head_layer_widths, activation), OutHeadFC(last_core_layer_width, head_layer_widths[::-1] + [1], activation, out_activation=out_activation, **kwargs)

    def get_config(self):
        '''
        Returns the config necessary for building same instance. 
        Useful for storing / loading the dataset.
        '''
        return {
            'scaling_type': self.scaling_type, 
            'max_value': self.max_value,
            'min_value': self.min_value,
            'mean_value': self.mean_value,
            'std_value': self.std_value,
            **super().get_config()
        }