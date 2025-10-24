import numpy as np
import torch.nn as nn
from typing import Dict, List, Union

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dataset.features.real import DataReal
from models.heads import InHeadConv2D, OutHeadConv2D
from dataset.features.data_object import DataTransformer

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
    def validate_data(data:np.array) -> np.array:
        '''
        Brings the data into a valid format or raises an error, if it is not recoverable

        Parameters
        ----------
        data : np.array
            Data to be validated
        '''
        if len(data.shape) == 3:
            return data[:, :, :, np.newaxis]
        elif len(data.shape) == 4:
            return data
        else:
            raise Exception('Data should have four dimensions, but has ' + str(len(data.shape)))


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
        self.type = 'matrix'

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
        super()._init_transformer(data.reshape(-1, 1))
        return self.transformer

    def transform(self, data:Union[Dict[str, np.array], np.array]) -> np.array:
        """
        Transforms the data using the transformer.

        Parameters
        ----------
        data : Union[Dict[str, np.array], np.array]
            The data to be transformed. Can be a dictionary with feature names as keys and data as values, or just the data array.

        Returns
        -------
        np.array
            The transformed data.
        """
        if isinstance(data, dict):
            data = data[self.name]

        shape = data.shape
        transf_data = self.transformer.transform(data.reshape(-1, 1))
        return transf_data.reshape(shape)

    def inverse_transform(self, data:Union[Dict[str, np.array], np.array]) -> np.array:
        """
        Inverse transforms the data using the transformer.

        Parameters
        ----------
        data : Union[Dict[str, np.array], np.array]
            The data to be inverse transformed. Can be a dictionary with feature names as keys and data as values, or just the data array.

        Returns
        -------
        np.array
            The inverse transformed data.
        """
        if isinstance(data, dict):
            data = data[self.name]
            
        shape = data.shape
        itranf_data = self.transformer.inverse_transform(data.reshape(-1, 1)).astype(self.data_type)
        return itranf_data.reshape(shape)

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
        raise NotImplementedError()

    def get_heads(self, head_layer_widths:List[int], last_core_layer_width:int, activation:nn.Module, **kwargs) -> tuple:
        '''
        Returns a custom head for encoding and decoding this feature

        Parameters
        ----------
        head_layer_widths : List[int]
            List of integers where the length denotes number of layers and values the width of the layers in the head
        last_core_layer_width : int
            Dimensionality of autoencoder output to be decoded by this feature head
        activation : nn.Module
            Activation function to be used in this head
        '''
        raise NotImplementedError()