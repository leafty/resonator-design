import numpy as np
import torch.nn as nn

from typing import Dict, List, Union

from models.heads import InHeadFC, OutHeadFC
from dataset.features.data_object import DataObject, DataTransformer
from utils import bar_plot, confusion_matrix_plot

class DataCategorical(DataObject):
    '''
    A class representing categorical data.
    '''

    @staticmethod
    def compare(data:dict, title:str, axis=None, path:str=None, evaluation:bool=False):
        '''
        Compares different sets of categorical data by producing bar plots and confusion matrices.

        Parameters
        ----------
        data : dict
            A dictionary, where each key describes a distribution (e.g. ground truth, predictions, etc.) and the value contains the data
        title : str
            The title that should appear on each plot
        axis : plt.axis
            The axis where the plots should be places. If None, a new axis is created.
        path : str
            Where the plots should be saved. If None, plots are not saved.
        evaluation : bool
            Whether the comparison is an evaluation or not. If yes, a confusion matrix is plotted.
        '''

        for dv in data.values():
            if len(list(data.values())[0].shape) != 2 or list(data.values())[0].shape != dv.shape:
                raise Exception(f'Expected data to have shape (n_samples, 1), but was {dv.shape}' + \
                                'Calling inverse_transform first might solve the problem.')
        bar_plot(data, title, axis, path)
        if evaluation:
            name_a, data_a = list(data.items())[0]
            name_b, data_b = list(data.items())[1]
            confusion_matrix_plot(name_a, data_a, name_b, data_b, title, axis, path)

    @staticmethod
    def inspect_latent(latent_dim_0:np.array, latent_dim_1:np.array, values:np.array, **kwargs):
        '''
        Creates scatter plot of latent space and assigns different color to each unique class.

        Parameters
        ----------
        latent_dim_0, latent_dim_1 : np.array
            Coordinates of each point in the latent space
        values : np.array
            Value of each point in the latent space (used for coloring)
        '''
        super(DataCategorical, DataCategorical).inspect_latent(latent_dim_0, latent_dim_1, values.flatten(), **kwargs)


    def inspect_discrete_latent(self, class_ixs:np.array, values:np.array, **kwargs):
        '''
        Creates scatter plot of latent space and assigns colors to points according to "values"

        Parameters
        ----------
        class_ixs : np.array
            Class to which the samples have been attributed in the latent space.
        values : np.array
            Value of each point in the latent space (used for coloring)
        '''
        confusion_matrix_plot(self.name, values, 'Latent class', class_ixs, hide_zero_rows=True, hide_zero_columns=True, title=kwargs.get('title', self.name + ' vs discrete latent variable'))


    def __init__(self, name: str, classes: Union[list, np.array] = None, data_transformer:DataTransformer=None, **kwargs):
        '''
        Parameters
        ----------
        name : str
            The name of this feature
        classes : Union[list, np.array]
            Optional, a list or numpy array of distinct classes. 
            If not provided, will be infered from data.
        '''

        super().__init__(name, **kwargs)
        self.type = 'categorical'
        self.classes = classes

        if data_transformer is None:
            self.transformer = DataTransformer.deserialize('onehot')


    def _init_transformer(self, data:np.array = None) -> DataTransformer:
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
        if self.data is None:
            self.transformer.fit(np.array([self.classes]).T)
        else:
            self.transformer.fit(self.data)
        return self.transformer

    def set_data(self, data: np.array):
        """
        Sets the data attribute and initializes the transformer and shape.

        Parameters
        ----------
        data : np.array
            The data to be set as the attribute.
        """
        if self.classes is not None:
            self.classes = np.unique(np.concatenate([self.classes, np.unique(data)]))
        else:
            self.classes = np.unique(data)
        super().set_data(data)

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
        if self.classes is not None:
            self.classes = np.unique(np.concatenate([self.classes, np.unique(data)]))
        else:
            self.classes = np.unique(data)
        return super().append_data(data, update_transformer)

    def transform(self, data: Union[Dict[str, np.array], np.array]) -> np.array:
        '''
        Performs binary encoding if only two unique classes exist, otherwise one-hot encoding.

        Parameters
        ----------
        data : Union[Dict[str, np.array], np.array]
            The data to be transformed
        '''
        if isinstance(data, dict):
            data = data[self.name]
            
        if data.dtype == np.floating:
            res = super().transform(np.around(data).astype(int))
        else:
            res = super().transform(data)
            
        if isinstance(res, np.ndarray):
            return res
        else:
            return res.toarray()

    def inspect(self, axis=None, path:str=None, **kwargs):
        '''
        Visualises the data associated with this feature in a bar plot for inspection.

        Parameters
        ----------
        axis : plt.axis
            The axis where the plot should be places. If None, a new axis is created.
        path : str
            Where the plot should be saved. If None, plot is not saved.
        '''
        if self.data is None:
            raise Exception('Data is None, call "set_data" in ' + self.name + ' before calling "inspect".')

        bar_plot({self.name: self.data}, self.name, axis, path, **kwargs)
    
    def get_objective(self):
        '''
        Returns the loss function for approximating this feature.
        '''

        if len(self.classes) <= 2:
            return nn.BCEWithLogitsLoss()
        else:
            return nn.CrossEntropyLoss()

    def get_heads(self, head_layer_widths:List[int], last_core_layer_width:int, activation:nn.Module, **kwargs) -> tuple:
        '''
        Returns a fully-connected head with the appropriate number of in / out channels for encoding / decoding this feature

        Parameters
        ----------
        head_layer_widths : List[int]
            List of integers where the length denotes number of layers and values the width of the layers in the head
        last_core_layer_width : int
            Dimensionality of autoencoder output to be decoded by this feature head
        activation : nn.Module
            Activation function to be used in this head
        '''
        return InHeadFC(self.shape[-1], head_layer_widths, activation), \
               OutHeadFC(last_core_layer_width, head_layer_widths[::-1] + [self.shape[-1]], activation)

    def get_config(self):
        '''
        Returns the config necessary for building same instance. 
        Useful for storing / loading the dataset.
        '''
        return {'classes': self.classes, **super().get_config()}