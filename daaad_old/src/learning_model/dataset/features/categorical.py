import os
from typing import List
from src.learning_model.models.heads import InHeadFC, OutHeadFC
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from src.learning_model.dataset.features.data_object import DataObject, DataTransformer
from src.utils import bar_plot, confusion_matrix_plot, scatter_plot, FIG_SIZE

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

        axis = plt.subplots(1, 1, figsize=FIG_SIZE)[1] if kwargs.get('axis', None) is None else kwargs.get('axis')

        for v in np.unique(values):
            scatter_plot('Latent dimension 0', latent_dim_0[values == v], 'Latent dimension 1', latent_dim_1[values == v], 
                            label=str(v), axis=axis, title=kwargs.get('title', None), path=kwargs.get('path', None))
        plt.legend()

        if kwargs.get('title', None) is not None:
            axis.set_title(kwargs['title'])
        if kwargs.get('path', None) is not None:
            os.makedirs(kwargs['path'], exist_ok=True)
            plt.savefig(os.path.join(kwargs['path'], 'latent_scatterplot'), bbox_inches='tight', dpi=400)
        if kwargs.get('axis', None) is None:
            plt.show()
            plt.close()

    def __init__(self, name:str, **kwargs):
        '''
        Parameters
        ----------
        data : np.array
            The data associated with this feature
        name : str
            The name of this feature
        '''

        super(DataCategorical, self).__init__(name, **kwargs)
        self.type = 'categorical'

    def _init_transformer(self, data:np.array) -> DataTransformer:
        self.transformer = OneHotEncoder()
        self.transformer.fit(data)
        return self.transformer

    def set_data(self, data: np.array):
        super().set_data(data)
        self._init_transformer(self.data)
        self.distinct_values = np.unique(self.data)
        
        if len(self.distinct_values) <= 2:
            self.transformer = LabelBinarizer()
            self.transformer.fit(self.data)

    def transform(self, data:np.array) -> np.array:
        '''
        Performs binary encoding if only two unique classes exist, otherwise one-hot encoding.

        Parameters
        ----------
        data : np.array
            The data to be transformed
        '''
        if data.dtype == np.floating:
            res = super(DataCategorical, self).transform(np.around(data).astype(int))
        else:
            res = super(DataCategorical, self).transform(data)
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

        if len(self.distinct_values) <= 2:
            return nn.BCEWithLogitsLoss()
        else:
            return nn.CrossEntropyLoss()

    def get_heads(self, head_layer_widths:List[int], last_core_layer_width:int, activation:nn.Module, name:str='CategoricalHead') -> tuple:
        in_channels = len(self.distinct_values) if len(self.distinct_values) > 2 else 1
        in_head = InHeadFC(in_channels, head_layer_widths, activation)
        return in_head, OutHeadFC(last_core_layer_width, in_head)

    def drop_values(self, drop_values:list):
        '''
        Used to remove values from the data, e.g. for dropping infrequent classes.
        Results in one-hot encodings of all zeros for the values in "drop_values"

        Parameters
        ----------
        drop_values : list
            Classes to be removed from the data
        '''

        if len(self.distinct_values) > 2:
            if not isinstance(drop_values, list):
                drop_values = [drop_values]
            self.distinct_values = [v for v in self.distinct_values if v not in drop_values]
            self.transformer = OneHotEncoder(categories=[[v for v in self.distinct_values]], handle_unknown='ignore')
            self.transformer.fit(self.data)

    def get_config(self):
        '''
        Returns the config necessary for building same instance. 
        Useful for storing / loading the dataset.
        '''
        return {'distinct_values': self.distinct_values, **super().get_config()}