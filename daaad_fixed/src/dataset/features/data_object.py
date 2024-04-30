import numpy as np
import torch.nn as nn
from typing import Dict, List, Union
from copy import deepcopy
from utils import density_plot, scatter_plot, swarm_plot
from dataset.data_transformers import DataTransformer

class DataObject():

    @staticmethod
    def compare(distributions:Dict[str, np.array], title:str, axis=None, path:str=None, evaluation:bool=False):
        '''
        Compare several distributions against each other.

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
        density_plot(distributions, title, axis, path)
        if evaluation:
            assert len(distributions) == 2

            name_a, data_a = list(distributions.items())[0]
            name_b, data_b = list(distributions.items())[1]
            assert data_a.shape == data_b.shape, \
                f'Provided data does not have same shape: {data_a.shape} and {data_b.shape}'
            error = data_a - data_b
            scatter_plot(name_a, data_a, name_b, data_b, 'abs error', np.abs(error), title, True, False, axis, path)
            scatter_plot(name_a, data_a, 'difference', error, 'abs error', np.abs(error), title, False, True, axis, path)
        else:
            scatter_plot(name_a, data_a, name_b, data_b, None, None, title, True, False, axis, path)

    @staticmethod
    def inspect_latent(latent_dim_0:np.array, latent_dim_1:np.array, values:np.array, **kwargs):
        '''
        Creates scatter plot of latent space and assigns colors to points according to "values"

        Parameters
        ----------
        latent_dim_0, latent_dim_1 : np.array
            Coordinates of each point in the latent space
        values : np.array
            Value of each point in the latent space (used for coloring)
        '''
        scatter_plot('Latent dimension 0', latent_dim_0, 'Latent dimension 1', latent_dim_1, name_hue=kwargs.get('title', 'value'), 
                        data_hue=values, title=kwargs.get('title', None), axis=kwargs.get('axis', None), path=kwargs.get('path', None))


    def inspect_discrete_latent(self, class_ixs:np.array, values:np.array, **kwargs):
        '''
        Creates swarm plot of discrete latent space

        Parameters
        ----------
        class_ixs : np.array
            Class to which the samples have been attributed in the latent space.
        values : np.array
            Value of each point in the latent space (used for coloring)
        '''
        swarm_plot(
            {str(i): values[class_ixs == i] for i in range(class_ixs.max() + 1)}, 
            title=kwargs.get('title', self.name + ' vs discrete latent variable'), 
            x_label=self.name, 
            y_label='Latent class', 
            plot_statistics=False, 
            plot_zero=False,
            kwargs=kwargs,
        )
    
    @staticmethod
    def augment(data:np.array) -> np.array:
        '''
        Slightly modifies values in "data" for augmentation
        Identity function by default

        Parameters
        ----------
        data : np.array
            The data to be augmented
        '''
        return data

    @staticmethod
    def get_objective():
        '''
        Returns the loss function for approximating this feature.
        '''
        return nn.MSELoss()
    
    @staticmethod
    def get_heads(head_layer_widths:List[int], last_core_layer_width:int, activation:nn.Module, **kwargs) -> tuple:
        '''
        Returns the NN head necessary for encoding and decoding this feature

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

    @staticmethod
    def validate_data(data:np.array) -> np.array:
        '''
        Brings the data into a valid format or raises an error, if it is not recoverable

        Parameters
        ----------
        data : np.array
            Data to be validated
        '''
        if len(data.shape) == 1:
            return data[:, np.newaxis]
        elif len(data.shape) == 2:
            return data
        else:
            raise Exception('Data should have two dimensions, but has ' + str(len(data.shape)))

    @classmethod
    def from_data(cls, data:np.array, name:str, **kwargs):
        '''
        Creates feature instance from an array of values and a name

        Parameters
        ----------
        data : np.array
            Data to be validated
        name : str
            The name of this feature
        '''
        data_obj = cls(name, **kwargs)
        data_obj.set_data(data)
        return data_obj

    @classmethod
    def from_dict(cls, params:dict):
        '''
        Creates feature instance from a dict

        Parameters
        ----------
        params : dict
            Dict containing all attributes necessary for constructino this feature
        '''
        data_obj = cls(params['name'])
        for k, v in params.items():
            setattr(data_obj, k, v)
        data_obj.data = None
        return data_obj


    def __init__(self, name:str, data_transformer:DataTransformer=None):
        '''
        Parameters
        ----------
        name : str
            The name of this feature
        '''
        self.type = 'data_obj'
        self.name = name
        self.data = None
        self.transformer = data_transformer if data_transformer else DataTransformer.deserialize()

    def _init_transformer(self, data:np.array) -> "DataTransformer":
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
        return self.transformer
        
    def set_data(self, data:np.array):
        """
        Sets the data attribute and initializes the transformer and shape.

        Parameters
        ----------
        data : np.array
            The data to be set as the attribute.
        """
        self.data = self.validate_data(data)
        self.data_type = data.dtype
        self._init_transformer(self.data)
        self.shape = self.transform(self.data[:1]).shape[1:]

    def append_data(self, data:np.array, update_transformer:bool=True):
        """
        Appends new data to the existing attribute and updates the transformer and shape if specified.
        
        Parameters
        ----------
        data : np.array
            The data to be appended to the existing attribute.
        update_transformer : bool
            Whether to update the transformer and shape after appending the data.
        """
        if self.data is None:
            self.set_data(data)
        else:
            self.data = np.vstack([self.data, self.validate_data(data)])
            if update_transformer:
                self._init_transformer(self.data)
                self.shape = self.transform(self.data[1:1]).shape[1:]

    def size(self):
        return len(self.data)

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
        return self.transformer.transform(data)

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
        return self.transformer.inverse_transform(data).astype(self.data_type)

    def get_shape(self) -> tuple:
        """
        Returns the shape of the transformed data.

        Returns
        -------
        tuple
            The shape of the transformed data.
        """
        return self.transformer.transform(self.data[:1]).shape[1:]

    def get_batch(self, ix:np.array, transform:bool, augment:bool=False) -> np.array:
        """
        Returns a batch of data from the object's data attribute.

        Parameters
        ----------
        ix : np.array
            The indices of the data to be returned. Can be an integer, in which case a single sample will be returned, or an array of indices.
        transform : bool
            Whether to transform the data before returning it.
        augment : bool
            Whether to augment the data before returning it.

        Returns
        -------
        np.array
        """
        if isinstance(ix, int):
            if ix < 0:
                ix = np.arange(stop=self.num_samples, step=1, dtype=int)
            else:
                ix = np.arange(start=ix, stop=ix+1, step=1, dtype=int)
            
        if transform:
            res = self.transform(self.data[ix])
        else:
            res = self.data[ix]
        
        if augment:
            return self.augment(res)
        return res

    def inspect(self, axis=None, path:str=None) -> None:
        """
        Visualises the data contained in self.data in adequate plots.

        Parameters
        ----------
        axis : plt.axis
            The axis where the plots should be places. If None, a new axis is created
        path : str
            Where the plots should be saved. If None, plots are not saved
        """
        raise NotImplementedError()

    def get_config(self):
        '''
        Returns the config necessary for building same instance. 
        Useful for storing / loading the dataset.
        '''
        return {
            'name': self.name, 
            'transformer': self.transformer,
            'shape': self.shape,
            'data_type': self.data_type,
        }

    def copy(self, data:np.array=None) -> 'DataObject':
        """
        Creates a deep copy of this instance and adds data to it.

        Parameters
        ----------
        data : np.array
            Data to be added to the object
        """
        new_inst = deepcopy(self)
        if data is not None:
            new_inst.data = data
        return new_inst

    def save_data(self, path:str) -> None:
        """
        Saves the data under a specified path.

        Parameters
        ----------
        path : str
            Where the data should be saved.
        """
        np.save(path, self.data)

    def load_data(self, path:str) -> None:
        """
        Loads the data from a specified path.

        Parameters
        ----------
        path : str
            Where the data should be loaded from.
        """
        self.set_data(np.load(path))