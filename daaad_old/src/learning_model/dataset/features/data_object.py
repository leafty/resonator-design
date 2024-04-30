import numpy as np
import torch.nn as nn
from copy import deepcopy
from src.utils import density_plot, scatter_plot

class DataObject():

    @staticmethod
    def compare(distributions:dict, title:str, axis=None, path:str=None, evaluation:bool=False):
        density_plot(distributions, title, axis, path)
        if evaluation:
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
        scatter_plot('Latent dimension 0', latent_dim_0, 'Latent dimension 1', latent_dim_1, name_hue=kwargs.get('title', 'value'), 
                        data_hue=values, title=kwargs.get('title', None), axis=kwargs.get('axis', None), path=kwargs.get('path', None))
    
    @staticmethod
    def augment(data:np.array) -> np.array:
        return data

    @staticmethod
    def get_objective():
        return nn.MSELoss()
    
    @staticmethod
    def get_heads(head_latent_dims:list, latent_dim:int, activation:nn.Module) -> tuple:
        raise NotImplementedError()

    @staticmethod
    def validate_data(data:np.array) -> np.array:
        if len(data.shape) == 1:
            return data[:, np.newaxis]
        elif len(data.shape) == 2:
            return data
        else:
            raise Exception('Data should have two dimensions, but has ' + str(len(data.shape)))

    @classmethod
    def from_data(cls, data:np.array, name:str):
        data_obj = cls(name)
        data_obj.set_data(data)
        return data_obj

    @classmethod
    def from_dict(cls, params:dict):
        data_obj = cls(params['name'])
        for k, v in params.items():
            setattr(data_obj, k, v)
        data_obj.data = None
        return data_obj

    def __init__(self, name:str, data:np.array=None):
        '''
        Parameters
        ----------
        name : str
            The name of this feature
        data : np.array
            The data associated with this feature
        '''

        self.name = name

        if data is not None:
            self.set_data(data)

    def _init_transformer(self, data:np.array) -> "DataTransformer":
        self.transformer = IdentityTransformer()
        return self.transformer
        
    def set_data(self, data:np.array):
        self.data = self.validate_data(data)
        self.data_type = data.dtype
        self.max_value, self.min_value = data.max(), data.min()
        self._init_transformer(self.data)

    def size(self):
        return len(self.data)

    def transform(self, data:np.array) -> np.array:
        return self.transformer.transform(data)

    def inverse_transform(self, data:np.array) -> np.array:
        return self.transformer.inverse_transform(data).reshape((-1, 1)).astype(self.data_type)

    def get_shape(self) -> tuple:
        return self.transformer.transform(self.data[:1]).shape[1:]

    def get_batch(self, ix:np.array, transform:bool, augment:bool=False) -> np.array:
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

    def inspect(self, path:str=None):
        raise NotImplementedError()

    def get_config(self):
        return {
            'name': self.name, 
            'transformer': self.transformer,
            'max_value': self.max_value,
            'min_value': self.min_value,
            'data_type': self.data_type,
        }

    def copy(self, data:np.array=None) -> 'DataObject':
        new_inst = deepcopy(self)
        if data is not None:
            new_inst.data = data
        return new_inst


class DataTransformers:
    def __init__(self, transformers:list):
        if isinstance(transformers, list):
            self.transformers = transformers
        else:
            self.transformers = [transformers]

    def fit(self, data:np.array):
        self.fit_transform(data)

    def fit_transform(self, data:np.array) -> np.array:
        transf_data = data
        for t in self.transformers:
            transf_data = t.fit_transform(transf_data)
        return transf_data

    def transform(self, data:np.array) -> np.array:
        transf_data = data
        for t in self.transformers:
            transf_data = t.transform(transf_data)
        return transf_data

    def inverse_transform(self, transf_data:np.array) -> np.array:
        data = transf_data
        for t in self.transformers[::-1]:
            data = t.inverse_transform(data)
        return data

class DataTransformer:
    def fit(self, data:np.array) -> None:
        pass

    def fit_transform(self, data:np.array) -> np.array:
        return data

    def transform(self, data:np.array) -> np.array:
        raise NotImplementedError()
    
    def inverse_transform(self, data:np.array) -> np.array:
        raise NotImplementedError()


class IdentityTransformer(DataTransformer):
    def transform(self, data:np.array) -> np.array:
        return data
    
    def inverse_transform(self, data:np.array) -> np.array:
        return data


class LogTransformer(DataTransformer):
    def transform(self, data:np.array) -> np.array:
        return np.log(data)
    
    def inverse_transform(self, data:np.array) -> np.array:
        return np.exp(data)