import numpy as np
import torch.nn as nn

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from src.learning_model.dataset.features.real import DataReal
from src.learning_model.dataset.features.data_object import DataTransformer
from src.learning_model.dataset.features.categorical import DataCategorical
from src.utils import bar_plot

class DataOrdinal(DataReal):
    '''
    A class representing ordinal data.
    '''

    @staticmethod
    def compare(distributions:dict, title:str, axis=None, path:str=None, evaluation:bool=False):
        '''
        Compares different sets of ordinal data by producing plots and confusion matrices.

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

        DataCategorical.compare(distributions, title, axis, path, evaluation)

    
    @staticmethod
    def augment(data: np.array) -> np.array:
        return data

    def __init__(self, name:str, **kwargs):
        '''
        Parameters
        ----------
        name : str
            The name of this feature
        '''

        super().__init__(name, **kwargs)
        self.type = 'ordinal'

    def _init_transformer(self, data:np.array) -> DataTransformer:
        self.transformer = OrdinalDataTransformer()
        self.transformer.fit(data)
        return self.transformer

    def set_data(self, data: np.array):
        super().set_data(data)
        self._init_transformer(self.data)
        if self.data_type == np.floating and np.all(np.isclose(self.data, np.around(self.data))):
            self.data_type = int
            self.data = self.data.astype(int)
        self.distinct_values = np.unique(data)

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


class OrdinalDataTransformer(DataTransformer):
    def __init__(self):
        self.ordinal = OrdinalEncoder()
        self.real = MinMaxScaler(feature_range=(-1, 1))

    def fit(self, data:np.array) -> None:
        self.fit_transform(data)

    def fit_transform(self, data:np.array) -> np.array:
        data = self.ordinal.fit_transform(data)
        return self.real.fit_transform(data)

    def transform(self, data:np.array) -> np.array:
        data = self.ordinal.transform(data)
        return self.real.transform(data)
    
    def inverse_transform(self, data:np.array) -> np.array:
        '''
        Transforms encoded data back to original values

        Parameters
        ----------
        data : np.array
            The data to be transformed
        '''

        # First, inverse transform the StandardScaler
        data = self.real.inverse_transform(data)
        # OrdinalEncoder rounds values down per default, but we want to round to nearest integer
        int_data = np.around(data)
        int_data[int_data < 0] = 0
        int_data[int_data > len(self.ordinal.categories_[0])-1] = len(self.ordinal.categories_[0]) - .999
        return self.ordinal.inverse_transform(int_data)