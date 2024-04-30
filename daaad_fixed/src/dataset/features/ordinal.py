from typing import Union
from dataset.data_transformers import DataTransformerSequence
import numpy as np

from dataset.features.real import DataReal
from dataset.features.data_object import DataTransformer
from dataset.features.categorical import DataCategorical
from utils import bar_plot

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
        '''
        Slightly modifies values in "data" for augmentation
        Identity function because there is no easy way of augmenting ordinal data.

        Parameters
        ----------
        data : np.array
            The data to be augmented
        '''
        return data

    def __init__(self, name: str, classes: Union[list, np.array]=None, data_transformer:DataTransformer=None, **kwargs):
        '''
        Parameters
        ----------
        name : str
            The name of this feature
        classes : Union[list, np.array]
            Optional, a ordered list or numpy array of distinct classes. 
            If not provided, will be infered from data.
        '''

        super().__init__(name, data_transformer, **kwargs)
        self.type = 'ordinal'
        self.classes = classes
        
        if data_transformer is None:
            self.transformer = DataTransformerSequence(['ordinal', 'standard'])

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
            
        self._init_transformer(data)
        if self.data_type == np.floating and np.all(np.isclose(self.data, np.around(self.data))):
            self.data_type = int
            self.data = self.data.astype(int)
        self.classes = np.unique(data)

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
        super().append_data(data, update_transformer)

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

    def get_config(self):
        '''
        Returns the config necessary for building same instance. 
        Useful for storing / loading the dataset.
        '''
        return super().get_config() | {'classes': self.classes}

