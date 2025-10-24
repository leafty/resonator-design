import numpy as np
from typing import List

from sklearn.preprocessing import QuantileTransformer, KBinsDiscretizer

MAX_QUANTILES = 1000

class Sampler:
    '''
    Allows to sample points according to some range and distribution
    '''
    def transform(self, values:np.array) -> np.array:
        '''
        Transforms data from a dataset to a value between 0 and 1

        Parameters
        ----------
        values : np.array
            Data to be mapped between 0 and 1
        '''
        pass

    def inverse_transform(self, values:np.array):
        '''
        Transforms data ranging from 0 to 1 to its original form

        Parameters
        ----------
        values : np.array
            Data between 0 and 1 to be mapped back
        '''
        pass

class UniformSampler(Sampler):
    '''
    Allows to sample points uniformly between min_val and max_val value
    '''
    def __init__(self, data:np.array=None, min_val:float=None, max_val:float=None):
        super().__init__()
        if data is None and min_val is None and max_val is None:
            raise ValueError('Either "data" or "min_val" and "max_val" must be provided')
        self.min_val = min_val if min_val is not None else data.min()
        self.max_val = max_val if max_val is not None else data.max()

    def transform(self, values:np.array):
        '''
        Transforms continuous data from min_val to max_val to the range 0 to 1

        Parameters
        ----------
        values : np.array
            Data to be mapped between 0 and 1
        '''
        return (values - self.min_val) / (self.max_val - self.min_val)

    def inverse_transform(self, values:np.array):
        '''
        Transforms data ranging from 0 to 1 to the range min_val to max_val

        Parameters
        ----------
        values : np.array
            Data between 0 and 1 to be mapped back
        '''
        return values * (self.max_val - self.min_val) + self.min_val

class QuantileSampler(Sampler):
    '''
    Allows to sample points between min_val and max_val following the distribution in "data" using QuantileTransformer
    '''
    def __init__(self, data:np.array, min_val:float=None, max_val:float=None):
        '''
        Parameters
        ----------
        data : np.array
            Data whose distribution should serve as basis for sampling
        '''
        super().__init__()
        self.min_val = min_val if min_val is not None else data.min()
        self.max_val = max_val if max_val is not None else data.max()
        self.qt = QuantileTransformer(n_quantiles=min(MAX_QUANTILES, len(data)), output_distribution="uniform", random_state=42)
        self.qt.fit(data[(data >= self.min_val) & (data <= self.max_val)].reshape(-1, 1))

    def transform(self, values: np.array):
        '''
        Transforms continuous data from min_val to max_val to the range 0 to 1

        Parameters
        ----------
        values : np.array
            Data to be mapped between 0 and 1
        '''
        return self.qt.transform(values.reshape(-1, 1)).reshape(values.shape)

    def inverse_transform(self, values: np.array):
        '''
        Transforms data ranging from 0 to 1 to the range min_val to max_val

        Parameters
        ----------
        values : np.array
            Data between 0 and 1 to be mapped back
        '''
        return self.qt.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)

class CategoricalUniformSampler(Sampler):
    '''
    Allows to sample discrete points from a set of given classes uniformly
    '''
    def __init__(self, data:np.array=None, classes:List[object]=None):
        '''
        Either data or classes must be provided

        Parameters
        ----------
        data : np.array
            Data array containing all classes that should be sampled from uniformly
        classes : List[object]
            List of elements counting as classes to be sampled from uniformly
        '''
        super().__init__()
        if data is None and classes is None:
            raise ValueError('Either "data" or "classes" must be provided')
        self.classes = classes if classes is not None else np.unique(data)
        self.classes_ixs = {c: i for i, c in enumerate(self.classes)}

    def transform(self, values: np.array):
        '''
        Transforms categorical data containing self.classes to the range 0 to 1

        Parameters
        ----------
        values : np.array
            Data to be mapped between 0 and 1
        '''
        ixs = np.array([self.classes_ixs[values.flatten()[i]] for i in range(len(values))])
        return ixs / len(self.classes)

    def inverse_transform(self, values: np.array):
        '''
        Transforms data ranging from 0 to 1 to categorical values

        Parameters
        ----------
        values : np.array
            Data between 0 and 1 to be mapped back
        '''
        return self.classes[(values * len(self.classes)).astype(int)].reshape(-1, 1)


class CategoricalQuantileSampler(CategoricalUniformSampler):
    '''
    Allows to sample discrete points from set of given classes following the distribution in "data"
    '''
    def __init__(self, data:np.array=None):
        '''
        Creates histogram of unique values in data, converts into prob. distr. and uses
        cumulative sum to get edges for binning

        Parameters
        ----------
        data : np.array
            Data array containing all classes that should be sampled from and whose distribution
            should be followed
        '''
        super().__init__(data)
        
        self.bd = KBinsDiscretizer(n_bins=len(self.classes), encode='ordinal', strategy="uniform", random_state=42)
        h = np.histogram(super().transform(data))
        freq = h[0][h[0] > 0] / len(data)
        edges = np.concatenate([np.array([0.]), np.cumsum(freq)], axis=0)

        self.bd.fit(super().transform(data).reshape(-1, 1))
        self.bd.bin_edges_[0] = edges

    def transform(self, values: np.array):
        '''
        Transforms categorical data containing self.classes to the range 0 to 1,
        dividing the space according to prob. distr. of the classes

        Parameters
        ----------
        values : np.array
            Data to be mapped between 0 and 1
        '''
        return self.bd.inverse_transform(np.array([self.classes_ixs[values.flatten()[i]] for i in range(len(values))]).reshape(-1, 1)).reshape(values.shape)

    def inverse_transform(self, values: np.array):
        '''
        Transforms data ranging from 0 to 1 to categorical values

        Parameters
        ----------
        values : np.array
            Data between 0 and 1 to be mapped back
        '''
        return self.classes[self.bd.transform(values.reshape(-1, 1)).astype(int)[:, 0]].reshape(-1, 1)