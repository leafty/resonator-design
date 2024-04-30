import numpy as np
from typing import List, Union, Tuple

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, OneHotEncoder, LabelBinarizer


class DataTransformer:
    @staticmethod
    def deserialize(identifier:Union['DataTransformer', str, List['DataTransformer']]=None) -> 'DataTransformer':
        """
        Deserialize a DataTransformer object.

        Parameters:
        identifier (Union[DataTransformer, str, List[DataTransformer]]): 
        The serialized identifier for the DataTransformer. 
        This can be a DataTransformer object, a string, or a list of DataTransformer objects. 
        If None, an IdentityDataTransformer object is returned.

        Returns:
        DataTransformer: The deserialized DataTransformer object.
        
        Raises:
        ValueError: If the identifier for the DataTransformer is unknown.
        """
        if isinstance(identifier, DataTransformer):
            return identifier
        if isinstance(identifier, list):
            return DataTransformerSequence([DataTransformer.deserialize(id) for id in identifier])
        if identifier is None or identifier == 'identity':
            return IdentityDataTransformer()
        if identifier in ['minmax', 'norm_0to1']:
            return MinMaxDataTransformer(feature_range=(0, 1))
        if identifier in ['minmax_-1to1', 'norm_m1to1']:
            return MinMaxDataTransformer(feature_range=(-1, 1))
        if identifier in ['standard', 'standardize']:
            return StandardDataTransformer()
        if identifier in ['onehot', 'categorical', 'binary']:
            return OneHotDataTransformer()
        if identifier in ['ordinal']:
            return OrdinalDataTransformer()
        if identifier == 'log':
            return LogDataTransformer()
        
        raise ValueError(f'Identifier for DataTransformer {identifier} is unknown. \
            Valid identifiers are "standard", "minmax", "norm_0to1" or "norm_m1to1".')

    def serialize(self):
        """
        Serialize the DataTransformer object.

        Returns:
        str: The serialized identifier for the DataTransformer.
        """
        return self.identifier

    def fit(self, data:np.array) -> None:
        """
        Fit the DataTransformer object to the data.

        Parameters:
        data (np.array): The data to fit the DataTransformer object to.

        Returns:
        None
        """
        self.transformer.fit(data)

    def fit_transform(self, data:np.array) -> np.array:
        """
        Fit the DataTransformer object to the data and transform it.

        Parameters:
        data (np.array): The data to fit the DataTransformer object to and transform.

        Returns:
        np.array: The transformed data.
        """
        return self.transformer.fit_transform(data)

    def transform(self, data:np.array) -> np.array:
        """
        Transform the data using the DataTransformer object.

        Parameters:
        data (np.array): The data to transform.

        Returns:
        np.array: The transformed data.
        """
        return self.transformer.transform(data)
    
    def inverse_transform(self, data:np.array) -> np.array:
        """
        Transform the data back to its base form using the DataTransformer object.

        Parameters:
        data (np.array): The data to transform.

        Returns:
        np.array: The transformed data.
        """
        return self.transformer.inverse_transform(data)


class IdentityDataTransformer(DataTransformer):
    """
    A DataTransformer that applies no transformation to the data.
    """
    def __init__(self):
        """
        Initialize the IdentityDataTransformer object.
        """
        super().__init__()
        self.identifier = 'identity'

    def fit(self, data:np.array) -> None:
        """
        Fit the IdentityDataTransformer object to the data. 
        This method does nothing as the IdentityDataTransformer does not require fitting.

        Parameters:
        data (np.array): The data to fit the IdentityDataTransformer object to.

        Returns:
        None
        """
        pass

    def fit_transform(self, data:np.array) -> np.array:
        """
        Fit the IdentityDataTransformer object to the data and transform it. 
        This method simply returns the input data as the IdentityDataTransformer does not require fitting and applies no transformation.

        Parameters:
        data (np.array): The data to fit the IdentityDataTransformer object to and transform.

        Returns:
        np.array: The input data.
        """
        return data

    def transform(self, data:np.array) -> np.array:
        """
        Transform the data using the IdentityDataTransformer object. 
        This method simply returns the input data as the IdentityDataTransformer applies no transformation.

        Parameters:
        data (np.array): The data to transform.

        Returns:
        np.array: The input data.
        """
        return data
    
    def inverse_transform(self, data:np.array) -> np.array:
        """
        Inverse transform the data using the IdentityDataTransformer object. 
        This method simply returns the input data as the IdentityDataTransformer applies no transformation.

        Parameters:
        data (np.array): The data to inverse transform.

        Returns:
        np.array: The input data.
        """
        return data


class MinMaxDataTransformer(DataTransformer):
    """
    A DataTransformer that scales the data using the min-max method.
    """
    def __init__(self, feature_range:Tuple[int]):
        """
        Initialize the MinMaxDataTransformer object.

        Parameters:
        feature_range (Tuple[int]): The range of the transformed data.
        """
        self.identifier = f'minmax_{feature_range[0]}to{feature_range[1]}'
        self.transformer = MinMaxScaler(feature_range=feature_range)


class StandardDataTransformer(DataTransformer):
    """
    A DataTransformer that standardizes the data.
    """
    def __init__(self):
        """
        Initialize the StandardDataTransformer object.
        """
        self.transformer = StandardScaler()
        self.identifier = 'standard'


class OneHotDataTransformer(DataTransformer):
    """
    A DataTransformer that one-hot encodes categorical data.
    """
    def __init__(self):
        """
        Initialize the OneHotDataTransformer object.
        """
        self.transformer = OneHotEncoder()
        self.identifier = 'onehot'
        
    def fit(self, data:np.array) -> None:
        """
        Fit the OneHotDataTransformer object to the data. 
        If the number of distinct classes is 1 or 2, a LabelBinarizer is used.
        Otherwise, a OneHotEncoder is used.

        Parameters:
        data (np.array): The data to fit the OneHotDataTransformer object to.

        Returns:
        None
        """
        if len(np.unique(data)) > 2:
            self.transformer = OneHotEncoder()
        else:
            self.transformer = LabelBinarizer()
        self.transformer.fit(data)

    def fit_transform(self, data: np.array) -> np.array:
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.array) -> np.array:
        inv_data = super().inverse_transform(data)
        if len(inv_data.shape) == 1:
            inv_data = inv_data.reshape(-1, 1)
        return inv_data

class LogDataTransformer(DataTransformer):
    """
    A DataTransformer that applies the natural logarithm function to the data.
    """
    def __init__(self):
        """
        Initialize the LogDataTransformer object.
        """
        self.identifier = 'log'

    def transform(self, data:np.array) -> np.array:
        """
        Transform the data using the LogDataTransformer object.

        Parameters:
        data (np.array): The data to transform.

        Returns:
        np.array: The transformed data.
        """
        return np.log(data)
    
    def inverse_transform(self, data:np.array) -> np.array:
        """
        Inverse transform the data using the LogDataTransformer object.

        Parameters:
        data (np.array): The data to inverse transform.

        Returns:
        np.array: The inverse transformed data.
        """
        return np.exp(data)

class ClipToRangeDataTransformer(DataTransformer):
    """
    A DataTransformer that clips the data to a specified range.
    """
    def __init__(self, minval:float, maxval:float):
        """
        Initialize the ClipToRangeDataTransformer object.

        Parameters:
        minval (float): The minimum value of the range.
        maxval (float): The maximum value of the range.
        """
        self.identifier = 'clip_to_range'
        self.minval = minval
        self.maxval = maxval

    def transform(self, data:np.array) -> np.array:
        """
        Transform the data using the ClipToRangeDataTransformer object.

        Parameters:
        data (np.array): The data to transform.

        Returns:
        np.array: The transformed data.
        """
        data_copy = np.copy(data)
        data_copy[data_copy < self.minval] = self.minval
        data_copy[data_copy > self.maxval] = self.maxval
        return data_copy
    
    def inverse_transform(self, data:np.array) -> np.array:
        """
        Inverse transform the data using the ClipToRangeDataTransformer object.

        Parameters:
        data (np.array): The data to inverse transform.

        Returns:
        np.array: The inverse transformed data.
        """
        data_copy = np.copy(data)
        data_copy[data_copy < self.minval] = self.minval
        data_copy[data_copy > self.maxval] = self.maxval
        return data_copy

class ClipLowerDataTransformer(DataTransformer):
    """
    A DataTransformer that clips the data to a minimum value.
    """
    def __init__(self, minval:float):
        """
        Initialize the ClipLowerDataTransformer object.

        Parameters:
        minval (float): The minimum value to clip the data to.
        """
        self.identifier = 'clip_lower'
        self.minval = minval

    def transform(self, data:np.array) -> np.array:
        """
        Transform the data using the ClipLowerDataTransformer object.

        Parameters:
        data (np.array): The data to transform.

        Returns:
        np.array: The transformed data.
        """
        data_copy = np.copy(data)
        data_copy[data_copy < self.minval] = self.minval
        return data_copy
    
    def inverse_transform(self, data:np.array) -> np.array:
        """
        Inverse transform the data using the ClipLowerDataTransformer object.

        Parameters:
        data (np.array): The data to inverse transform.

        Returns:
        np.array: The inverse transformed data.
        """
        data_copy = np.copy(data)
        data_copy[data_copy < self.minval] = self.minval
        return data_copy

class ClipUpperDataTransformer(DataTransformer):
    """
    A DataTransformer that clips the data to a maximum value.
    """
    def __init__(self, maxval:float):
        """
        Initialize the ClipUpperDataTransformer object.

        Parameters:
        maxval (float): The maximum value to clip the data to.
        """
        self.identifier = 'clip_upper'
        self.maxval = maxval

    def transform(self, data:np.array) -> np.array:
        """
        Transform the data using the ClipUpperDataTransformer object.

        Parameters:
        data (np.array): The data to transform.

        Returns:
        np.array: The transformed data.
        """
        data_copy = np.copy(data)
        data_copy[data_copy > self.maxval] = self.maxval
        return data_copy
    
    def inverse_transform(self, data:np.array) -> np.array:
        """
        Inverse transform the data using the ClipUpperDataTransformer object.

        Parameters:
        data (np.array): The data to inverse transform.

        Returns:
        np.array: The inverse transformed data.
        """
        data_copy = np.copy(data)
        data_copy[data_copy > self.maxval] = self.maxval
        return data_copy


class OrdinalDataTransformer(DataTransformer):
    """
    A DataTransformer that ordinal encodes categorical data and scales it using the min-max method.
    """
    def __init__(self):
        """
        Initialize the OrdinalDataTransformer object.
        """
        self.ordinal = OrdinalEncoder()
        self.real = MinMaxScaler(feature_range=(-1, 1))

    def fit(self, data:np.array) -> None:
        """
        Fit the OrdinalDataTransformer object to the data. This method calls the fit_transform method.

        Parameters:
        data (np.array): The data to fit the OrdinalDataTransformer object to.

        Returns:
        None
        """
        self.fit_transform(data)

    def fit_transform(self, data:np.array) -> np.array:
        """
        Fit the OrdinalDataTransformer object to the data and transform it.

        Parameters:
        data (np.array): The data to fit the OrdinalDataTransformer object to and transform.

        Returns:
        np.array: The transformed data.
        """
        data = self.ordinal.fit_transform(data)

        return self.real.fit_transform(data)

    def transform(self, data:np.array) -> np.array:
        """
        Transform the data using the OrdinalDataTransformer object.

        Parameters:
        data (np.array): The data to transform.

        Returns:
        np.array: The transformed data.
        """
        data = self.ordinal.transform(data)
        return self.real.transform(data)
    
    def inverse_transform(self, data:np.array) -> np.array:
        """
        Inverse transform the data using the OrdinalDataTransformer object.

        Parameters:
        data (np.array): The data to inverse transform.

        Returns:
        np.array: The inverse transformed data.
        """
        # First, inverse transform the StandardScaler
        data = self.real.inverse_transform(data)
        # OrdinalEncoder rounds values down per default, but we want to round to nearest integer
        int_data = np.around(data)
        int_data[int_data < 0] = 0
        int_data[int_data > len(self.ordinal.categories_[0])-1] = len(self.ordinal.categories_[0]) - .999
        return self.ordinal.inverse_transform(int_data)


class DataTransformerSequence:
    """
    A class for chaining multiple DataTransformer objects together.
    """
    def __init__(self, transformers:list):
        """
        Initialize the DataTransformerSequence object.

        Parameters:
        transformers (list): A list of DataTransformer objects to chain together.
        """
        if isinstance(transformers, list):
            self.transformers = []
            for t in transformers:
                if isinstance(t, DataTransformer):
                    self.transformers.append(t)
                elif isinstance(t, DataTransformerSequence):
                    self.transformers += t.transformers
                elif isinstance(t, str):
                    self.transformers.append(DataTransformer.deserialize(t))
                else:
                    raise ValueError(f'Type of transformers {transformers.__class__} not in list of valid types: \
                        list, DataTransformer, DataTransformerSequence')
        elif isinstance(transformers, DataTransformerSequence):
            self.transformers = transformers.transformers
        elif isinstance(transformers, DataTransformer):
            self.transformers = [transformers]
        else:
            raise ValueError('Type of transformers {transformers.__class__} not in list of valid types: \
                list, DataTransformer, DataTransformerSequence')

    def fit(self, data:np.array):
        """
        Fit the DataTransformerSequence object to the data. This method calls the fit_transform method.

        Parameters:
        data (np.array): The data to fit the DataTransformerSequence object to.

        Returns:
        None
        """
        self.fit_transform(data)

    def fit_transform(self, data:np.array) -> np.array:
        """
        Fit the DataTransformerSequence object to the data and transform it.

        Parameters:
        data (np.array): The data to fit the DataTransformerSequence object to and transform.

        Returns:
        np.array: The transformed data.
        """
        transf_data = data
        for t in self.transformers:
            transf_data = t.fit_transform(transf_data)
        return transf_data

    def transform(self, data:np.array) -> np.array:
        """
        Transform the data using the DataTransformerSequence object.

        Parameters:
        data (np.array): The data to transform.

        Returns:
        np.array: The transformed data.
        """
        transf_data = data
        for t in self.transformers:
            transf_data = t.transform(transf_data)
        return transf_data

    def inverse_transform(self, transf_data:np.array) -> np.array:
        """
        Inverse transform the data using the DataTransformerSequence object.

        Parameters:
        transf_data (np.array): The data to transform.

        Returns:
        np.array: The transformed data.
        """
        data = transf_data
        for t in self.transformers[::-1]:
            data = t.inverse_transform(data)
        return data