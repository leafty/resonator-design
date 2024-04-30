import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List
from multiprocessing import Pool
from src.learning_model.dataset.features.real import DataReal
from src.learning_model.dataset.features.ordinal import DataOrdinal
from src.learning_model.dataset.features.data_object import DataObject
from src.learning_model.dataset.features.categorical import DataCategorical

class DataBlock:
    def __init__(self, features:List[DataObject], name:str='data_block'):
        self.features = features
        self.name = name

    def get_batch(self, ix:np.array, transform:bool=True, augment:bool=False):
        #with Pool() as p:
        #    return np.concatenate(p.map(lambda f: f.get_batch(ix, transform, augment), self.features), axis=-1)
        return {f.name: f.get_batch(ix, transform, augment) for f in self.features}

    def get_size(self) -> int:
        return self.features[0].num_samples

    def get_shapes(self) -> dict:
        return {f.name: f.get_shape() for f in self.features}

    def inspect(self, path:str=None):
        fig, axes = plt.subplots(len(self.features), 1, figsize=(12, len(self.features)*4))

        for i, f in enumerate(self.features):
            f.inspect(axes[i], path)
        fig.suptitle(self.name)
        plt.show()

    def union(self, block:'DataBlock') -> 'DataBlock':
        return DataBlock(self.features + block.features, name=self.name + '_union_' + block.name)

    def intersection(self, block:'DataBlock') -> 'DataBlock':
        new_features = []
        new_features_names = []
        for f in self.features + block.features:
            if f.name not in new_features_names:
                new_features.append(f)
        return DataBlock(new_features, name=self.name + '_inters_' + block.name)

    def drop_features(self, feats_to_drop:List[str]) -> 'DataBlock':
        return DataBlock([f for f in self.features if f.name not in feats_to_drop], name=self.name + '_dropped_' + str(feats_to_drop))

    def get_objectives(self):
        objectives = {}
        for f in self.features:
            objectives.update(f.get_objective())
        return objectives


def create_DataBlock(data:pd.DataFrame, feature_names:List[str]=None, feature_types:List[str]=None, name:str='data_block') -> DataBlock:
    if feature_names is None:
        feature_names = list(data.columns)

    features = []
    for i, name in enumerate(feature_names):
        if feature_types is not None and feature_types[i] == 'real' or \
            data[name].dtype in [np.dtype('float64'), np.dtype('float32')]:
            features.append(DataReal(data[name], name))
        elif feature_types is not None and feature_types[i] == 'ordinal':
            features.append(DataOrdinal(data[name], name))
        elif feature_types is not None and feature_types[i] == 'categorical' or \
            data[name].dtype not in [np.dtype('float64'), np.dtype('float32')]:
            features.append(DataCategorical(data[name], name))
        else:
            raise Exception('feature '+name+' could not attributed to any of the data types (real, ordinal, categorical)')
    return DataBlock(features, name)


def create_real_DataBlock(data:pd.DataFrame, feature_names:List[str]=None, feature_scaling_types:List[str]=None, name:str='real_data_block') -> DataBlock:
    if feature_scaling_types is None:
        feature_scaling_types = ['standardize']*len(data)
    features = [DataReal(data[fname].to_numpy(), fname, scaling) for fname, scaling in zip(feature_names, feature_scaling_types)]
    return DataBlock(features, name)

def create_ordinal_DataBlock(data:pd.DataFrame, feature_names:List[str]=None, name:str='ordinal_data_block') -> DataBlock:
    features = [DataOrdinal(data[fname].to_numpy(), fname) for fname in feature_names]
    return DataBlock(features, name)

def create_categorical_DataBlock(data:pd.DataFrame, feature_names:List[str]=None, name:str='categorical_data_block') -> DataBlock:
    features = [DataCategorical(data[fname].to_numpy(), fname) for fname in feature_names]
    return DataBlock(features, name)