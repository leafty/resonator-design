import os
import torch
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple
from torch.utils.data import DataLoader, Subset

from src.learning_model.dataset.features.real import DataReal
from src.learning_model.dataset.features.ordinal import DataOrdinal
from src.learning_model.dataset.features.data_object import DataObject
from src.learning_model.dataset.features.categorical import DataCategorical
from src.utils import rec_concat_dict

DATA_TYPES = {
    'real': DataReal, 'categorical': DataCategorical, 'ordinal': DataOrdinal,
}

class DataSet(torch.utils.data.Dataset):
    @classmethod
    def from_path(cls, path:str):
        files_in_path = os.listdir(path)
        if 'meta_data.pkl' not in files_in_path:
            raise Exception('No meta_data.pkl file under this path '+path)
        meta_data = MetaData.from_file(path)
        return cls.from_meta_data(meta_data)

    @staticmethod
    def from_meta_data(meta_data:'MetaData'):
        ds = DataSet()
        ds.x = {fdata['name']: DATA_TYPES[ftype].from_dict(fdata) for ftype, fdata in meta_data.config['x']}
        ds.y = {fdata['name']: DATA_TYPES[ftype].from_dict(fdata) for ftype, fdata in meta_data.config['y']}
        
        try:
            x = pd.read_csv(os.path.join(meta_data.config['path'], 'x.csv'))
            y = pd.read_csv(os.path.join(meta_data.config['path'], 'y.csv'))

            for name, data_obj in ds.x.items():
                data_obj.set_data(x[name['name']].to_numpy())
            for name, data_obj in ds.y.items():
                data_obj.set_data(y[name['name']].to_numpy())
        except:
            pass

        return ds

    @staticmethod
    def from_features_list(x:List[DataObject], y:List[DataObject]=None):
        ds = DataSet()
        ds.x = {f.name: f for f in x}
        ds.y = {f.name: f for f in y} if y else {}
        return ds

    def get_size(self) -> int:
        if len(self.x) > 0:
            return list(self.x.values())[0].size()
        elif len(self.y) > 0:
            return list(self.y.values())[0].size()
        else:
            return 0

    def get_shapes(self, add_batch_dim:bool=False, batch_size:int=None, separate:bool=False) -> dict:
        x_shapes = {fname: (batch_size,) + f.get_shape() if add_batch_dim else f.get_shape() for fname, f in self.x.items()}
        y_shapes = {fname: (batch_size,) + f.get_shape() if add_batch_dim else f.get_shape() for fname, f in self.y.items()}
        
        if separate:
            return x_shapes, y_shapes
        else:
            return {**x_shapes, **y_shapes}

    def save(self, path:str, include_data:bool=True):
        os.makedirs(path, exist_ok=True)
        meta_data = MetaData.from_data_set(self)
        meta_data.save(path)

        if include_data:
            X, y = self.gather()
            X.to_csv(os.path.join(path, 'x.csv'))
            y.to_csv(os.path.join(path, 'y.csv'))

    def get_batch(self, ix:np.array, transform:bool=False, augment:bool=False):
        x = {fname: f.get_batch(ix, transform, augment) for fname, f in self.x.items()}
        y = {fname: f.get_batch(ix, transform, augment=False) for fname, f in self.y.items()}
        return x, y

    def set_data(self, data:dict) -> None:
        if isinstance(data, tuple):
            data = data[0] | data[1]
            
        for key, value in data.items():
            if key in self.x.keys():
                self.x[key].set_data(value)
            elif key in self.y.keys():
                self.y[key].set_data(value)
            else:
                raise ValueError('Key ' + str(key) + ' neither in x nor in y: ' + \
                                str(self.x.keys()) + ' ' + str(self.y.keys()))

    def transform(self, data:dict) -> dict:
        return {key: ({**self.x, **self.y})[key].transform(value) for key, value in data.items()}

    def inverse_transform(self, data:dict):
        return {key: ({**self.x, **self.y})[key].inverse_transform(value) for key, value in data.items()}

    def augment(self, data:dict) -> dict:
        return {key: ({**self.x, **self.y})[key].augment(value) for key, value in data.items()}
        
    def inspect(self, path:str=None):
        for f in (self.x | self.y).values():
            f.inspect(axis=None, path=path)

    def get_objectives(self):
        return {fname: f.get_objective() for fname, f in {**self.x, **self.y}.items()}

    def get_transformers(self):
        return {fname: f.get_transformer() for fname, f in {**self.x, **self.y}.items()}

    def get_data_loaders(self, batch_size:int, shuffle:bool=True, val_split:float=0.1, test_split:float=0.2, collate_fn=rec_concat_dict, **kwargs) -> Tuple[DataLoader]:
        assert val_split >= 0 and test_split >= 0 and val_split + test_split < 1, \
            f'Enter valid values for val_split and test_split, must be leq to 0 and their sum smaller than 1: {val_split}, {test_split}'
        
        rnd_ixs = np.arange(0, self.get_size(), dtype=int)
        np.random.shuffle(rnd_ixs)

        train_ixs = rnd_ixs[:int(self.get_size()*(1 - val_split - test_split))]
        val_ixs = rnd_ixs[len(train_ixs):len(train_ixs) + int(self.get_size()*val_split)]
        test_ixs = rnd_ixs[len(train_ixs) + len(val_ixs):self.get_size()]

        ds_train = Subset(self, train_ixs)
        ds_val   = Subset(self, val_ixs)
        ds_test  = Subset(self, test_ixs)

        return (DataLoader(ds_train, batch_size=batch_size, drop_last=True, shuffle=shuffle, collate_fn=collate_fn, **kwargs),
                DataLoader(ds_val, batch_size=len(val_ixs), drop_last=False, shuffle=False, collate_fn=collate_fn, **kwargs),
                DataLoader(ds_test, batch_size=len(val_ixs), drop_last=False, shuffle=False, collate_fn=collate_fn, **kwargs))

    def gather(self):
        return pd.DataFrame(np.hstack([f.data for f in self.x.values()]), columns=list(self.x.keys())), \
                pd.DataFrame(np.hstack([f.data for f in self.y.values()]), columns=list(self.y.keys()))

    def evaluate(self, predictions:dict, targets:dict, return_dict:bool=False, transform:bool=True):
        if transform:
            targets = {k: v.astype(float) for k, v in self.transform(targets).items()}
            predictions = {k: v.astype(float) for k, v in self.transform(predictions).items()}

        rec_loss = {}
        cond_loss = {}
        for k in predictions.keys():
            if k in list(self.x.keys()):
                rec_loss[k] = self.x[k].get_objective()(torch.as_tensor(targets[k]), torch.as_tensor(predictions[k])).numpy()
            elif k in list(self.y.keys()):
                cond_loss[k] = self.y[k].get_objective()(torch.as_tensor(targets[k]), torch.as_tensor(predictions[k])).numpy()
        if return_dict:
            return rec_loss, cond_loss
        else:
            return sum(list(rec_loss.values())), sum(list(cond_loss.values()))
    
    def __len__(self) -> int:
        return self.get_size()

    def __getitem__(self, ix) -> dict:
        return {fname: f.data[ix:ix+1] for fname, f in (self.x | self.y).items()}

class MetaData:

    @staticmethod
    def from_file(path:str):
        with open(os.path.join(path, 'meta_data.pkl'), 'rb') as f:
            config = pickle.load(f)
        return MetaData(config)

    @staticmethod
    def from_data_set(data_set:DataSet):
        config = {
            'x': [(feature.type, feature.get_config()) for feature in data_set.x.values()],
            'y': [(feature.type, feature.get_config()) for feature in data_set.y.values()],
        }
        return MetaData(config)

    def __init__(self, config:dict):
        self.config = config

    def save(self, path:str):
        self.config['path'] = path
        with open(os.path.join(path, 'meta_data.pkl'), 'wb') as f:
            pickle.dump(self.config, f, pickle.HIGHEST_PROTOCOL)


class MultiModalDataSet(DataSet):
    @classmethod
    def from_path(cls, path:str) -> List[MetaData]:
        ds_paths = [ds_path for ds_path in os.listdir(path) if 'mode_' in ds_path]
        return [cls.from_meta_data(MetaData.from_file(path)) for path in ds_paths]

    @staticmethod
    def from_meta_data(meta_data:List[MetaData]) -> List[MetaData]:
        ds = DataSet()
        for md in meta_data:
            ds.x.update({fdata['name']: DATA_TYPES[ftype].from_dict(fdata) for ftype, fdata in md.config['x']})
            ds.y.update({fdata['name']: DATA_TYPES[ftype].from_dict(fdata) for ftype, fdata in md.config['y']})
            
            try:
                x = pd.read_csv(os.path.join(md.config['path'], 'x.csv'))
                y = pd.read_csv(os.path.join(md.config['path'], 'y.csv'))

                for name, data_obj in ds.x.items():
                    data_obj.set_data(x[name['name']].to_numpy())
                for name, data_obj in ds.y.items():
                    data_obj.set_data(y[name['name']].to_numpy())
            except:
                pass

        return ds

    @staticmethod
    def from_features_list(feature_lists:List[Tuple[List[DataObject], List[DataObject]]]):
        ds = DataSet()
        for x, y in feature_lists:
            ds.x.update({f.name: f for f in x})
            ds.y.update({f.name: f for f in y} if y else {})
        return ds

    def __init__(self, datasets:List[DataSet], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datasets = datasets
        
        self.max_len = max([len(ds) for ds in datasets])
        self.x, self.y = {}, {}
        for ds in datasets:
            x, y = ds.get_batch(np.arange(0, self.max_len % len(ds)), dtype=int)
            self.x.update(x)
            self.y.update(y)

    def __len__(self) -> List[int]:
        return self.max_len