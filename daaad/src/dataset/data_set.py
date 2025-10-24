import os
import torch
import pickle
import numpy as np

from typing import Dict, List, Tuple, Union
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.trainer.supporters import CombinedLoader

from utils import make_path_unique, rec_concat_dict
from dataset.data_module import DataModule
from dataset.features.real import DataReal
from dataset.features.images import DataImage
from dataset.features.matrix import DataMatrix
from dataset.features.ordinal import DataOrdinal
from dataset.features.data_object import DataObject
from dataset.features.categorical import DataCategorical
from dataset.data_transformers import DataTransformer

DATA_TYPES = {
    "real": DataReal,
    "categorical": DataCategorical,
    "ordinal": DataOrdinal,
    "matrix": DataMatrix,
    "image": DataImage,
}


class DataSet(torch.utils.data.Dataset):
    @classmethod
    def from_path(cls, path: str) -> "DataSet":
        """
        Create a DataSet object from a path containing data stored in the format specified in the MetaData class.

        Parameters:
        path (str): Path to the directory containing the data and meta_data.pkl file.

        Returns:
        DataSet: The created DataSet object.
        """
        files_in_path = os.listdir(path)
        if "meta_data.pkl" not in files_in_path:
            raise Exception("No meta_data.pkl file under this path " + path)
        meta_data = MetaData.from_file(path)
        return cls.from_meta_data(meta_data)

    @staticmethod
    def from_meta_data(meta_data: "MetaData") -> "DataSet":
        """
        Create a DataSet object from a MetaData object.

        Parameters:
        meta_data (MetaData): MetaData object containing the data configuration.

        Returns:
        DataSet: The created DataSet object.
        """
        cls = DataSet if meta_data.config.get("include_data", False) else MetaDataSet
        ds = cls.from_features_list(
            [
                DATA_TYPES[ftype].from_dict(fconfig)
                for ftype, fconfig in meta_data.config["x"]
            ],
            [
                DATA_TYPES[ftype].from_dict(fconfig)
                for ftype, fconfig in meta_data.config["y"]
            ],
            name=meta_data.config["name"],
        )

        if meta_data.config.get("include_data", False):
            data_path = os.path.join(meta_data.config["path"], "data")
            for fname, fobject in (ds.x | ds.y).items():
                fobject.load_data(os.path.join(data_path, fname))

        return ds

    @classmethod
    def from_features_list(
        cls, x: List[DataObject], y: List[DataObject] = None, name: str = "default"
    ) -> "DataSet":
        """
        Creates a DataSet object from lists of DataObjects representing the input and target data.

        Parameters:
        x (List[DataObject]): A list of DataObjects representing the input data.
        y (List[DataObject], optional): A list of DataObjects representing the target data.
        name (str, optional): Name of the dataset.

        Returns:
        DataSet: The created DataSet object.
        """
        ds = cls(name)
        ds.x = {f.name: f for f in x}
        ds.y = {f.name: f for f in y} if y else {}
        return ds

    def __init__(self, name: str):
        self.name = name

    def get_size(self) -> int:
        """
        Returns the size of the DataSet, i.e. the number of data points in it.

        Returns:
        int: The size of the DataSet.
        """
        if len(self.x) > 0:
            return list(self.x.values())[0].size()
        elif len(self.y) > 0:
            return list(self.y.values())[0].size()
        else:
            return 0

    def get_shapes(
        self,
        add_batch_dim: bool = False,
        batch_size: int = None,
        separate: bool = False,
    ) -> Union[Tuple[Dict[str, tuple], Dict[str, tuple]], Dict[str, tuple]]:
        """
        Get the shape of the features in the dataset.

        Parameters:
        add_batch_dim (bool): Whether to include the batch dimension in the returned shape.
        batch_size (int): The size of the batch dimension to include.
        separate (bool): Whether to return the shapes of the input and output features separately.

        Returns:
        Union[Tuple[Dict[str, tuple], Dict[str, tuple]], Dict[str, tuple]]: The shapes of the features.
        """
        x_shapes = {
            fname: (batch_size,) + f.get_shape() if add_batch_dim else f.get_shape()
            for fname, f in self.x.items()
        }
        y_shapes = {
            fname: (batch_size,) + f.get_shape() if add_batch_dim else f.get_shape()
            for fname, f in self.y.items()
        }

        if separate:
            return x_shapes, y_shapes
        else:
            return {**x_shapes, **y_shapes}

    def save(self, path: str, include_data: bool = True):
        """
        Save the dataset to a given path.

        Parameters:
        path (str): The path to save the dataset to.
        include_data (bool): Whether to include the actual data in the saved dataset.
        """
        os.makedirs(path, exist_ok=True)
        meta_data = MetaData.from_data_set(self)
        meta_data.config["include_data"] = include_data
        meta_data.save(path)

        if include_data:
            data_path = os.path.join(path, "data")
            os.makedirs(data_path, exist_ok=True)
            for fname, fobject in (self.x | self.y).items():
                feature_path = os.path.join(data_path, fname)
                os.makedirs(feature_path, exist_ok=True)
                fobject.save_data(feature_path)

    def get_batch(
        self, ix: np.array, transform: bool = False, augment: bool = False
    ) -> Tuple[Dict[str, np.array], Dict[str, np.array]]:
        """
        Retrieve a batch of data from the dataset based on the provided indices.
        If `transform` is True, the data will be transformed before being returned.
        If `augment` is True, the data will be augmented before being returned.

        Parameters:
        ix (np.array): The indices of the data points to retrieve.
        transform (bool): Whether to apply the transformation to the data before returning it.
        augment (bool): Whether to apply data augmentation to the data before returning it.

        Returns:
        Tuple[Dict[str, np.array], Dict[str, np.array]]: A tuple containing the input data and output data,
            each represented as a dictionary mapping feature names to their respective data arrays.
        """
        x = {fname: f.get_batch(ix, transform, augment) for fname, f in self.x.items()}
        y = {
            fname: f.get_batch(ix, transform, augment=False)
            for fname, f in self.y.items()
        }
        return x, y

    def set_data(self, data: dict) -> None:
        """
        Set the data in the DataSet object.

        Parameters:
        data (dict): A dictionary containing the data to set, with keys corresponding to feature names
            and values corresponding to the data arrays.
        """
        if isinstance(data, tuple):
            data = data[0] | data[1]

        for key, value in data.items():
            if key in self.x.keys():
                self.x[key].set_data(value)
            elif key in self.y.keys():
                self.y[key].set_data(value)
            else:
                raise ValueError(
                    "Key "
                    + str(key)
                    + " neither in x nor in y: "
                    + str(self.x.keys())
                    + " "
                    + str(self.y.keys())
                )

    def transform(self, data: Dict[str, np.array]) -> Dict[str, np.array]:
        """
        Transforms data of features according to their transformation functions.

        Parameters:
        data (Dict[str, np.array]): A dictionary of features to be transformed.

        Returns:
        Dict[str, np.array]: A dictionary of transformed features.
        """
        return {key: ({**self.x, **self.y})[key].transform(data) for key in data.keys()}

    def inverse_transform(self, data: Dict[str, np.array]) -> Dict[str, np.array]:
        """
        Inversely transforms data of features according to their transformation functions.

        Parameters:
        data (Dict[str, np.array]): A dictionary of features to be inversely transformed.

        Returns:
        Dict[str, np.array]: A dictionary of inversely transformed features.
        """
        return {
            key: ({**self.x, **self.y})[key].inverse_transform(data)
            for key in data.keys()
        }

    def augment(self, data: Dict[str, np.array]) -> Dict[str, np.array]:
        """
        Augments data of features according to their augmentation functions.

        Parameters:
        data (Dict[str, np.array]): A dictionary of features to be augmented.

        Returns:
        Dict[str, np.array]: A dictionary of augmented features.
        """
        return {
            key: ({**self.x, **self.y})[key].augment(value)
            for key, value in data.items()
        }

    def inspect(self, path: str = None):
        """
        Visualizes data of features with their respective visualization functions.

        Parameters:
        path (str): The directory to save the visualizations to. If None, the visualizations will be shown instead of saved.
        """
        for f in (self.x | self.y).values():
            f.inspect(axis=None, path=path)

    def get_objectives(self) -> Dict[str, torch.nn.Module]:
        """
        Returns the loss functions for each feature in the dataset.
        """
        return {fname: f.get_objective() for fname, f in {**self.x, **self.y}.items()}

    def get_transformers(self) -> Dict[str, DataTransformer]:
        """
        Returns the data transformers for each feature in the dataset.
        """
        return {fname: f.get_transformer() for fname, f in {**self.x, **self.y}.items()}

    def get_data_loaders(
        self,
        batch_size: int,
        shuffle: bool = True,
        val_split: float = 0.1,
        test_split: float = 0.2,
        collate_fn=rec_concat_dict,
        random_split=True,
        **kwargs,
    ) -> Tuple[DataLoader]:
        """
        Create data loaders for train, validation, and test sets.

        Parameters:
        batch_size (int): The number of samples per batch.
        shuffle (bool): Whether to shuffle the data (default: True).
        val_split (float): Proportion of the dataset to use for validation (default: 0.1).
        test_split (float): Proportion of the dataset to use for testing (default: 0.2).
        collate_fn (function): The function to use for collating the data (default: rec_concat_dict).
        kwargs (dict): Additional arguments to pass to the `DataLoader` constructor.

        Returns:
        Tuple[DataLoader]: Tuple of the created data loaders in the order (train, val, test).
        """
        assert (
            val_split >= 0 and test_split >= 0 and val_split + test_split < 1
        ), f"Enter valid values for val_split and test_split, must be leq to 0 and their sum smaller than 1: {val_split}, {test_split}"

        rnd_ixs = np.arange(0, self.get_size(), dtype=int)
        if random_split:
            np.random.shuffle(rnd_ixs)

        train_ixs = rnd_ixs[: int(self.get_size() * (1 - val_split - test_split))]
        val_ixs = rnd_ixs[
            len(train_ixs) : len(train_ixs) + int(self.get_size() * val_split)
        ]
        test_ixs = rnd_ixs[len(train_ixs) + len(val_ixs) : self.get_size()]

        ds_train = Subset(self, train_ixs)
        ds_val = Subset(self, val_ixs)
        ds_test = Subset(self, test_ixs)

        return (
            DataLoader(
                ds_train,
                batch_size=batch_size,
                drop_last=True,
                shuffle=shuffle,
                collate_fn=collate_fn,
                **kwargs,
            ),
            DataLoader(
                ds_val,
                batch_size=batch_size,
                drop_last=False,
                shuffle=False,
                collate_fn=collate_fn,
                **kwargs,
            ),
            DataLoader(
                ds_test,
                batch_size=batch_size,
                drop_last=False,
                shuffle=False,
                collate_fn=collate_fn,
                **kwargs,
            ),
        )

    def get_data_module(
        self,
        batch_size: int,
        shuffle: bool = True,
        val_split: float = 0.1,
        test_split: float = 0.2,
        collate_fn=rec_concat_dict,
        **kwargs,
    ) -> DataModule:
        """
        Creates a DataModule object from the current DataSet object.

        Parameters:
        batch_size (int): The size of the batches that the DataModule's DataLoader objects will return.
        shuffle (bool): Whether the training DataLoader should shuffle its data.
        val_split (float): The proportion of the dataset to be used for validation.
        test_split (float): The proportion of the dataset to be used for testing.
        collate_fn (callable): The collation function to be used by the DataLoader objects.

        Returns:
        DataModule: A DataModule object representing the DataSet.
        """
        train_loader, val_loader, test_loader = self.get_data_loaders(
            batch_size, shuffle, val_split, test_split, collate_fn, **kwargs
        )
        return DataModule(train_loader, val_loader, test_loader)

    def evaluate(
        self,
        predictions: Dict[str, np.array],
        targets: Dict[str, np.array],
        return_dict: bool = False,
        transform: bool = True,
    ) -> Union[Dict[str, float], Tuple[float, float]]:
        """
        Evaluate the performance of the model's predictions on the given targets.

        Parameters
        predictions: Dict[str, np.array]
            A dictionary mapping feature names to predicted data for those features.
        targets: Dict[str, np.array]
            A dictionary mapping feature names to target data for those features.
        return_dict: bool, optional
            If True, return a dictionary mapping feature names to evaluation scores for those features.
            If False, return a tuple of the sum of the x feature scores and the sum of the y feature scores.
            Default is False.
        transform: bool, optional
            If True, apply any data transformations to the targets and predictions before evaluating.
            If False, assume the data is already transformed.
            Default is True.

        Returns
        Union[Dict[str, float], Tuple[float, float]]
            A dictionary mapping feature names to evaluation scores for those features,
            or a tuple of the sum of the x feature scores and the sum of the y feature scores.
        """
        if transform:
            targets = {k: v.astype(float) for k, v in self.transform(targets).items()}
            predictions = {
                k: v.astype(float) for k, v in self.transform(predictions).items()
            }

        x_loss = {}
        y_loss = {}
        for k in predictions.keys():
            if k in list(self.x.keys()):
                x_loss[k] = (
                    self.x[k]
                    .get_objective()(
                        torch.as_tensor(targets[k]), torch.as_tensor(predictions[k])
                    )
                    .numpy()
                )
            elif k in list(self.y.keys()):
                y_loss[k] = (
                    self.y[k]
                    .get_objective()(
                        torch.as_tensor(targets[k]), torch.as_tensor(predictions[k])
                    )
                    .numpy()
                )
        if return_dict:
            return {"x": x_loss, "y": y_loss}
        else:
            return sum(list(x_loss.values())), sum(list(y_loss.values()))

    def __len__(self) -> int:
        """
        Return the length of the dataset (number of samples).
        """
        return self.get_size()

    def __getitem__(self, ix) -> Dict[str, np.array]:
        """
        Return the sample at index `ix` as a dictionary of numpy arrays.
        The keys in the dictionary represent the names of the features in the sample.
        """
        return {fname: f.data[ix : ix + 1] for fname, f in (self.x | self.y).items()}


class MetaDataSet(DataSet):
    """
    MetaDataSet is a subclass of DataSet that does not contain data.
    It is intended to store metadata such as the shape, data type, and normalization parameters of the data.

    MetaDataSet can be converted to a DataSet object by calling the set_data method.
    """

    def set_data(self, data: dict) -> None:
        """
        Convert the MetaDataSet object to a DataSet object by adding data.

        Parameters:
        data (dict): A dictionary containing data for the features in the MetaDataSet object.
                     The keys should match the names of the features in the MetaDataSet object.
                     The values should be numpy arrays with the same shape as the corresponding feature in the MetaDataSet object.

        Returns:
        DataSet: A DataSet object with the same metadata as the MetaDataSet object, but now with data.
        """
        print("Converting MetaDataSet to DataSet")
        ds = DataSet().from_features_list(self.x.values(), self.y.values())
        ds.set_data(data)
        return ds

    def get_batch(self, ix: np.array, transform: bool = False, augment: bool = False):
        raise TypeError(
            'MetaDataSets do not contain data and "get_batch" can therefore not be called.'
            + 'Consider converting to "DataSet" by calling "set_data".'
        )

    def get_size(self) -> int:
        raise TypeError(
            'MetaDataSets do not contain data and "get_size" can therefore not be called.'
            + 'Consider converting to "DataSet" by calling "set_data".'
        )

    def get_data_loaders(
        self,
        batch_size: int,
        shuffle: bool = True,
        val_split: float = 0.1,
        test_split: float = 0.2,
        collate_fn=rec_concat_dict,
        **kwargs,
    ) -> Tuple[DataLoader]:
        raise TypeError(
            'MetaDataSets do not contain data and "get_data_loaders" can therefore not be called.'
            + 'Consider converting to "DataSet" by calling "set_data".'
        )

    def __len__(self) -> int:
        raise TypeError(
            'MetaDataSets do not contain data and "__len__" can therefore not be called.'
            + 'Consider converting to "DataSet" by calling "set_data".'
        )

    def __getitem__(self, ix) -> dict:
        raise TypeError(
            'MetaDataSets do not contain data and "__getitem__" can therefore not be called.'
            + 'Consider converting to "DataSet" by calling "set_data".'
        )

    def save(self, path: str):
        """
        Save the dataset (without data) to a given path.

        Parameters:
        path (str): The path to save the dataset to.
        """
        return super().save(path, include_data=False)


class MetaData:
    """
    Helper class that stores metadata about a DataSet object.
    """

    @staticmethod
    def from_file(path: str):
        """
        Creates a MetaData object from a pickle file stored at the given path.
        This is useful for loading metadata about a DataSet object that has already been saved.
        """
        with open(os.path.join(path, "meta_data.pkl"), "rb") as f:
            config = pickle.load(f)
        return MetaData(config)

    @staticmethod
    def from_data_set(data_set: DataSet):
        """
        Creates a MetaData object from a DataSet object.
        It stores information about the features of the DataSet, including their types and configurations.
        """
        config = {
            "x": [
                (feature.type, feature.get_config()) for feature in data_set.x.values()
            ],
            "y": [
                (feature.type, feature.get_config()) for feature in data_set.y.values()
            ],
            "name": data_set.name,
        }
        return MetaData(config)

    def __init__(self, config: dict):
        """
        Parameters
        config (dict): a dictionary containing the metadata.
        """
        self.config = config

    def save(self, path: str):
        """
        Saves the metadata to a pickle file at the specified path.
        """
        self.config["path"] = path
        with open(os.path.join(path, "meta_data.pkl"), "wb") as f:
            pickle.dump(self.config, f, pickle.HIGHEST_PROTOCOL)


class MultiTaskDataSet:
    @classmethod
    def from_path(cls, path: str) -> "MultiTaskDataSet":
        ds_paths = [ds_path for ds_path in os.listdir(path) if "task_" in ds_path]
        return cls.from_meta_data([MetaData.from_file(path) for path in ds_paths])

    @classmethod
    def from_meta_data(cls, meta_data: List[MetaData], name: str) -> "MultiTaskDataSet":
        return cls([DataSet.from_meta_data(md) for md in meta_data], name)

    def __init__(self, datasets: List[DataSet]):
        self.datasets = datasets

        self.x, self.y = {}, {}
        dataset_names = []
        for dataset in datasets:
            self.x.update(dataset.x)
            self.y.update(dataset.y)
            if dataset.name in dataset_names:
                raise ValueError(
                    f"There are two datasets with the same name {dataset.name}. Names must be unique. Change one of the names or remove it."
                )
            dataset_names.append(dataset.name)

        fnames = set()
        for dataset in datasets:
            fnames.update({fname for fname in list((dataset.x | dataset.y).keys())})

        for fname in fnames:
            fdata = [
                (dataset.x | dataset.y)[fname].data
                for dataset in datasets
                if (fname in dataset.x or fname in dataset.y)
                and ((dataset.x | dataset.y)[fname].data is not None)
            ]
            if len(fdata) > 1:
                print(
                    f"WARNING: Feature {fname} appears in {len(fdata)} datasets. The data from {fname} in all datasets will be \
                    combined and the DataTransformers updated. Note that only the DataTransformers are modified. The data will remain unmodified."
                )
                fdata = np.concatenate(fdata, axis=0)
                for dataset in datasets:
                    (dataset.x | dataset.y)[fname]._init_transformer(fdata)

    def get_data_loaders(
        self,
        batch_size: int,
        shuffle: bool = True,
        val_split: float = 0.1,
        test_split: float = 0.2,
        collate_fn=rec_concat_dict,
        **kwargs,
    ) -> Tuple[DataLoader]:
        train_loaders, val_loaders, test_loaders = [], [], []
        for dataset in self.datasets:
            train, val, test = dataset.get_data_loaders(
                batch_size, shuffle, val_split, test_split, collate_fn
            )
            train_loaders.append(train)
            val_loaders.append(val)
            test_loaders.append(test)

        return (
            CombinedLoader(train_loaders),
            CombinedLoader(val_loaders),
            CombinedLoader(test_loaders),
        )

    def save(self, path: str, include_data: bool = True):
        for dataset in self.datasets:
            d_path = make_path_unique(os.path.join(path, dataset.name))
            dataset.save(d_path, include_data)
