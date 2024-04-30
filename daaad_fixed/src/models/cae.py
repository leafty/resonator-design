import os
import torch
import pickle
import warnings
import numpy as np
import pytorch_lightning as pl

from sklearn_som.som import SOM
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Union

from models.encoders import Encoder, VEncoder
from models.decoders import Decoder
from models.finetune_cae import FineTuningModel
from dataset.data_set import DataSet
from dataset.data_module import DataModule
from utils import (
    batch_call,
    numpy_dict_to_tensor,
    rec_concat_dict,
    swarm_plot,
    torch_dict_to_numpy,
)

warnings.filterwarnings("ignore", ".*does not have many workers.*")


class CondAEModel(pl.LightningModule):
    """
    Class representing a Conditional Autoencoder model.
    """

    def __init__(
        self,
        dataset: DataSet,
        layer_widths: List[int],
        latent_dim: int,
        x_heads_layer_widths: Dict[str, List[int]] = {},
        y_heads_layer_widths: Dict[str, List[int]] = {},
        loss_weights: Dict[str, float] = None,
        activation: Union[torch.nn.Module, str] = "leaky_relu",
        optimizer: torch.optim.Optimizer = None,
        pass_y_to_encoder: bool = False,
        name: str = "CondAEModel",
        **kwargs
    ):
        """
        Parameters:
        dataset (DataSet): Object representing the input and output data for the model.
        layer_widths (List[int]): List of integers specifying the number of units in each hidden layer of the autoencoder's
            encoder and decoder (i.e., the "core" of the autoencoder). The first element of the list corresponds to the
            number of units in the first hidden layer of the encoder, the last element corresponds to the number of units
            in the last hidden layer of the decoder, and the elements in between correspond to the number of units in each
            hidden layer of the autoencoder in the order they appear (encoder followed by decoder).
        latent_dim (int): Integer specifying the number of units in the latent (i.e., encoded) representation of the data.
        x_heads_layer_widths (Dict[str, List[int]], optional): Dictionary specifying the number of units in the "head" layers
            that are prepended to the autoencoder's encoder and appended to the autoencoder's decoder. The keys of the dictionary
            are the names of the features, the values are a sequence of integers specifying the number of units in each hidden layer of the head.
        y_heads_layer_widths (Dict[str, List[int]], optional): Dictionary specifying the number of units in the "head" layers
            that are appended to the autoencoder's encoder and prepended to the autoencoder's decoder. The keys of the dictionary
            are the names of the features, the values are a sequence of integers specifying the number of units in each hidden layer of the head.
        loss_weights (Dict[str, int]), optional: Dictionary containing the weights with which each loss term should be multiplied before being
            added to the total loss used for backpropagation. Defaults to {'x': 1., 'y': 1., 'decorrelation': 0.}.
        activation (Union[torch.nn.Module, str], optional): Activation function to be used in the latent layers of the autoencoder. Defaults to leaky ReLU.
        optimizer (torch.optim.Optimizer), optional: Optimizer to be used for updating the model's weights. Defaults to Adam.
        pass_y_to_encoder (bool), optional: Whether to pass the conditional features y to the autoencoder's encoder (vanilla cVAE formulation) or not.
            In the first case, the encoder maps from x to z and is solely used for finding the latent vector needed to reconstruct x given y.
            In the latter case, the encoder represents a surrogate model mapping from x to y as well as a latent vector z. Defaults to False.
        name (str): Name of the model.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        if len(layer_widths) == 0:
            raise ValueError(
                "The core of the autoencoder must have at least one layer, i.e. `layer_widths` cannot be empty."
            )

        # Dictionary mapping from input feature names to tuples, where the first element in the tuple is the encoding head to be prepended to the encoder,
        # and the second element is the decoding head to be appended to the decoder.
        self.x_heads = {
            x_key: x_value.get_heads(
                x_heads_layer_widths.get(x_key, []),
                layer_widths[0],
                activation,
                **kwargs,
            )
            for x_key, x_value in dataset.x.items()
        }

        # Dictionary mapping from output feature names to tuples, where the first element in the tuple is the encoding head to be prepended to the decoder,
        # and the second element is the decoding head to be appended to the encoder.
        self.y_heads = {
            y_key: y_value.get_heads(
                y_heads_layer_widths.get(y_key, []),
                layer_widths[-1],
                activation,
                **kwargs,
            )
            for y_key, y_value in dataset.y.items()
        }

        # Build the encoder based on the above head dictionaries. If `pass_y_to_encoder`, the conditional features y are also passed as inputs to the encoder.
        # In this case, the encoder is not tasked with predicting the conditional features y. Otherwise, the encoder is a surrogate model predicting y and z.
        self.encoder = Encoder(
            {
                x_key: x_heads[0]
                for x_key, x_heads in (
                    self.x_heads.items()
                    if not pass_y_to_encoder
                    else (self.x_heads | self.y_heads).items()
                )
            },
            {
                y_key: y_heads[1]
                for y_key, y_heads in self.y_heads.items()
                if not pass_y_to_encoder
            }
            if not pass_y_to_encoder
            else {},
            layer_widths,
            latent_dim,
            activation,
        )

        # Build the decoder based on the above head dictionaries.
        self.decoder = Decoder(
            {y_key: y_heads[0] for y_key, y_heads in self.y_heads.items()},
            {x_key: x_heads[1] for x_key, x_heads in self.x_heads.items()},
            layer_widths[::-1],
            latent_dim,
            activation,
        )

        self.dataset = dataset
        self.name = name
        self.layer_widths = layer_widths
        self.latent_dim = latent_dim
        self.x_heads_layer_widths = x_heads_layer_widths
        self.y_heads_layer_widths = y_heads_layer_widths
        self.loss_weights = loss_weights if loss_weights else {}
        self.feature_losses = dataset.get_objectives()
        self.activation = activation
        self.optimizer = optimizer
        self.pass_y_to_encoder = pass_y_to_encoder
        self.model_trainer = None

    def get_config(self):
        return {
            "layer_widths": self.layer_widths,
            "latent_dim": self.latent_dim,
            "x_heads_layer_widths": self.x_heads_layer_widths,
            "y_heads_layer_widths": self.y_heads_layer_widths,
            "loss_weights": self.loss_weights,
            "activation": self.activation,
            "optimizer": self.optimizer,
            "pass_y_to_encoder": self.pass_y_to_encoder,
        }

    def configure_optimizers(self):
        """
        Configure the optimizers for the model.

        Returns:
            dict: A dictionary containing the optimizer(s) and learning rate scheduler(s) to be used during training.
        """
        # Initialize the optimizer with Adam, using the model parameters as the input arguments
        optimizer = (
            self.optimizer
            if self.optimizer is not None
            else torch.optim.Adam(self.parameters())
        )

        # Initialize the learning rate scheduler with ReduceLROnPlateau
        # This scheduler reduces the learning rate when the validation loss stops improving
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=6, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
            },
        }

    def _step(
        self, batch: Dict[str, np.array], batch_idx: int, mode: str
    ) -> Dict[str, float]:
        """
        Process a single batch of data.

        Args:
            batch (Dict[str, np.array]): A dictionary containing the data for a single batch.
            batch_idx (int): The index of the current batch.
            mode (str): The mode in which the model is being run (either 'train' or 'val').

        Returns:
            Dict[str, float]: A dictionary containing the losses for the batch.
        """
        # Select the relevant data from the batch and apply any necessary transformations
        data = self.dataset.transform(
            {
                d_key: batch[d_key]
                for d_key in list(self.encoder.in_heads.keys())
                + list(self.decoder.in_heads.keys())
            }
        )

        # Augment the data if we are in training mode
        if mode == "train":
            data = self.dataset.augment(data)

        # Convert the data to tensors and move to the correct device
        data = numpy_dict_to_tensor(data, device=self.device)

        pred = self(data, transform=False)

        x_losses = {
            key: self.loss_weights.get("x", 1.0)
            * self.feature_losses[key](pred["x"][key], data[key].float())
            for key in self.decoder.out_heads.keys()
        }
        x_loss = torch.stack(list(x_losses.values()), dim=0).sum()
        y_losses = {
            key: self.loss_weights.get("y", 1.0)
            * self.feature_losses[key](pred["y"][key], data[key].float())
            for key in self.encoder.out_heads.keys()
        }
        y_loss = (
            torch.stack(list(y_losses.values()), dim=0).sum()
            if len(y_losses) > 0
            else 0.0
        )

        # calculate only if decorrelation weight is > 0 to avoid computing gradients for nothing
        if (
            self.loss_weights.get("decorrelation", 0) > 0
            and len(self.encoder.out_heads) > 0
        ):
            # decorrelate by reducing the covariance: https://arxiv.org/abs/1904.01277v1
            decorrelation_loss = (
                self.loss_weights["decorrelation"]
                * (
                    (
                        torch.permute(
                            torch.cat(
                                [
                                    torch.Tensor(
                                        data[k]
                                        - torch.unsqueeze(
                                            data[k].mean(
                                                axis=tuple(range(1, len(data[k].shape)))
                                            ),
                                            dim=1,
                                        )
                                    )
                                    for k in self.encoder.out_heads.keys()
                                ],
                                dim=-1,
                            ),
                            dims=(1, 0),
                        ).float()
                        @ pred["z"].float()
                    )
                    ** 2
                ).mean()
            )
        else:
            # weight is zero, so decorrelation_loss is detached from the graph
            decorrelation_loss = 0.0

        total_loss = x_loss + y_loss + decorrelation_loss

        loss_dict = (
            {
                mode + "_loss": total_loss,
                mode + "_features_loss": x_loss + y_loss,
            }
            | {mode + "_" + key + "_loss": value for key, value in x_losses.items()}
            | {mode + "_" + key + "_loss": value for key, value in y_losses.items()}
        )
        if self.loss_weights.get("decorrelation", 0):
            loss_dict[mode + "_decorrelation_loss"] = decorrelation_loss
        return pred, loss_dict

    def forward(
        self, data: Dict[str, Union[np.array, torch.Tensor]], transform: bool = True
    ) -> Union[Dict[str, np.array], Dict[str, torch.Tensor]]:
        if transform and not torch.is_tensor(data[list(data.keys())[0]]):
            data = self.dataset.transform(
                {
                    d_key: data[d_key]
                    for d_key in list(self.encoder.in_heads.keys())
                    + list(self.decoder.in_heads.keys())
                }
            )
            torch.set_grad_enabled(False)
            self.eval()

        x = numpy_dict_to_tensor(
            {x_key: data[x_key] for x_key in self.x_heads.keys()}, device=self.device
        )
        y = numpy_dict_to_tensor(
            {y_key: data[y_key] for y_key in self.y_heads.keys()}, device=self.device
        )

        if self.pass_y_to_encoder:
            pred = self.encoder(x | y)
        else:
            pred = self.encoder(x)

        pred.update(self.decoder({"z": pred["z"], "y": y}))

        if transform and not torch.is_tensor(data[list(data.keys())[0]]):
            pred = torch_dict_to_numpy(pred)
            pred["x"] = self.dataset.inverse_transform(pred["x"])
            pred["y"] = self.dataset.inverse_transform(pred["y"])
            torch.set_grad_enabled(True)
            self.train()
        return pred

    def training_step(self, batch: Dict[str, np.array], batch_idx: int) -> float:
        """
        Perform a single training step.

        Args:
            batch (Dict[str, np.array]): A dictionary containing the data for a single batch.
            batch_idx (int): The index of the current batch.

        Returns:
            float: The training loss.
        """
        # Process the batch and retrieve the loss dictionary
        _, loss_dict = self._step(batch, batch_idx, mode="train")

        # Log the loss values to various outputs
        self.log_dict(
            loss_dict,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            batch_size=len(batch[list(batch.keys())[0]]),
        )
        return loss_dict["train_loss"]

    def validation_step(self, batch: Dict[str, np.array], batch_idx: int) -> float:
        """
        Perform a single validation step.

        Args:
            batch (Dict[str, np.array]): A dictionary containing the data for a single batch.
            batch_idx (int): The index of the current batch.

        Returns:
            float: The validation loss.
        """
        # Process the batch and retrieve the loss dictionary
        _, loss_dict = self._step(batch, batch_idx, mode="val")

        # Log the loss values to various outputs
        self.log_dict(
            loss_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch[list(batch.keys())[0]]),
        )
        return loss_dict["val_loss"]

    def test_step(self, batch: Dict[str, np.array], batch_idx: int) -> float:
        """
        Perform a single test step.

        Args:
            batch (Dict[str, np.array]): A dictionary containing the data for a single batch.
            batch_idx (int): The index of the current batch.

        Returns:
            float: The test loss.
        """
        # Process the batch and retrieve the loss dictionary
        _, loss_dict = self._step(batch, batch_idx, mode="test")

        # Log the loss values to various outputs
        self.log_dict(
            loss_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch[list(batch.keys())[0]]),
        )
        return loss_dict["test_loss"]

    def encode(
        self,
        data: Union[DataLoader, Dict[str, np.array]],
        transform: Union[bool, Tuple[bool, bool]] = True,
        batch_size=1,
        return_lat: bool = False,
    ) -> Union[Dict[str, np.array], Dict[str, torch.Tensor]]:
        """
        Encode the input data into a latent representation.

        Args:
            data (Union[DataLoader, Dict[str, np.array]]): The input data, either as a data loader or a dictionary of tensors.
            transform (Union[bool, Tuple[bool, bool]], optional): If boolean and True, the input data will be transformed.
                If boolean and False, the input data is assumed to be in the original data space.
                If Tuple and first element is True, data is transformed.
                If Tuple and second element is True, predictions will be returned to original data space. Default: True.
            batch_size (int, optional): If data is a dictionary, this sets the size of the batches passed to the model. Defaults to 1.

        Returns:
            Union[Dict[str, np.array], Dict[str, torch.Tensor]]: A dictionary containing the latent representation of the input data.
        """
        if isinstance(transform, bool):
            transform = (transform, transform)

        if transform[0]:
            self.eval()
            torch.set_grad_enabled(False)

        if isinstance(data, DataLoader):
            pred = []
            for batch in data:
                if transform[0]:
                    batch = self.dataset.transform(batch)
                pred.append(
                    self.encoder(
                        numpy_dict_to_tensor(batch, device=self.device), return_lat
                    )
                )
            pred = rec_concat_dict([batch for batch in pred])
        else:
            if transform[0]:
                data = self.dataset.transform(data)
            pred = batch_call(
                data,
                self.encoder,
                batch_size=batch_size,
                device=self.device,
                return_lat=return_lat,
            )

        if transform[1]:
            pred = torch_dict_to_numpy(pred)
            pred["y"] = self.dataset.inverse_transform(pred["y"])
            torch.set_grad_enabled(True)
            self.train()
        return pred

    def decode(
        self,
        data: Union[DataLoader, Dict[str, np.array]],
        z: np.array = None,
        transform: Union[bool, Tuple[bool, bool]] = True,
        batch_size: int = 1,
        return_lat: bool = False,
    ) -> Union[Dict[str, np.array], Dict[str, torch.Tensor]]:
        """
        Decode the latent representation into the original data space.

        Args:
            data (Union[DataLoader, Dict[str, np.array]]): The conditional data, either as a data loader or a dictionary of tensors.
            z (np.array): The latent representation to decode.
            transform (Union[bool, Tuple[bool, bool]], optional): If boolean and True, the input data will be transformed.
                If boolean and False, the input data is assumed to be in the original data space.
                If Tuple and first element is True, data is transformed.
                If Tuple and second element is True, predictions will be returned to original data space. Default: True.
            batch_size (int, optional): If data is a dictionary, this sets the size of the batches passed to the model. Defaults to 1.

        Returns:
            Union[Dict[str, np.array], Dict[str, torch.Tensor]]: A dictionary containing the decoded data.
        """
        if isinstance(transform, bool):
            transform = (transform, transform)

        if z is None:
            if isinstance(data, dict) and "z" in data:
                z = data["z"]
            else:
                raise ValueError(
                    "The latent embedding `z` must either be passed to the function or be included in `data`."
                )

        if transform[0]:
            self.eval()
            torch.set_grad_enabled(False)

        if isinstance(data, DataLoader):
            pred = []
            for i, batch in enumerate(data):
                if transform[0]:
                    batch = self.dataset.transform(batch)
                pred.append(
                    self.decoder(
                        numpy_dict_to_tensor(
                            {
                                "z": z[i * data.batch_size : (i + 1) * data.batch_size],
                                "y": batch,
                            },
                            device=self.device,
                        ),
                        return_lat,
                    )
                )
            pred = rec_concat_dict([batch for batch in pred])
        else:
            if transform[0]:
                data = self.dataset.transform(data)
            pred = batch_call(
                {"z": z, "y": data},
                self.decoder,
                batch_size=batch_size,
                device=self.device,
                return_lat=return_lat,
            )

        if transform[1]:
            pred["x"] = self.dataset.inverse_transform(torch_dict_to_numpy(pred["x"]))
            torch.set_grad_enabled(True)
            self.train()

        return pred

    def fit(
        self,
        data_module: DataModule = None,
        train_loader: DataLoader = None,
        val_loader: DataLoader = None,
        max_epochs: int = 100,
        callbacks: list = None,
        loggers: list = None,
        accelerator: str = "auto",
        **kwargs
    ):
        """
        Train the model.

        Args:
            data_module (DataModule, optional): DataModule object that provides the training, validation, and test data.
                If not provided, train_loader and val_loader must be provided. Default: None.
            train_loader (DataLoader, optional): A PyTorch DataLoader object that provides the training data.
                If not provided, data_module must be provided. Default: None.
            val_loader (DataLoader, optional): A PyTorch DataLoader object that provides the validation data.
                If not provided, data_module must be provided. Default: None.
            max_epochs (int, optional): The maximum number of epochs to train for. Default: 100.
            callbacks (list, optional): A list of PyTorch Lightning Callback objects to use during training. Default: None.
            loggers (list, optional): A list of PyTorch Lightning Logger objects to use during training. Default: None.
            accelerator (str, optional): Which accelerator should be used (e.g. cpu, gpu, mps, etc.) Default: auto.
        """
        self.model_trainer = pl.Trainer(
            accelerator=accelerator,
            auto_select_gpus=True,
            max_epochs=max_epochs,
            callbacks=callbacks if callbacks else [],
            logger=loggers if loggers else [],
            enable_progress_bar=True,
            **kwargs,
        )

        if data_module is not None:
            self.model_trainer.fit(self, data_module)
        elif train_loader is not None and val_loader is not None:
            self.model_trainer.fit(
                self, train_dataloaders=train_loader, val_dataloaders=val_loader
            )
        else:
            raise ValueError(
                "Either data_module or train_loader and val_loader must be provided."
            )

    def two_stage_fine_tune(
        self,
        data_module: DataModule = None,
        train_loader: DataLoader = None,
        val_loader: DataLoader = None,
        max_epochs: int = 100,
        callbacks: list = None,
        loggers: list = None,
        accelerator: str = "auto",
        gen_z_strategy: str = "sample_around",
        **kwargs
    ):
        """
        Second training run is started by swapping the encoder and decoder
        and freezing the encoder. This way, the decoder is fine-tuned on respecting the conditionals y.

        Args:
            data_module (DataModule, optional): DataModule object that provides the training, validation, and test data.
                If not provided, train_loader and val_loader must be provided. Default: None.
            train_loader (DataLoader, optional): A PyTorch DataLoader object that provides the training data.
                If not provided, data_module must be provided. Default: None.
            val_loader (DataLoader, optional): A PyTorch DataLoader object that provides the validation data.
                If not provided, data_module must be provided. Default: None.
            max_epochs (int, optional): The maximum number of epochs to train for. Default: 100.
            callbacks (list, optional): A list of PyTorch Lightning Callback objects to use during training. Default: None.
            loggers (list, optional): A list of PyTorch Lightning Logger objects to use during training. Default: None.
            accelerator (str, optional): Which accelerator should be used (e.g. cpu, gpu, mps, etc.) Default: auto.
            gen_z_strategy (str): Used only if two_stage is true.
                Which strategy should be employed for optaining the latent vectors z.
                One of 'encode', 'sample' or 'sample_around'.
                -   if 'encode', the latent variables z are generated by the (frozen) encoder
                -   if 'sample', the latent variables z are sampled normally
                -   if 'sample_around', the latent variables z are generated by the (frozen) encoder and added to random gaussian noise with std `sample_around_std`.
        """
        finetune_model = FineTuningModel(self, gen_z_strategy)
        finetune_model.fit(
            data_module,
            train_loader,
            val_loader,
            max_epochs,
            callbacks,
            loggers,
            accelerator,
            **kwargs,
        )
        return

    def validate(
        self, val_loader: DataLoader, accelerator: str = "auto", **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate the model on the validation data.

        Args:
            val_loader (DataLoader): A PyTorch DataLoader object that provides the validation data.
            accelerator (str, optional): Which accelerator should be used (e.g. cpu, gpu, mps, etc.) Default: auto

        Returns:
            Dict[str, float]: A dictionary containing the validation loss and metrics.
        """
        # Create a trainer object if one does not already exist
        if self.model_trainer is None:
            self.model_trainer = pl.Trainer(accelerator=accelerator, **kwargs)

        res = self.trainer.validate(self, dataloaders=val_loader)
        return res

    def test(
        self, test_loader: DataLoader, accelerator: str = "auto", **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate the model on the test data.

        Args:
            test_loader (DataLoader): A PyTorch DataLoader object that provides the test data.
            accelerator (str, optional): Which accelerator should be used (e.g. cpu, gpu, mps, etc.) Default: auto

        Returns:
            Dict[str, float]: A dictionary containing the test loss and metrics.
        """
        # Create a trainer object if one does not already exist
        if self.model_trainer is None:
            self.model_trainer = pl.Trainer(accelerator=accelerator, **kwargs)

        return self.trainer.test(self, dataloaders=test_loader)

    def predict(
        self, data: Union[DataLoader, dict], accelerator: str = "auto"
    ) -> Dict[str, np.array]:
        """
        Make predictions using the model.

        Args:
            data (Union[DataLoader, dict]): A PyTorch DataLoader object to make predictions on,
                or a dictionary of tensors to make predictions on.
            accelerator (str, optional): Which accelerator should be used (e.g. cpu, gpu, mps, etc.) Default: auto

        Returns:
            dict: A dictionary of predictions.
        """
        if isinstance(data, DataLoader):
            # Create a trainer object if one does not already exist
            if self.model_trainer is None:
                self.model_trainer = pl.Trainer(accelerator=accelerator)

            preds = self.model_trainer.predict(self, dataloaders=data)
            return rec_concat_dict(preds)
        else:
            torch.set_grad_enabled(False)
            self.eval()

            preds = self(data)

            torch.set_grad_enabled(True)
            self.train()
            return preds

    def visual_evaluate(self, data, path: str = None):
        """
        Create plots to visualise the accuracy and performance of the model.

        Args:
            data (Union[DataLoader, dict]): A PyTorch DataLoader object or dictionary of tensors to evaluate the model on.
            path (str, optional): The directory to save the plots to. If not provided, the plots will be displayed instead of saved.
        """
        # Make predictions on the provided data
        pred = self.predict(data)

        if isinstance(data, DataLoader):
            # Convert the data from a DataLoader to a dictionary
            data = torch_dict_to_numpy(rec_concat_dict([batch for batch in data]))

        # Loop through the predicted values for each input feature (x)
        for x_key, x_data in pred["x"].items():
            feature = self.dataset.x[x_key]
            # Create a plot comparing the true values and predicted values
            feature.compare(
                {
                    "True values": data[x_key].astype(feature.data_type),
                    "Predicted values": x_data.astype(feature.data_type),
                },
                title=feature.name,
                axis=None,
                path=path,
                evaluation=True,
            )

        # Loop through the predicted values for each conditional feature (y)
        for y_key, y_data in pred["y"].items():
            feature = self.dataset.y[y_key]
            # Create a plot comparing the true values and predicted values
            feature.compare(
                {
                    "True values": data[y_key].astype(feature.data_type),
                    "Predicted values": y_data.astype(feature.data_type),
                },
                title=feature.name,
                axis=None,
                path=path,
                evaluation=True,
            )

    def inspect_latent(
        self,
        data,
        dim_reduction_method: str = None,
        dim_ix_0: int = 0,
        dim_ix_1: int = 1,
        path: str = None,
    ):
        z = self.encode(data)["z"]

        if isinstance(data, DataLoader):
            data = torch_dict_to_numpy(rec_concat_dict([batch for batch in data]))

        if z.shape[-1] > 2 and dim_reduction_method is not None:
            if dim_reduction_method == "pca":
                reducer = PCA(n_components=2)
            if dim_reduction_method == "tsne":
                reducer = TSNE()
            if dim_reduction_method == "som":
                reducer = SOM(m=2, n=1, dim=z.shape[-1])
            z = reducer.fit_transform(z)
        elif z.shape[-1] > 2 and dim_reduction_method is None:
            z = np.stack([z[:, dim_ix_0], z[:, dim_ix_1]], axis=1)

        for key, feature in (self.dataset.x | self.dataset.y).items():
            feature.inspect_latent(
                z[:, 0],
                z[:, 1],
                data[key].astype(feature.data_type),
                title=feature.name,
                path=path,
            )

    def __sensitivity_gradients_x(
        self, data: Dict[str, np.array], x_name: str
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Calculate the gradients of the output feature with respect to the input and latent features.

        Args:
            data (Dict[str, np.array]): Input data.
            x_name (str): Name of the output feature.

        Returns:
            tuple: Tuple containing the gradients of the output feature with respect to the input and latent features.
        """
        data = {
            key: torch.Tensor(data[key]).float().requires_grad_()
            for key in list(self.encoder.in_heads.keys())
            + list(self.decoder.in_heads.keys())
        }
        batch_size = data[list(data.keys())[0]].shape[0]
        self(data, transform=False)["x"][x_name].sum(axis=0).mean().backward()
        return torch_dict_to_numpy(
            {
                x_key: torch.reshape(data[x_key].grad, (batch_size, -1)).mean(dim=1)
                for x_key in self.encoder.in_heads
            }
        ), torch_dict_to_numpy(
            {
                y_key: torch.reshape(data[y_key].grad, (batch_size, -1)).mean(dim=1)
                for y_key in self.decoder.in_heads
            }
        )

    def __sensitivity_gradients_y(
        self, data: Dict[str, np.array], y_name: str
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Calculate the gradients of the latent feature with respect to the input features.

        Args:
            data (Dict[str, np.array]): Input data.
            y_name (str): Name of the latent feature.

        Returns:
            dict: Gradients of the latent feature with respect to the input features.
        """
        data = {
            x_key: torch.Tensor(data[x_key]).float().requires_grad_()
            for x_key in self.encoder.in_heads.keys()
        }
        batch_size = data[list(data.keys())[0]].shape[0]
        self.encoder(data)["y"][y_name].sum(axis=0).mean().backward()
        return torch_dict_to_numpy(
            {
                x_key: torch.reshape(data[x_key].grad, (batch_size, -1)).mean(dim=1)
                for x_key in self.encoder.in_heads
            }
        )

    def sensitivity_analysis(
        self,
        data: Union[DataLoader, Dict[str, np.array]],
        features: List[str] = None,
        path: str = None,
    ) -> None:
        """
        Creates swarm plots of the sensitivity analysis for the given data. The sensitivity of the output features is calculated
        with respect to the input features.

        Parameters:
        - data (dict or DataLoader): Dictionary containing the input and output data, or a PyTorch DataLoader that returns a
        dictionary containing the input and output data.
        - features (list): List of features for which to create the sensitivity plots. If None, sensitivity plots are created
        for all features.
        - path (str): Path to the directory where the plots will be saved. If None, the plots will not be saved.
        """
        self.eval()
        if isinstance(data, DataLoader):
            data = rec_concat_dict([batch for batch in data])
        data_transformed = numpy_dict_to_tensor(
            self.dataset.transform(
                {
                    key: data[key]
                    for key in list(self.encoder.in_heads.keys())
                    + list(self.decoder.in_heads.keys())
                }
            )
        )
        all_features = list(self.dataset.x.keys()) + list(self.dataset.y.keys())

        # If no specific feature is given, create sensitivities for all features
        if features is None:
            features = all_features

        std_sens = 0
        med_sens = 0
        num_std_sens = 0
        for feature in features:
            if feature in self.decoder.out_heads.keys():
                x_wrt_x, x_wrt_y = self.__sensitivity_gradients_x(
                    data_transformed, feature
                )
                swarm_plot(
                    features=x_wrt_x,
                    data_hue=data[feature].astype(self.dataset.x[feature].data_type)
                    if len(data[feature].shape) == 2
                    else data[feature].reshape(len(data[feature]), -1).mean(axis=-1),
                    title=feature,
                    y_label="Input features",
                    path=os.path.join(path, "dx") if path is not None else None,
                )
                std_sens += np.vstack([v for v in x_wrt_y.values()]).std()
                med_sens += np.median(np.abs(np.vstack([v for v in x_wrt_y.values()])))
                num_std_sens += 1
                swarm_plot(
                    features=x_wrt_y,
                    data_hue=data[feature].astype(self.dataset.x[feature].data_type)
                    if len(data[feature].shape) == 2
                    else data[feature].reshape(len(data[feature]), -1).mean(axis=-1),
                    title=feature,
                    y_label="Conditional features",
                    path=os.path.join(path, "dx") if path is not None else None,
                )
            elif feature in self.encoder.out_heads.keys():
                y_wrt_x = self.__sensitivity_gradients_y(data_transformed, feature)
                swarm_plot(
                    features=y_wrt_x,
                    data_hue=data[feature].astype(self.dataset.y[feature].data_type)
                    if len(data[feature].shape) == 2
                    else data[feature].reshape(len(data[feature]), -1).mean(axis=-1),
                    title=feature,
                    y_label="Input features",
                    path=os.path.join(path, "dy") if path is not None else None,
                )
        # print('sens:', std_sens / num_std_sens, med_sens / num_std_sens)
        self.train()

    def summary(self, max_depth: int = 1) -> None:
        """
        Prints a summary of the encoder and decoder, including the number of parameters, the layers,
        their names, and the dimensionality.

        Parameters:
        max_depth (int, optional): Maximum depth of modules to show. Use -1 to show all modules or 0 to show no summary. Defaults to 1.
        """
        # Register example input array such that Model_Summary can print data shapes
        self.example_input_array = (
            {
                fname: torch.zeros(1, *(self.dataset.x | self.dataset.y)[fname].shape)
                for fname in list(self.encoder.in_heads.keys())
                + list(self.decoder.in_heads.keys())
            },
        )
        print(pl.utilities.model_summary.ModelSummary(self, max_depth=max_depth))

    def save(self, path: str) -> None:
        """
        Save the model, dataset configuration, and model weights to the specified path.

        Parameters:
        path (str): The directory path where the model, dataset configuration, and weights should be saved.
        """
        self.dataset.save(os.path.join(path, "dataset"), include_data=False)
        torch.save(self.state_dict(), os.path.join(path, "weights.pt"))
        with open(os.path.join(path, "config.pkl"), "wb") as f:
            pickle.dump(self.get_config(), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_model(cls, path: str) -> "CondAEModel":
        """
        Load a model from the specified path.

        Parameters:
        path (str): The directory path where the model, dataset configuration, and weights are saved.

        Returns:
        The loaded model.
        """
        ds = DataSet.from_path(os.path.join(path, "dataset"))
        with open(os.path.join(path, "config.pkl"), "rb") as f:
            config = pickle.load(f)
        cae = cls(ds, **config)
        state_dict = torch.load(os.path.join(path, "weights.pt"))
        cae.load_state_dict(state_dict)
        return cae


class CondVAEModel(CondAEModel):
    """
    Class representing a Conditional Variational Autoencoder model.
    """

    def __init__(
        self,
        dataset: DataSet,
        layer_widths: List[int],
        latent_dim: int,
        x_heads_layer_widths: Dict[str, List[int]] = {},
        y_heads_layer_widths: Dict[str, List[int]] = {},
        loss_weights: Dict[str, float] = None,
        activation: Union[torch.nn.Module, str] = "leaky_relu",
        optimizer: torch.optim.Optimizer = None,
        pass_y_to_encoder: bool = False,
        name: str = "CondVAEModel",
        **kwargs
    ):
        """
        Parameters:
        dataset (DataSet): Object representing the input and output data for the model.
        layer_widths (List[int]): List of integers specifying the number of units in each hidden layer of the autoencoder's
            encoder and decoder (i.e., the "core" of the autoencoder). The first element of the list corresponds to the
            number of units in the first hidden layer of the encoder, the last element corresponds to the number of units
            in the last hidden layer of the decoder, and the elements in between correspond to the number of units in each
            hidden layer of the autoencoder in the order they appear (encoder followed by decoder).
        latent_dim (int): Integer specifying the number of units in the latent (i.e., encoded) representation of the data.
        x_heads_layer_widths (Dict[str, List[int]], optional): Dictionary specifying the number of units in the "head" layers
            that are prepended to the autoencoder's encoder and appended to the autoencoder's decoder. The keys of the dictionary
            are the names of the features, the values are a sequence of integers specifying the number of units in each hidden layer of the head.
        y_heads_layer_widths (Dict[str, List[int]], optional): Dictionary specifying the number of units in the "head" layers
            that are appended to the autoencoder's encoder and prepended to the autoencoder's decoder. The keys of the dictionary
            are the names of the features, the values are a sequence of integers specifying the number of units in each hidden layer of the head.
        loss_weights (Dict[str, int]): Dictionary containing the weights with which each loss term should be multiplied before being
            added to the total loss used for backpropagation.
        activation (Union[torch.nn.Module, str]): Activation function to be used in the latent layers of the autoencoder.
        optimizer (torch.optim.Optimizer): Optimizer to be used for updating the model's weights.
        pass_y_to_encoder (bool): Whether to pass the conditional features y to the autoencoder's encoder (vanilla cVAE formulation) or not.
            In the first case, the encoder maps from x to z and is solely used for finding the latent vector needed to reconstruct x given y.
            In the latter case, the encoder represents a surrogate model mapping from x to y as well as a latent vector z.
        name (str): Name of the model.
        """
        super().__init__(
            dataset,
            layer_widths,
            latent_dim,
            x_heads_layer_widths,
            y_heads_layer_widths,
            loss_weights,
            activation,
            optimizer,
            pass_y_to_encoder,
            name,
            **kwargs,
        )

        self.encoder = VEncoder(
            {
                x_key: x_heads[0]
                for x_key, x_heads in (
                    self.x_heads.items()
                    if not pass_y_to_encoder
                    else (self.x_heads | self.y_heads).items()
                )
            },
            {
                y_key: y_heads[1]
                for y_key, y_heads in self.y_heads.items()
                if not pass_y_to_encoder
            }
            if not pass_y_to_encoder
            else {},
            layer_widths,
            latent_dim,
            activation,
        )

    def _step(self, batch: Dict[str, np.array], batch_idx: int, mode: str):
        """
        Process a single batch of data.
        Extends the parent's `_step` function by adding a kl_loss.

        Args:
            batch (Dict[str, np.array]): A dict of tensors containing the data for a single batch.
            batch_idx (int): The index of the current batch.
            mode (str): The mode in which the model is being run (either 'train' or 'val').

        Returns:
            Dict[str, float]: A dictionary containing the losses for the batch.
        """
        pred, losses = super()._step(batch, batch_idx, mode)

        z_mean, z_log_var = pred["z_mean"], pred["z_log_var"]
        kl_loss = (
            self.loss_weights.get("kl", 1.0)
            * (
                (-0.5 * (1 + z_log_var - z_mean**2 - torch.exp(z_log_var))).sum(dim=1)
            ).mean()
        )

        if mode == "train":
            losses["train_loss"] += kl_loss

        losses[mode + "_kl_loss"] = kl_loss
        return pred, losses

    def decode(
        self,
        data: Union[DataLoader, Dict[str, np.array]],
        z: np.array = None,
        z_std: float = 1,
        transform: bool = True,
        batch_size: int = 1,
        return_lat: bool = False,
    ):
        """
        Decode the latent representation into the original data space.

        Args:
            data (Union[DataLoader, Dict[str, np.array]]): The conditional data, either as a data loader or a dictionary of tensors.
            z (np.array): The latent representation to decode.
            z_std (float, optional): Used only if z is None. Defines the standard deviation
                of the Gaussian with which z will be sampled. Conrols diversity vs. reliability.
            transform (Union[bool, Tuple[bool, bool]], optional): If boolean and True, the input data will be transformed.
                If boolean and False, the input data is assumed to be in the original data space.
                If Tuple and first element is True, data is transformed.
                If Tuple and second element is True, predictions will be returned to original data space. Default: True.
            batch_size (int, optional): If data is a dictionary, this sets the size of the batches passed to the model. Defaults to 1.

        Returns:
            Union[Dict[str, np.array], Dict[str, torch.Tensor]]: A dictionary containing the decoded data.
        """
        if z is None:
            if isinstance(data, DataLoader):
                num_samples = len(data.dataset)
            else:
                num_samples = len(list(data.values())[0])
            z = np.random.normal(0, z_std, size=(num_samples, self.latent_dim))
        return super().decode(data, z, transform, batch_size, return_lat)
