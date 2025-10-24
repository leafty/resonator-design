import os
import torch
import pickle
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Union
from pytorch_lightning.trainer.supporters import CombinedLoader

from models.decoders import Decoder
from models.encoders import Encoder, VEncoder
from dataset.data_set import MultiTaskDataSet
from models.cae import CondAEModel, CondVAEModel
from utils import numpy_dict_to_tensor, rec_concat_dict, sum_join_dicts, swarm_plot, torch_dict_to_numpy


class MultiTaskCondAEModel(pl.LightningModule):
    """
        Parameters:
        dataset (MultiTaskDataSet): Object containing the datasets corresponding to each task.
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
    def __init__(self, dataset:MultiTaskDataSet, layer_widths:List[int], latent_dim:int, x_heads_layer_widths:Dict[str, List[int]]={}, y_heads_layer_widths:Dict[str, List[int]]={}, 
                        loss_weights:Dict[str, float]=None, activation:Union[torch.nn.Module, str]='leaky_relu', optimizer:torch.optim.Optimizer=None, pass_y_to_encoder:bool=False, 
                        name:str='CondAEModel', **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # Dictionary mapping from input feature names to tuples, where the first element in the tuple is the encoding head to be prepended to the encoder,
        # and the second element is the decoding head to be appended to the decoder.
        self.x_heads = {x_key: x_value.get_heads(x_heads_layer_widths.get(x_key, []), layer_widths[0], activation) \
            for x_key, x_value in dataset.x.items()}

        # Dictionary mapping from output feature names to tuples, where the first element in the tuple is the encoding head to be prepended to the decoder,
        # and the second element is the decoding head to be appended to the encoder.
        self.y_heads = {y_key: y_value.get_heads(y_heads_layer_widths.get(y_key, []), layer_widths[-1], activation) \
            for y_key, y_value in dataset.y.items()}

        # Build the encoder based on the above head dictionaries. If `pass_y_to_encoder`, the conditional features y are also passed as inputs to the encoder.
        # In this case, the encoder is not tasked with predicting the conditional features y. Otherwise, the encoder is a surrogate model predicting y and z.
        self.encoder = Encoder({x_key: x_heads[0] for x_key, x_heads in (self.x_heads.items() if not pass_y_to_encoder else (self.x_heads | self.y_heads).items())}, 
                               {y_key: y_heads[1] for y_key, y_heads in self.y_heads.items() if not pass_y_to_encoder} if not pass_y_to_encoder else {}, 
                               layer_widths, latent_dim, activation)
            
        # Build the decoder based on the above head dictionaries.
        self.decoder = Decoder({y_key: y_heads[0] for y_key, y_heads in self.y_heads.items()}, 
                               {x_key: x_heads[1] for x_key, x_heads in self.x_heads.items()}, 
                               layer_widths[::-1], latent_dim, activation)
        
        self.models = nn.ModuleDict({ds.name: CondAEModel(ds, layer_widths, latent_dim, pass_y_to_encoder=pass_y_to_encoder, name=name + '_model') for ds in dataset.datasets})
        self._share_weights_across_models()

        self.name = name
        self.dataset = dataset
        self.x_heads_layer_widths = x_heads_layer_widths
        self.y_heads_layer_widths = y_heads_layer_widths
        self.layer_widths = layer_widths
        self.latent_dim = latent_dim
        self.loss_weights = loss_weights if loss_weights else {}
        self.activation = activation
        self.optimizer = optimizer
        self.pass_y_to_encoder = pass_y_to_encoder
        self.model_trainer = None


    def _share_weights_across_models(self):
        # replace shared heads and core
        for model in self.models.values():
            model.x_heads = {x_key: self.x_heads[x_key] for x_key in model.x_heads.keys()}
            model.y_heads = {y_key: self.y_heads[y_key] for y_key in model.y_heads.keys()}

            # share encoder weights
            model.encoder.in_heads = nn.ModuleDict({in_h_name: self.encoder.in_heads[in_h_name] for in_h_name in model.encoder.in_heads.keys()})
            model.encoder.out_heads = nn.ModuleDict({out_h_name: self.encoder.out_heads[out_h_name] for out_h_name in model.encoder.out_heads.keys()})
            model.encoder.blocks = [model.encoder.blocks[0]] + list(self.encoder.blocks[1:])
            model.encoder.seq_blocks = torch.nn.Sequential(*model.encoder.blocks)
            model.encoder.fc_z = self.encoder.fc_z
            model.encoder.set_custom_modules(self.encoder.get_custom_modules())

            # share decoder weights
            model.decoder.in_heads = nn.ModuleDict({in_h_name: self.decoder.in_heads[in_h_name] for in_h_name in model.decoder.in_heads.keys()})
            model.decoder.out_heads = nn.ModuleDict({out_h_name: self.decoder.out_heads[out_h_name] for out_h_name in model.decoder.out_heads.keys()})
            model.decoder.blocks = [model.decoder.blocks[0]] + list(self.decoder.blocks[1:])
            model.decoder.seq_blocks = torch.nn.Sequential(*model.decoder.blocks)
            model.decoder.set_custom_modules(self.decoder.get_custom_modules())


    def get_config(self):
        return {
            'layer_widths': self.layer_widths,
            'latent_dim': self.latent_dim,
            'x_heads_layer_widths': self.x_heads_layer_widths,
            'y_heads_layer_widths': self.y_heads_layer_widths,
            'loss_weights': self.loss_weights,
            'activation': self.activation,
            'optimizer': self.optimizer,
            'pass_y_to_encoder': self.pass_y_to_encoder,
        }

    def configure_optimizers(self):
        """
        Configure the optimizers for the model.
        
        Returns:
            dict: A dictionary containing the optimizer(s) and learning rate scheduler(s) to be used during training.
        """
        # Initialize the optimizer with Adam, using the model parameters as the input arguments
        optimizer = self.optimizer if self.optimizer is not None else torch.optim.Adam(self.parameters())
        
        # Initialize the learning rate scheduler with ReduceLROnPlateau
        # This scheduler reduces the learning rate when the validation loss stops improving
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
            },
        }

    def _step(self, batches:List[Dict[str, np.array]], batch_idx:int, mode:str) -> Tuple[List[Dict[str, np.array]], List[Dict[str, float]]]:
        return zip(*[model._step(batch, batch_idx, mode) if batch is not None else None for model, batch in zip(self.models.values(), batches)])

    def forward(self, data:List[Dict[str, Dict[str, torch.Tensor]]], transform:bool=True) -> Dict[str, Dict[str, Union[np.array, torch.Tensor]]]:
        return {task_name: task_model.forward(d, transform) if d is not None else None for (task_name, task_model), d in zip(self.models.items(), data)}

    def training_step(self, batch:List[Dict[str, np.array]], batch_idx:int) -> float:
        _, loss_dicts = self._step(batch, batch_idx, mode='train')
        loss_dict = sum_join_dicts(loss_dicts)
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch[0][list(batch[0].keys())[0]]))
        return loss_dict['train_loss']

    def validation_step(self, batch:List[Dict[str, np.array]], batch_idx:int) -> float:
        _, loss_dicts = self._step(batch, batch_idx, mode='val')
        loss_dict = sum_join_dicts(loss_dicts)
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch[0][list(batch[0].keys())[0]]))
        return loss_dict['val_loss']

    def test_step(self, batch:List[Dict[str, np.array]], batch_idx:int) -> float:
        _, loss_dicts = self._step(batch, batch_idx, mode='test')
        loss_dict = sum_join_dicts(loss_dicts)
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch[0][list(batch[0].keys())[0]]))
        return loss_dict['test_loss']

    def encode(self, data:Union[CombinedLoader, List[Dict[str, np.array]]], transform:Union[bool, Tuple[bool, bool]]=True, batch_size:int=1) -> Dict[str, Dict[str, np.array]]:
        if isinstance(data, CombinedLoader):
            pred = []
            for batch in data:
                pred.append({t_name: t_model.encode(batch[i], transform, batch_size) for i, (t_name, t_model) in enumerate(self.models.items())})
            return rec_concat_dict(pred)
        else:
            return {t_name: t_model.encode(data[i], transform) for i, (t_name, t_model) in enumerate(self.models.items())}

    def encode_multitask(self, data:Union[DataLoader, Dict[str, np.array]], source_task:str, transform:Union[bool, Tuple[bool, bool]]=True, batch_size:int=1) -> Dict[str, Dict[str, np.array]]:
        pred = self.models[source_task].encode(data, transform=(transform, False), batch_size=batch_size, return_lat=True)
        mt_pred = {t_name: {
            'y': {name: head(pred['lat']) for name, head in model.encoder.out_heads.items()},
            'z': pred['z']
         } for t_name, model in self.models.items()}

        if transform:
            mt_pred = torch_dict_to_numpy(mt_pred)
            for t_name, t_pred in mt_pred.items():
                mt_pred[t_name]['y'] = self.models[t_name].dataset.inverse_transform(t_pred['y'])
            torch.set_grad_enabled(True)
            self.train()

        return mt_pred

    def decode(self, data:Union[CombinedLoader, Dict[str, np.array]], z:np.array, transform:Union[bool, Tuple[bool, bool]]=True, batch_size:int=1) -> Dict[str, Dict[str, np.array]]:
        if isinstance(data, CombinedLoader):
            pred = []
            for i, batch in enumerate(data):
                batch_size = data.batch_size
                pred.append({t_name: t_model.decode(batch[j], z[i * batch_size : (i + 1) * batch_size], transform, batch_size) for j, (t_name, t_model) in enumerate(self.models.items())})
            return rec_concat_dict(pred)
        else:
            return {t_name: t_model.encode(data[i], z, transform) for i, (t_name, t_model) in enumerate(self.models.items())}


    def decode_multitask(self, data:Union[DataLoader, Dict[str, np.array]], source_task:str, z:Dict[str, np.array], transform:bool=True, batch_size:int=1) -> Dict[str, Dict[str, np.array]]:
        pred = self.models[source_task].decode(data, z, transform=(transform, False), batch_size=batch_size, return_lat=True)
        mt_pred = {t_name: {
            'x': {name: head(pred['lat']) for name, head in model.decoder.out_heads.items()},
        } for t_name, model in self.models.items()}

        if transform:
            mt_pred = torch_dict_to_numpy(mt_pred)
            for t_name, t_pred in mt_pred.items():
                mt_pred[t_name]['x'] = self.models[t_name].dataset.inverse_transform(t_pred['x'])
            torch.set_grad_enabled(True)
            self.train()

        return mt_pred

    def fit(self, train_loader:CombinedLoader, val_loader:CombinedLoader, 
            max_epochs:int=100, callbacks:list=None, loggers:list=None, accelerator:str='auto', **kwargs):
        """
        Train the model.
        
        Args:
            train_loader (DataLoader): A PyTorch DataLoader object that provides the training data. 
            val_loader (DataLoader): A PyTorch DataLoader object that provides the validation data. 
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
        
        self.model_trainer.fit(self, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    def validate(self, val_loader:CombinedLoader, accelerator:str='auto', **kwargs) -> Dict[str, float]:
        """
        Evaluate the model on the validation data.
        
        Args:
            val_loader (CombinedLoader): A CombinedLoader object that provides the validation data.
            accelerator (str, optional): Which accelerator should be used (e.g. cpu, gpu, mps, etc.) Default: auto
        
        Returns:
            Dict[str, float]: A dictionary containing the validation loss and metrics.
        """
        # Create a trainer object if one does not already exist
        if self.model_trainer is None:
            self.model_trainer = pl.Trainer(accelerator=accelerator, **kwargs)

        return self.trainer.validate(self, dataloaders=val_loader)

    def test(self, test_loader:CombinedLoader, accelerator:str='auto', **kwargs) -> Dict[str, float]:
        """
        Evaluate the model on the test data.
        
        Args:
            test_loader (CombinedLoader): A CombinedLoader object that provides the test data.
            accelerator (str, optional): Which accelerator should be used (e.g. cpu, gpu, mps, etc.) Default: auto
        
        Returns:
            Dict[str, float]: A dictionary containing the test loss and metrics.
        """
        # Create a trainer object if one does not already exist
        if self.model_trainer is None:
            self.model_trainer = pl.Trainer(accelerator=accelerator, **kwargs)

        return self.trainer.test(self, dataloaders=test_loader)

    def predict(self, data:Union[CombinedLoader, DataLoader, Dict[str, Dict[str, np.array]]], source_task:str=None, accelerator:str='auto') -> Dict[str, Dict[str, np.array]]:
        if isinstance(data, CombinedLoader):
            # data contains input for a all tasks (as CombinedLoader) and an output is computed for all tasks
            if self.model_trainer is None:
                self.model_trainer = pl.Trainer(accelerator=accelerator)
            preds = self.model_trainer.predict(self, dataloaders=data)
            return rec_concat_dict(preds)

        elif isinstance(data, DataLoader) or isinstance(data, dict) and source_task is not None:
            # data contains input for a single task and an output is computed for all tasks
            preds = self.encode_multitask(data, source_task)
            preds.update(self.decode_multitask(data, source_task, preds[source_task]['z'], transform=False))
            return preds

        else:
            # data contains input for a all tasks (as dict) and an output is computed for all tasks
            return {t_name: self.models[t_name].predict(t_data, accelerator) for t_name, t_data in data.items()}
       
    def visual_evaluate(self, data, task:str=None, path:str=None):
        if isinstance(data, CombinedLoader):
            data = [torch_dict_to_numpy(rec_concat_dict(feature_batches)) for feature_batches in zip(*[batch for batch in data])]
            for (t_name, t_model), t_data in zip(self.models.items(), data):
                print('========================================================')
                print('Model', t_name)
                t_model.visual_evaluate(t_data, os.path.join(path, t_name) if path is not None else None)
                print('========================================================')
        elif isinstance(data, DataLoader) or (isinstance(data, dict) and task is not None):
            self.models[task].visual_evaluate(data, os.path.join(path, task) if path is not None else None)
        elif isinstance(data, dict) and task is None:
            for t_name, t_data in data.items():
                print('========================================================')
                print('Model', t_name)
                self.models[t_name].visual_evaluate(t_data, os.path.join(path, t_name) if path is not None else None)
                print('========================================================')
        else:
            raise ValueError(f'Type of "data" not recognized: {data.__class__}')

    def inspect_latent(self, data, task:str=None, path:str=None, **kwargs):
        if isinstance(data, CombinedLoader):
            data = [torch_dict_to_numpy(rec_concat_dict(feature_batches)) for feature_batches in zip(*[batch for batch in data])]
            for (t_name, t_model), t_data in zip(self.models.items(), data):
                t_model.inspect_latent(t_data, path=os.path.join(path, t_name) if path is not None else None, **kwargs)
        elif isinstance(data, DataLoader) or (isinstance(data, dict) and task is not None):
            self.models[task].inspect_latent(data, path=os.path.join(path, task) if path is not None else None, **kwargs)
        elif isinstance(data, dict) and task is None:
            for t_name, t_data in data.items():
                self.models[t_name].inspect_latent(t_data, path=os.path.join(path, t_name) if path is not None else None, **kwargs)
        else:
            raise ValueError(f'Type of "data" not recognized: {data.__class__}')

    def sensitivity_analysis(self, data, task:str, features:List[str]=None, path:str=None):
        if isinstance(data, dict) or isinstance(data, DataLoader):
            return self.models[task].sensitivity_analysis(data, features, path=os.path.join(path, task) if path is not None else None)
        else:
            raise ValueError(f'Type of "data" not supported: {data.__class__}')

    def __cross_task_sensitivity_gradients_x(self, data, task, x_name):
        data = {d_key: torch.Tensor(d_value).float().requires_grad_() for d_key, d_value in numpy_dict_to_tensor(data, device=self.device).items()}
        z = self.models[task].encode(data, transform=False)['z']
        dec = self.decode_multitask(data, task, z, transform=False)
        for t in dec.keys():
            if x_name in dec[t]['x'].keys():
                dec[t]['x'][x_name].mean().backward(retain_graph=True)
        
        return torch_dict_to_numpy({x_key: data[x_key].grad[:, 0] if data[x_key].grad is not None else torch.zeros_like(data[x_key])[:, 0] for x_key in self.models[task].decoder.out_heads.keys()}), \
               torch_dict_to_numpy({y_key: data[y_key].grad[:, 0] if data[y_key].grad is not None else torch.zeros_like(data[y_key])[:, 0] for y_key in self.models[task].encoder.out_heads.keys()})

    def __cross_task_sensitivity_gradients_y(self, data, task, y_name):
        data = {d_key: torch.Tensor(d_value).float().requires_grad_() for d_key, d_value in numpy_dict_to_tensor(data, device=self.device).items()}
        enc = self.encode_multitask(data, task, transform=False)
        for t in enc.keys():
            if y_name in enc[t]['y'].keys():
                enc[t]['y'][y_name].mean().backward(retain_graph=True)
        return torch_dict_to_numpy({x_key: data[x_key].grad[:, 0] if data[x_key].grad is not None else torch.zeros_like(data[x_key])[:, 0] for x_key in self.models[task].decoder.out_heads.keys()})

    def cross_task_sensitivity_analysis(self, data, tasks:Union[str, List[str]]=None, features:List[str]=None, path:str=None):
        self.eval()
        torch.set_grad_enabled(True)

        if features is None: 
            # If no specific feature is given, create sensitivities for all features
            features = list(self.encoder.out_heads.keys()) + list(self.decoder.out_heads.keys())
        elif isinstance(features, str):
            features = [features]

        if isinstance(tasks, str):
            tasks = [tasks]

        if isinstance(data, dict):
            # if data is a dict, then the top-most keys must denote the tasks for which data is provided
            if tasks is None:
                tasks = list(data.keys())
        elif isinstance(data, DataLoader) and tasks:
            unr_data = torch_dict_to_numpy(rec_concat_dict([batch for batch in data]))
            data = {tasks[0]: unr_data}
        elif isinstance(data, CombinedLoader):
            if tasks is None:
                raise ValueError('If "data" is a CombinedLoader, "tasks" must be provided.')
            unr_data = [torch_dict_to_numpy(rec_concat_dict(feature_batches)) for feature_batches in zip(*[batch for batch in data])]
            data = {m: d for m, d in zip(tasks, unr_data)}
        else:
            raise ValueError(f'Type of "data" not supported: {data.__class__}')

        data_transformed = {t: self.models[t].dataset.transform({key: data[t][key] for key in (list(self.models[t].encoder.in_heads.keys()) + list(self.models[t].decoder.in_heads.keys()))}) for t in tasks}
        
        for feature in features:
            involved_tasks = [task for task in tasks if feature in data[task]]

            if feature in self.dataset.x.keys():
                mt_x_wrt_x, mt_x_wrt_y = [], []
                for task in tasks:
                    if feature in data[task]:
                        x_wrt_x, x_wrt_y = self.__cross_task_sensitivity_gradients_x(data_transformed[task], task, feature)
                        mt_x_wrt_x.append(x_wrt_x)
                        mt_x_wrt_y.append(x_wrt_y)
                x_wrt_x = rec_concat_dict(mt_x_wrt_x)
                x_wrt_y = rec_concat_dict(mt_x_wrt_y)

                #data_hue = [data[task][feature] for task in involved_tasks]

                swarm_plot(
                    features={k + ' (' + ', '.join([task for task in tasks if k in data[task]]) + ')': v for k, v in x_wrt_x.items()}, 
                    #data_hue=np.concatenate(data_hue, axis=0) if len(data_hue) > 1 else data_hue[0], 
                    title=feature + ' (' + ', '.join(involved_tasks) + ')',
                    y_label='Input features', 
                    path=os.path.join(path, 'dx') if path is not None else None,
                )
                swarm_plot(
                    features={k + ' (' + ', '.join([task for task in tasks if k in data[task]]) + ')': v for k, v in x_wrt_y.items()}, 
                    #data_hue=np.concatenate(data_hue, axis=0) if len(data_hue) > 0 else data_hue[0], 
                    title=feature + ' (' + ', '.join(involved_tasks) + ')', 
                    y_label='Conditional features', 
                    path=os.path.join(path, 'dx') if path is not None else None,
                )
            else:
                mt_y_wrt_x = []
                for task in tasks:
                    if feature in data[task]:
                        mt_y_wrt_x.append(self.__cross_task_sensitivity_gradients_y(data_transformed[task], task, feature))
                y_wrt_x = rec_concat_dict(mt_y_wrt_x)

                #data_hue = [data[task][feature] for task in involved_tasks]
                
                swarm_plot(
                    features={k + ' (' + ', '.join([task for task in tasks if k in data[task]]) + ')': v for k, v in y_wrt_x.items()}, 
                    #data_hue=np.concatenate(data_hue, axis=0) if len(data_hue) > 1 else data_hue[0], 
                    title=feature + ' (' + ', '.join(involved_tasks) + ')', 
                    y_label='Input features', 
                    path=os.path.join(path, 'dy') if path is not None else None,
                )
        self.train()

    def summary(self, max_depth:int=1) -> None:
        """
        Prints a summary of the encoder and decoder, including the number of parameters, the layers, 
        their names, and the dimensionality.
        
        Parameters:
        max_depth (int, optional): Maximum depth of modules to show. Use -1 to show all modules or 0 to show no summary. Defaults to 1.
        """
        # Register example input array such that Model_Summary can print data shapes
        for t_name, t_model in self.models.items():
            print(f'Task {t_name}')
            t_model.summary(max_depth)
            print('\n')
        
        print('Full model with shared weights')
        print(pl.utilities.model_summary.ModelSummary(self, max_depth=max_depth))

    def save(self, path:str):
        for task, model in self.models.items():
            model.save(os.path.join(path, task))
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.get_config(), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_model(cls, path:str):
        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        tasks = [mode_path for mode_path in os.listdir(path) if os.path.isdir(os.path.join(path, mode_path))]
        models = {task: CondAEModel.load_model(os.path.join(path, task)) for task in tasks}
        obj = cls(MultiTaskDataSet([model.dataset for model in models.values()]), **config)
        obj.models = nn.ModuleDict(models)
        return obj

class MultiTaskCondVAEModel(MultiTaskCondAEModel):
    """
        Parameters:
        dataset (MultiTaskDataSet): Object containing the datasets corresponding to each task.
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
            added to the total loss used for backpropagation. Defaults to {'x': 1., 'y': 1., 'kl': 0.1, 'decorrelation': 0.}.
        activation (Union[torch.nn.Module, str], optional): Activation function to be used in the latent layers of the autoencoder. Defaults to leaky ReLU.
        optimizer (torch.optim.Optimizer), optional: Optimizer to be used for updating the model's weights. Defaults to Adam.
        pass_y_to_encoder (bool), optional: Whether to pass the conditional features y to the autoencoder's encoder (vanilla cVAE formulation) or not.
            In the first case, the encoder maps from x to z and is solely used for finding the latent vector needed to reconstruct x given y.
            In the latter case, the encoder represents a surrogate model mapping from x to y as well as a latent vector z. Defaults to False.
        name (str): Name of the model.
        """
    def __init__(self, dataset:MultiTaskDataSet, layer_widths:List[int], latent_dim:int, x_heads_layer_widths:Dict[str, List[int]]={}, y_heads_layer_widths:Dict[str, List[int]]={}, 
                        loss_weights:Dict[str, float]=None, activation:Union[torch.nn.Module, str]='leaky_relu', optimizer:torch.optim.Optimizer=None, pass_y_to_encoder:bool=False, 
                        name:str='CondAEModel', **kwargs):
        super().__init__(dataset, layer_widths, latent_dim, x_heads_layer_widths, y_heads_layer_widths, loss_weights, activation, optimizer, pass_y_to_encoder, name, **kwargs)

        self.encoder = VEncoder({x_key: x_heads[0] for x_key, x_heads in self.x_heads.items()}, 
                               {y_key: y_heads[1] for y_key, y_heads in self.y_heads.items()}, 
                               self.layer_widths, self.latent_dim, self.activation)
        
        self.models = torch.nn.ModuleDict({ds.name: CondVAEModel(ds, self.layer_widths, self.latent_dim, name=name + '_model') for ds in dataset.datasets})

        for model in self.models.values():
            model.encoder.fc_z_log_var = self.encoder.fc_z_log_var


    def decode(self, data:Union[CombinedLoader, Dict[str, np.array]], z:Dict[str, np.array]=None, transform:bool=True, batch_size:int=1) -> Dict[str, Dict[str, np.array]]:
        if z is None:
            if isinstance(data, dict):
                z = data.get('z', torch.normal(mean=0., std=1., size=(len(list(data.values())[0]), self.latent_dim)))
            else:
                z = torch.normal(mean=0., std=1., size=(len(data.dataset), self.latent_dim))
        elif isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()
        return super().decode(data, z, transform, batch_size)

    def decode_multitask(self, data:Union[DataLoader, Dict[str, np.array]], source_task:str, z:Dict[str, np.array]=None, transform:bool=True, batch_size:int=1) -> Dict[str, Dict[str, np.array]]:
        if z is None:
            if isinstance(data, dict):
                z = data.get('z', torch.normal(mean=0., std=1., size=(len(list(data.values())[0]), self.latent_dim)))
            else:
                z = torch.normal(mean=0., std=1., size=(len(data.dataset), self.latent_dim))
        elif isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()
        return super().decode_multitask(data, source_task, z, transform, batch_size)

    @classmethod
    def load_model(cls, path:str):
        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        tasks = [mode_path for mode_path in os.listdir(path) if os.path.isdir(os.path.join(path, mode_path))]
        models = {task: CondVAEModel.load_model(os.path.join(path, task)) for task in tasks}
        obj = cls(MultiTaskDataSet([model.dataset for model in models.values()]), **config)
        obj.models = torch.nn.ModuleDict(models)
        return obj