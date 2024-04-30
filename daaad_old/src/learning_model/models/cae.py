import os
import torch
import pickle
import numpy as np
import warnings
import pytorch_lightning as pl

from sklearn_som.som import SOM
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Union
from pytorch_lightning.trainer.supporters import CombinedLoader

from src.learning_model.models.encoders import Encoder
from src.learning_model.models.decoders import Decoder
from src.learning_model.dataset.data_set import DataSet
from src.utils import numpy_dict_to_tensor, rec_concat_dict, sum_join_dicts, swarm_plot, torch_dict_to_numpy

warnings.filterwarnings("ignore", ".*does not have many workers.*")

class CondAEModel(pl.LightningModule):

    def __init__(self, dataset:DataSet, core_layer_widths:List[int], latent_dim:int, x_heads_layer_widths:dict={}, y_heads_layer_widths:dict={}, 
                        loss_weights:dict=None, activation:str='leaky_relu', optimizer:torch.optim.Optimizer=None, pass_y_to_encoder:bool=False, guided:bool=False, name:str='CondAEModel', **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.x_heads  = {x_key: x_value.get_heads(x_heads_layer_widths.get(x_key, []), core_layer_widths[0], activation) \
            for x_key, x_value in dataset.x.items()}
        self.y_heads  = {y_key: y_value.get_heads(y_heads_layer_widths.get(y_key, []), core_layer_widths[-1], activation) \
            for y_key, y_value in dataset.y.items()}

        self.encoder = Encoder({x_key: x_heads[0] for x_key, x_heads in (self.x_heads.items() if not pass_y_to_encoder else (self.x_heads | self.y_heads).items())}, 
                               {y_key: y_heads[1] for y_key, y_heads in self.y_heads.items() if not pass_y_to_encoder} if not pass_y_to_encoder else {}, 
                               core_layer_widths, latent_dim, activation)
        self.decoder = Decoder({y_key: y_heads[0] for y_key, y_heads in self.y_heads.items()}, 
                               {x_key: x_heads[1] for x_key, x_heads in self.x_heads.items()}, 
                               core_layer_widths[::-1], latent_dim, activation, guided)

        self.dataset = dataset
        self.name = name
        self.core_layer_widths = core_layer_widths
        self.latent_dim = latent_dim
        self.x_heads_layer_widths = x_heads_layer_widths
        self.y_heads_layer_widths = y_heads_layer_widths
        self.loss_weights = loss_weights if loss_weights else {}
        self.feature_losses = dataset.get_objectives()
        self.activation = activation
        self.optimizer = optimizer
        self.pass_y_to_encoder = pass_y_to_encoder
        self.guided = guided

    def eval(self, *args, **kwargs):
        for heads in (self.x_heads | self.y_heads).values():
            heads[0].eval(*args, **kwargs)
            heads[1].eval(*args, **kwargs)
        super().eval(*args, **kwargs)

    def train(self, *args, **kwargs):
        for heads in (self.x_heads | self.y_heads).values():
            heads[0].train(*args, **kwargs)
            heads[1].train(*args, **kwargs)
        super().train(*args, **kwargs)

    def get_config(self):
        return {
            'core_layer_widths': self.core_layer_widths,
            'latent_dim': self.latent_dim,
            'x_heads_layer_widths': self.x_heads_layer_widths,
            'y_heads_layer_widths': self.y_heads_layer_widths,
            'loss_weights': self.loss_weights,
            'activation': self.activation,
            'optimizer': self.optimizer,
            'pass_y_to_encoder': self.pass_y_to_encoder,
            'guided': self.guided,
        }

    def configure_optimizers(self):
        optimizer = self.optimizer if self.optimizer is not None else torch.optim.Adam(self.parameters())
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True),
                "monitor": "val_loss",
            },
        }

    def _step(self, batch:dict, batch_idx:int, mode:str) -> dict:
        data = self.dataset.transform({d_key: batch[d_key] for d_key in (self.dataset.x | self.dataset.y).keys()})
        
        if mode == 'train':
            data = self.dataset.augment(data)
        data = numpy_dict_to_tensor(data)

        pred = self(data, transform=False)

        x_losses = {key: self.feature_losses[key](pred['x'][key], data[key].float()) for key in self.dataset.x.keys()}
        x_loss = torch.stack(list(x_losses.values()), dim=0).sum()
        y_losses = {key: self.feature_losses[key](pred['y'][key], data[key].float()) for key in self.dataset.y.keys()}
        y_loss = torch.stack(list(y_losses.values()), dim=0).sum() if len(y_losses) > 0 else 0.

        if mode == 'train' and self.guided:
            g_pred = self.decoder({'z': pred['z'].detach(), 'y': {key: torch.zeros_like(val) for key, val in data.items()}}, cond_weights=0.)['x']
            g_losses = {key: self.feature_losses[key](g_pred[key], data[key].float()) for key in self.dataset.x.keys()}
            g_loss = torch.stack(list(g_losses.values()), dim=0).sum()

        # calculate only if decorrelation weight is > 0 to avoid computing gradients for nothing
        if self.loss_weights.get('decorrelation', 0) > 0 and len(self.dataset.y) > 0:
            # decorrelate by reducing the covariance: https://arxiv.org/abs/1904.01277v1
            decorrelation_loss = ((torch.permute(torch.cat([torch.Tensor(data[k] - data[k].mean(axis=tuple(range(1, len(data[k].shape))))) for k in pred['y'].keys()], dim=-1), dims=(1, 0)).float() @ pred['z'].float())**2).mean()
        else:
            # weight is zero, so decorrelation_loss is detached from the graph
            decorrelation_loss = 0.

        total_loss = self.loss_weights.get('x', 1.) * x_loss + \
                     self.loss_weights.get('y', 1.) * y_loss + \
                     self.loss_weights.get('decorrelation', 0.) * decorrelation_loss
        if mode == 'train' and self.guided:
            total_loss += self.loss_weights.get('x', 1.) * g_loss

        loss_dict = {
                mode + '_loss': total_loss,
                mode + '_features_loss': x_loss + y_loss,
                mode + '_decorrelation_loss': decorrelation_loss,
            } | {mode + '_' + key + '_loss': value for key, value in x_losses.items()} \
              | {mode + '_' + key + '_loss': value for key, value in y_losses.items()}
        if mode == 'train' and self.guided:
            loss_dict['guided_loss'] = g_loss

        return pred, loss_dict

    def forward(self, data:dict, transform:bool=True, cond_weights:List[float]=None):
        if transform:
            data = self.dataset.transform({d_key: data[d_key] for d_key in (self.dataset.x | self.dataset.y).keys()})
            torch.set_grad_enabled(False)
            self.eval()

        x = numpy_dict_to_tensor({x_key: data[x_key] for x_key in self.dataset.x.keys()})
        y = numpy_dict_to_tensor({y_key: data[y_key] for y_key in self.dataset.y.keys()})

        if self.pass_y_to_encoder:
            pred = self.encoder(x | y)
        else:
            pred = self.encoder(x)
            
        pred.update(self.decoder({'z': pred['z'], 'y': y}, cond_weights))
        if transform:
            pred = torch_dict_to_numpy(pred)
            pred['x'] = self.dataset.inverse_transform(pred['x'])
            pred['y'] = self.dataset.inverse_transform(pred['y'])
            torch.set_grad_enabled(True)
            self.train()
        return pred

    def training_step(self, batch:dict, batch_idx:int):
        _, loss_dict = self._step(batch, batch_idx, mode='train')
        self.log_dict(loss_dict, batch_size=list(batch.values())[0].shape[0])
        return loss_dict['train_loss']

    def validation_step(self, batch, batch_idx):
        _, loss_dict = self._step(batch, batch_idx, mode='val')
        self.log_dict(loss_dict, batch_size=list(batch.values())[0].shape[0])
        return loss_dict['val_loss']

    def test_step(self, batch, batch_idx):
        _, loss_dict = self._step(batch, batch_idx, mode='test')
        self.log_dict(loss_dict, batch_size=list(batch.values())[0].shape[0])
        return loss_dict['test_loss']

    def encode(self, data:Union[DataLoader, dict], transform:bool=True):
        if isinstance(data, DataLoader):
            data = rec_concat_dict([batch for batch in data])
        if transform:
            data = numpy_dict_to_tensor(self.dataset.transform({x_key: data[x_key] for x_key in (self.dataset.x.keys() | (self.dataset.y.keys() if self.pass_y_to_encoder else {}))}))
            torch.set_grad_enabled(False)
            self.eval()
        
        pred = self.encoder(data)

        if transform:
            pred = torch_dict_to_numpy(pred)
            pred['y'] = self.dataset.inverse_transform(pred['y'])
            torch.set_grad_enabled(True)
            self.train()
        return pred

    def decode(self, data:Union[DataLoader, dict], z:np.array, transform:bool=True, cond_weights:List[float]=None):
        if isinstance(data, DataLoader):
            data = rec_concat_dict([batch for batch in data])
        if transform:
            data = numpy_dict_to_tensor(self.dataset.transform({y_key: data[y_key] for y_key in self.dataset.y.keys()}))
            torch.set_grad_enabled(False)
            self.eval()

        pred = self.decoder({'z': z.float() if torch.is_tensor(z) else torch.from_numpy(z).float(), 'y': data}, cond_weights)['x']

        if transform:
            pred = self.dataset.inverse_transform(torch_dict_to_numpy(pred))
            torch.set_grad_enabled(True)
            self.train()
        return pred

    def fit(self, train_loader:DataLoader, val_loader:DataLoader, max_epochs:int=100, callbacks:list=None, loggers:list=None, **kwargs):
        torch.set_grad_enabled(True)
        self.train()
        trainer = pl.Trainer(
            accelerator='auto',
            auto_select_gpus=True,
            max_epochs=max_epochs,
            callbacks=callbacks if callbacks else [],
            logger=loggers if loggers else [],
            **kwargs,
        )
        trainer.fit(self, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    def validate(self, val_loader:DataLoader):
        return pl.Trainer().validate(self, dataloaders=val_loader)
    
    def test(self, test_loader:DataLoader):
        return pl.Trainer().test(self, dataloaders=test_loader)

    def predict(self, data:Union[DataLoader, dict]) -> dict:
        if isinstance(data, DataLoader):
            preds = pl.Trainer().predict(self, dataloaders=data)
            return rec_concat_dict(preds)
        else:
            torch.set_grad_enabled(False)
            self.eval()
            preds = self(data)
            torch.set_grad_enabled(True)
            self.train()
            return preds
       
    def visual_evaluate(self, data, path:str=None):
        if isinstance(data, DataLoader):
            data = torch_dict_to_numpy(rec_concat_dict([batch for batch in data]))
        pred = self.predict(data)
        
        for x_key, x_data in pred['x'].items():
            feature = self.dataset.x[x_key]
            feature.compare(
                {
                    'True values': data[x_key].astype(feature.data_type), 
                    'Predicted values': x_data.astype(feature.data_type)
                }, 
                title=feature.name, axis=None, path=path, evaluation=True
            )

        for y_key, y_data in pred['y'].items():
            feature = self.dataset.y[y_key]
            feature.compare(
                {
                    'True values': data[y_key].astype(feature.data_type), 
                    'Predicted values': y_data.astype(feature.data_type)
                }, 
                title=feature.name, axis=None, path=path, evaluation=True
            )

    def inspect_latent(self, data, dim_reduction_method:str=None, dim_ix_0:int=0, dim_ix_1:int=1, path:str=None):
        if isinstance(data, DataLoader):
            data = torch_dict_to_numpy(rec_concat_dict([batch for batch in data]))
        z = self.encode(data)['z']

        if z.shape[-1] > 2 and dim_reduction_method is not None:
            if dim_reduction_method == 'pca': 
                reducer = PCA(n_components=2)
            if dim_reduction_method == 'tsne':
                reducer = TSNE()
            if dim_reduction_method == 'som': 
                reducer = SOM(m=2, n=1, dim=z.shape[-1])
            z = reducer.fit_transform(z)
        elif z.shape[-1] > 2 and dim_reduction_method is None:
            z = np.stack([z[:, dim_ix_0], z[:, dim_ix_1]], axis=1)

        print('Latent space for x features')
        for x_key, x_feature in self.dataset.x.items():
            x_feature.inspect_latent(
                z[:, 0], z[:, 1], data[x_key].flatten().astype(x_feature.data_type),
                title=x_feature.name, path=path
            )

        print('Latent space for y features')
        for y_key, y_feature in self.dataset.y.items():
            y_feature.inspect_latent(
                z[:, 0], z[:, 1], data[y_key].flatten().astype(x_feature.data_type),
                title=y_feature.name, path=path
            )

    def __sensitivity_gradients_x(self, data, x_name):
        data = {x_key: torch.Tensor(data[x_key]).float().requires_grad_() for x_key in (self.dataset.x | self.dataset.y).keys()}
        self(data, transform=False)['x'][x_name].sum(axis=0).mean().backward()
        return torch_dict_to_numpy({x_key: torch.reshape(data[x_key].grad, (data[x_key].grad.shape[0], -1)).mean(dim=1) for x_key in self.dataset.x.keys()}), \
               torch_dict_to_numpy({y_key: torch.reshape(data[y_key].grad, (data[y_key].grad.shape[0], -1)).mean(dim=1) for y_key in self.dataset.y.keys()})

    def __sensitivity_gradients_y(self, data, y_name):
        data = {x_key: torch.Tensor(data[x_key]).float().requires_grad_() for x_key in self.dataset.x.keys()}
        self.encoder(data)['y'][y_name].sum(axis=0).mean().backward()
        return torch_dict_to_numpy({x_key: torch.reshape(data[x_key].grad, (data[x_key].grad.shape[0], -1)).mean(dim=1) for x_key in self.dataset.x.keys()})

    def sensitivity_analysis(self, data, features:list=None, path:str=None):
        self.eval()
        if isinstance(data, DataLoader):
            data = rec_concat_dict([batch for batch in data])
        data_transformed = numpy_dict_to_tensor(self.dataset.transform({key: data[key] for key in (self.dataset.x | self.dataset.y).keys()}))
        all_features = list(self.dataset.x.keys()) + list(self.dataset.y.keys())

        # If no specific feature is given, create sensitivities for all features
        if features is None: 
            features = all_features
        
        for feature in features:
            if feature in self.dataset.x.keys():
                x_wrt_x, x_wrt_y = self.__sensitivity_gradients_x(data_transformed, feature)
                swarm_plot(
                    features=x_wrt_x, 
                    data_hue=data[feature], 
                    title=feature, 
                    y_label='Input features', 
                    path=os.path.join(path, 'dx') if path is not None else None,
                )
                swarm_plot(
                    features=x_wrt_y, 
                    data_hue=data[feature], 
                    title=feature, 
                    y_label='Conditional features', 
                    path=os.path.join(path, 'dx') if path is not None else None,
                )
            elif feature in self.dataset.y.keys():
                y_wrt_x = self.__sensitivity_gradients_y(data_transformed, feature)
                swarm_plot(
                    features=y_wrt_x, 
                    data_hue=data[feature], 
                    title=feature, 
                    y_label='Input features', 
                    path=os.path.join(path, 'dy') if path is not None else None,
                )
        self.train()

    def summary(self, **kwargs):
        data = self.dataset.get_shapes()
        print(self.name + ' encoder')
        self.encoder.summary(input_data=data, **kwargs)
        print(self.name + ' decoder')
        self.decoder.summary(input_data=data, **kwargs)

    def save(self, path:str):
        self.dataset.save(os.path.join(path, 'dataset'), include_data=False)
        torch.save(self.state_dict(), os.path.join(path, 'weights.pt'))
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.get_config(), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_model(cls, path:str):
        ds = DataSet.from_path(os.path.join(path, 'dataset'))
        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        cae = cls(ds, **config)
        cae.load_state_dict(torch.load(os.path.join(path, 'weights.pt')))
        return cae


class MultiModalCondAEModel(CondAEModel):
    def __init__(self, datasets:Dict[str, DataSet], core_layer_widths:List[int], latent_dim:int, name:str='MultiModalCondAEModel', *args, **kwargs):
        super_ds = DataSet.from_features_list([x for ds in datasets.values() for x in ds.x.values()], 
                                              [y for ds in datasets.values() for y in ds.y.values()])
        super().__init__(dataset=super_ds, core_layer_widths=core_layer_widths, latent_dim=latent_dim, name=name, *args, **kwargs)
        self.save_hyperparameters()

        self.x_heads = {x_key: x_value.get_heads(self.x_heads_layer_widths.get(x_key, []), self.core_layer_widths[0], self.activation) \
            for x_key, x_value in self.dataset.x.items()}
        self.y_heads = {y_key: y_value.get_heads(self.y_heads_layer_widths.get(y_key, []), self.core_layer_widths[-1], self.activation) \
            for y_key, y_value in self.dataset.y.items()}

        self.encoder = Encoder({x_key: x_heads[0] for x_key, x_heads in self.x_heads.items()}, 
                               {y_key: y_heads[1] for y_key, y_heads in self.y_heads.items()}, 
                               self.core_layer_widths, self.latent_dim, self.activation)
        self.decoder = Decoder({y_key: y_heads[0] for y_key, y_heads in self.y_heads.items()}, 
                               {x_key: x_heads[1] for x_key, x_heads in self.x_heads.items()}, 
                               self.core_layer_widths[::-1], self.latent_dim, self.activation)
        
        self.models = {name: CondAEModel(ds, self.core_layer_widths, self.latent_dim, name=name + '_model') for name, ds in datasets.items()}
        self._share_weights_across_models()


    def _share_weights_across_models(self):
        # replace shared heads and core
        for model in self.models.values():
            model.x_heads = {x_key: self.x_heads[x_key] for x_key in model.x_heads.keys()}
            model.y_heads = {y_key: self.y_heads[y_key] for y_key in model.y_heads.keys()}
            model.encoder.blocks = [model.encoder.blocks[0]] + self.encoder.blocks[1:]
            model.encoder.seq_blocks = torch.nn.Sequential(*model.encoder.blocks)
            model.decoder.blocks = [model.decoder.blocks[0]] + self.decoder.blocks[1:]
            model.decoder.seq_blocks = torch.nn.Sequential(*model.decoder.blocks)

    def get_config(self):
        return {
            'core_layer_widths': self.core_layer_widths,
            'latent_dim': self.latent_dim,
        } | super().get_config()

    def _step(self, batches:List[dict], batch_idx:int, mode:str) -> Tuple[List[dict]]:
        return zip(*[model._step(batch, batch_idx, mode) if batch is not None else None for model, batch in zip(self.models.values(), batches)])

    def forward(self, data:List[Dict[str, Dict[str, torch.Tensor]]], transform:bool=True, cond_weights:List[float]=None) -> Dict[str, Dict[str, torch.Tensor]]:
        return {m_name: model.forward(d, transform, cond_weights) if d is not None else None for (m_name, model), d in zip(self.models.items(), data)}

    def training_step(self, batch:List[dict], batch_idx:int):
        _, loss_dicts = self._step(batch, batch_idx, mode='train')
        loss_dict = sum_join_dicts(loss_dicts)
        self.log_dict(loss_dict, batch_size=list(batch[0].values())[0].shape[0])
        return loss_dict['train_loss']

    def validation_step(self, batch, batch_idx):
        _, loss_dicts = self._step(batch, batch_idx, mode='val')
        loss_dict = sum_join_dicts(loss_dicts)
        self.log_dict(loss_dict, batch_size=list(batch[0].values())[0].shape[0])
        return loss_dict['val_loss']

    def test_step(self, batch, batch_idx):
        _, loss_dicts = self._step(batch, batch_idx, mode='test')
        loss_dict = sum_join_dicts(loss_dicts)
        self.log_dict(loss_dict, batch_size=list(batch[0].values())[0].shape[0])
        return loss_dict['test_loss']

    def encode(self, data:Union[CombinedLoader, List[Dict[str, np.array]]], transform:bool=True) -> Dict[str, Dict[str, np.array]]:
        if isinstance(data, CombinedLoader):
            data = [rec_concat_dict(feature_batches) for feature_batches in zip(*[batch for batch in data])]
            return {m_name: model.encode(d, transform) for (m_name, model), d in zip(self.models.items(), data)}
        else:
            return {m_name: model.encode(data, transform) for (m_name, model) in self.models.items()}

    def encode_multimodal(self, data:Union[DataLoader, Dict[str, np.array]], mode:str, transform:bool=True) -> Dict[str, Dict[str, np.array]]:
        if isinstance(data, DataLoader):
            data = rec_concat_dict([batch for batch in data])

        if transform:
            data = numpy_dict_to_tensor(self.dataset.transform({x_key: data[x_key] for x_key in self.models[mode].dataset.x.keys()}))

        heads_emb = torch.cat([x_head(data[name]) for name, x_head in self.models[mode].encoder.in_heads.items()], dim=-1)
        emb = self.models[mode].encoder.blocks[0](heads_emb)
        mm_lat = {m_name: torch.nn.Sequential(*model.encoder.blocks[1:])(emb) for m_name, model in self.models.items()}
        mm_pred = {m_name: {
            'y': {y_name: y_head(mm_lat[m_name]) for y_name, y_head in model.encoder.out_heads.items()},
            'z': model.encoder.fc_z(mm_lat[m_name])
         } for m_name, model in self.models.items()}
        if transform:
            mm_pred = torch_dict_to_numpy(mm_pred)
            for m_name, pred in mm_pred.items():
                mm_pred[m_name]['y'] = self.dataset.inverse_transform(pred['y'])
        return mm_pred

    def decode(self, data:Union[CombinedLoader, Dict[str, np.array]], z:Dict[str, np.array], transform:bool=True, cond_weights:List[float]=None) -> Dict[str, Dict[str, np.array]]:
        if not isinstance(z, dict):
            z = {'z': z}

        if isinstance(data, CombinedLoader):
            data = [rec_concat_dict(feature_batches) for feature_batches in zip(*[batch for batch in data])]
            return {m_name: model.decode(d, z.get(m_name, z['z']), transform, cond_weights) for (m_name, model), d in zip(self.models.items(), data)}
        else:
            return {m_name: model.decode(data, z.get(m_name, z['z']), transform, cond_weights) for (m_name, model) in self.models.items()}

    def decode_multimodal(self, data:Union[DataLoader, Dict[str, np.array]], mode:str, z:Dict[str, np.array], transform:bool=True, cond_weights:List[float]=None) -> Dict[str, Dict[str, np.array]]:
        if isinstance(data, DataLoader):
            data = rec_concat_dict([batch for batch in data])

        if transform:
            data = numpy_dict_to_tensor(self.dataset.transform({y_key: data[y_key] for y_key in self.models[mode].dataset.y.keys()}))

        heads_emb = torch.cat([z] + [y_head(data[name]) for name, y_head in self.models[mode].decoder.in_heads.items()], dim=-1)
        emb = self.models[mode].decoder.blocks[0](heads_emb)
        mm_lat = {m_name: torch.nn.Sequential(*model.decoder.blocks[1:])(emb) for m_name, model in self.models.items()}
        mm_x = {m_name: {x_name: x_head(mm_lat[m_name]) for x_name, x_head in model.decoder.out_heads.items()} for m_name, model in self.models.items()}

        if transform:
            mm_x = torch_dict_to_numpy(mm_x)
            return {m_name: self.dataset.inverse_transform(x) for m_name, x in mm_x.items()}
        return mm_x

    def fit(self, train_loader:CombinedLoader, val_loader:CombinedLoader, max_epochs:int=100, callbacks:list=None, loggers:list=None, **kwargs):
        trainer = pl.Trainer(
            accelerator='auto',
            auto_select_gpus=True,
            max_epochs=max_epochs,
            callbacks=callbacks if callbacks else [],
            logger=loggers if loggers else [],
            **kwargs,
        )
        trainer.fit(self, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    def validate(self, val_loader:CombinedLoader):
        return pl.Trainer().validate(self, dataloaders=val_loader)
    
    def test(self, test_loader:CombinedLoader):
        return pl.Trainer().test(self, dataloaders=test_loader)

    def predict(self, data:Union[CombinedLoader, DataLoader, dict], mode:str=None) -> Dict[str, Dict[str, np.array]]:
        if isinstance(data, CombinedLoader):
            # data contains input for a all modes (as CombinedLoader) and an output is computed for all modes
            preds = pl.Trainer().predict(self, dataloaders=data)
            return rec_concat_dict(preds)
        elif isinstance(data, DataLoader) or isinstance(data, dict) and mode is not None:
            # data contains input for a single mode and an output is computed for all modes
            preds = self.encode_multimodal(data, mode)
            preds.update(self.decode_multimodal(preds[mode]['y'], mode, preds[mode]['z'], transform=False))
            return preds
        else:
            # data contains input for a all modes (as dict) and an output is computed for all modes
            return {m_name: self.models[m_name](d) for m_name, d in data.items()}
       
    def visual_evaluate(self, data, mode:str=None, path:str=None):
        if isinstance(data, CombinedLoader):
            data = [torch_dict_to_numpy(rec_concat_dict(feature_batches)) for feature_batches in zip(*[batch for batch in data])]
            for (m_name, model), d in zip(self.models.items(), data):
                model.visual_evaluate(d, os.path.join(path, m_name) if path is not None else None)
        elif isinstance(data, DataLoader):
            data = torch_dict_to_numpy(rec_concat_dict([batch for batch in data]))
            self.models[mode].visual_evaluate(data, os.path.join(path, m_name) if path is not None else None)
        elif isinstance(data, dict) and mode is not None:
            self.models[mode].visual_evaluate(data, os.path.join(path, m_name) if path is not None else None)
        elif isinstance(data, dict) and mode is None:
            for m_name, d in data.items():
                self.models[m_name].visual_evaluate(d, os.path.join(path, m_name) if path is not None else None)
        else:
            raise ValueError(f'Type of "data" not recognized: {data.__class__}')

    def inspect_latent(self, data, mode:str=None, path:str=None, **kwargs):
        if isinstance(data, CombinedLoader):
            data = [torch_dict_to_numpy(rec_concat_dict(feature_batches)) for feature_batches in zip(*[batch for batch in data])]
            for (m_name, model), d in zip(self.models.items(), data):
                model.inspect_latent(d, path=os.path.join(path, m_name) if path is not None else None, **kwargs)
        elif isinstance(data, DataLoader):
            data = torch_dict_to_numpy(rec_concat_dict([batch for batch in data]))
            self.models[mode].inspect_latent(data, path=os.path.join(path, m_name) if path is not None else None, **kwargs)
        elif isinstance(data, dict) and mode is not None:
            self.models[mode].inspect_latent(data, path=os.path.join(path, m_name) if path is not None else None, **kwargs)
        elif isinstance(data, dict) and mode is None:
            for m_name, d in data.items():
                self.models[m_name].inspect_latent(d, path=os.path.join(path, m_name) if path is not None else None, **kwargs)
        else:
            raise ValueError(f'Type of "data" not recognized: {data.__class__}')

    def sensitivity_analysis(self, data, mode:str, features:List[str]=None, path:str=None):
        if isinstance(data, dict) or isinstance(data, DataLoader):
            return self.models[mode].sensitivity_analysis(data, features, path=os.path.join(path, mode) if path is not None else None)
        else:
            raise ValueError(f'Type of "data" not supported: {data.__class__}')

    def __cross_modal_sensitivity_gradients_x(self, data, mode, x_name):
        data = {d_key: torch.Tensor(d_value).float().requires_grad_() for d_key, d_value in numpy_dict_to_tensor(data).items()}
        z = self.models[mode].encode(data, transform=False)['z']
        dec = self.decode_multimodal(data, mode, z, transform=False)
        for m in dec.keys():
            if x_name in dec[m].keys():
                dec[m][x_name].mean().backward(retain_graph=True)
        return torch_dict_to_numpy({x_key: data[x_key].grad[:, 0] for x_key in self.models[mode].dataset.x.keys()}), \
               torch_dict_to_numpy({y_key: data[y_key].grad[:, 0] for y_key in self.models[mode].dataset.y.keys()})

    def __cross_modal_sensitivity_gradients_y(self, data, mode, y_name):
        data = {d_key: torch.Tensor(d_value).float().requires_grad_() for d_key, d_value in numpy_dict_to_tensor(data).items()}
        enc = self.encode_multimodal(data, mode, transform=False)
        for m in enc.keys():
            if y_name in enc[m]['y'].keys():
                enc[m]['y'][y_name].mean().backward(retain_graph=True)
        return torch_dict_to_numpy({x_key: data[x_key].grad[:, 0] for x_key in self.models[mode].dataset.x.keys()})

    def cross_modal_sensitivity_analysis(self, data, modes:Union[str, List[str]]=None, features:List[str]=None, path:str=None):
        if features is None: 
            # If no specific feature is given, create sensitivities for all features
            features = list(self.dataset.x.keys()) + list(self.dataset.y.keys())
        elif isinstance(features, str):
            features = [features]

        if isinstance(modes, str):
            modes = [modes]

        if isinstance(data, dict):
            # if data is a dict, then the top-most keys must denote the modes for which data is provided
            if modes is None:
                modes = list(data.keys())
        elif isinstance(data, DataLoader) and modes:
            unr_data = torch_dict_to_numpy(rec_concat_dict([batch for batch in data]))
            data = {modes[0]: unr_data}
        elif isinstance(data, CombinedLoader):
            if modes is None:
                raise ValueError('If "data" is a CombinedLoader, "modes" must be provided.')
            unr_data = [torch_dict_to_numpy(rec_concat_dict(feature_batches)) for feature_batches in zip(*[batch for batch in data])]
            data = {m: d for m, d in zip(modes, unr_data)}
        else:
            raise ValueError(f'Type of "data" not supported: {data.__class__}')

        data_transformed = {m: self.models[m].dataset.transform({key: data[m][key] for key in (self.models[m].dataset.x | self.models[m].dataset.y).keys()}) for m in modes}
        
        for mode in modes:
            for feature in features:
                if feature in self.dataset.x.keys():
                    x_wrt_x, x_wrt_y = self.__cross_modal_sensitivity_gradients_x(data_transformed[mode], mode, feature)
                    swarm_plot(
                        features=x_wrt_x, 
                        data_hue=data[mode].get(feature, None), 
                        title=feature, 
                        y_label='Input features (' + mode + ')', 
                        path=os.path.join(path, 'dx') if path is not None else None,
                    )
                    swarm_plot(
                        features=x_wrt_y, 
                        data_hue=data[mode].get(feature, None), 
                        title=feature, 
                        y_label='Conditional features (' + mode + ')', 
                        path=os.path.join(path, 'dx') if path is not None else None,
                    )
                else:
                    y_wrt_x = self.__cross_modal_sensitivity_gradients_y(data_transformed[mode], mode, feature)
                    swarm_plot(
                        features=y_wrt_x, 
                        data_hue=data[mode].get(feature, None), 
                        title=feature, 
                        y_label='Input features (' + mode + ')', 
                        path=os.path.join(path, 'dy') if path is not None else None,
                    )


    def summary(self, **kwargs):
        for m in self.models.values():
            m.summary(**kwargs)

    def save(self, path:str):
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.get_config(), f, pickle.HIGHEST_PROTOCOL)
        for mode, model in self.models.items():
            model.save(os.path.join(path, mode))

    @classmethod
    def load_model(cls, path:str):
        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        modes = [mode_path for mode_path in os.listdir(path) if os.path.isdir(os.path.join(path, mode_path))]
        models = {mode: CondAEModel.load_model(os.path.join(path, mode)) for mode in modes}
        obj = MultiModalCondAEModel({mode: model.dataset for mode, model in models.items()}, **config)
        obj.models = models
        return obj