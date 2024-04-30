from typing import Dict, List
import torch

from torch import nn
from torchsummary import summary
from src.learning_model.models.blocks import ResBlockFC

class Decoder(nn.Module):
    def __init__(self, in_heads:Dict[str, nn.Module], out_heads:Dict[str, nn.Module], layer_widths:List[int], latent_dim:int, activation:str, guided:bool):
        super().__init__()
        for name, head in (in_heads | out_heads).items():
            self.add_module(name, head)
        self.in_heads = in_heads
        self.out_heads = out_heads
        self.guided = guided

        in_channels = sum([head.channels[-1] for head in in_heads.values()]) + latent_dim
        self.channels = [in_channels] + layer_widths
        self.blocks = [ResBlockFC(c, c_next, activation=activation) for c, c_next in zip(self.channels[:-1], self.channels[1:])]
        self.seq_blocks = nn.Sequential(*self.blocks)
        if guided:
            self.z_fc = nn.Linear(latent_dim, in_channels)

    def forward(self, inputs:Dict[str, torch.Tensor], cond_weights:List[float]=None):
        z, y = inputs['z'], inputs['y']
        if self.training:
            if not self.guided or cond_weights is None:
                emb = torch.cat([z] + [getattr(self, name)(y[name]) for name in self.in_heads.keys()], dim=-1)
            else:
                emb = self.z_fc(z)
            lat = self.seq_blocks(emb)
        else:
            if not self.guided or cond_weights is None:
                emb = torch.cat([z] + [getattr(self, name)(y[name]) for name in self.in_heads.keys()], dim=-1)
                lat = self.seq_blocks(emb)
            else:
                lat_unc = self.z_fc(z)
                lat = torch.cat([z] + [getattr(self, name)(y[name]) for name in self.in_heads.keys()], dim=-1)
                for i, b in enumerate(self.blocks):
                    lat_unc = b(lat_unc)
                    lat = (1 + cond_weights[i]) * b(lat) - cond_weights[i] * lat_unc
        return {'x': {name: getattr(self, name)(lat) for name in self.out_heads.keys()}}

    def summary(self, input_data:dict, **kwargs):
        for head_name, head in self.in_heads.items():
            print('\nIN HEAD', head_name)
            summary(head, input_data[head_name], **kwargs)
        for head_name, head in self.out_heads.items():
            print('\nOUT HEAD', head_name)
            summary(head, (self.channels[-1],), **kwargs)
        # print('\nCORE')
        # summary(self, (self.channels[0],), **kwargs)