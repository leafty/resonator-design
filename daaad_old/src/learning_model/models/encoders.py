from typing import Dict, List
import torch

from torch import nn
from torchsummary import summary
from src.learning_model.models.blocks import ResBlockFC

class Encoder(nn.Module):
    def __init__(self, in_heads:Dict[str, nn.Module], out_heads:Dict[str, nn.Module], layer_widths:List[int], latent_dim:int, activation:str):
        super().__init__()
        for name, head in (in_heads | out_heads).items():
            self.add_module(name, head)
        self.in_heads = in_heads
        self.out_heads = out_heads

        in_channels = sum([head.channels[-1] for head in in_heads.values()])
        self.channels = [in_channels] + layer_widths
        self.blocks = [ResBlockFC(c, c_next, activation=activation) for c, c_next in zip(self.channels[:-1], self.channels[1:])]
        self.seq_blocks = nn.Sequential(*self.blocks)
        self.fc_z = nn.Linear(self.channels[-1], latent_dim)

    def forward(self, x):
        emb = torch.cat([getattr(self, name)(x[name]) for name in self.in_heads.keys()], dim=-1)
        lat = self.seq_blocks(emb)
        z = self.fc_z(lat)
        y = {name: getattr(self, name)(lat) for name in self.out_heads.keys()}
        return {'z': z, 'y': y}

    def summary(self, input_data:dict, **kwargs):
        for head_name, head in self.in_heads.items():
            print('\nIN HEAD', head_name)
            summary(head, input_data=input_data[head_name], **kwargs)
        for head_name, head in self.out_heads.items():
            print('\nOUT HEAD', head_name)
            summary(head, input_data=(self.channels[-1],), **kwargs)
        # print('\nCORE')
        # summary(self, (self.channels[0],), **kwargs)

class VEncoder(Encoder):
    def __init__(self, in_heads:Dict[str, nn.Module], out_heads:Dict[str, nn.Module], layer_widths:List[int], latent_dim:int, activation:str):
        super().__init__(in_heads, out_heads, layer_widths, latent_dim, activation)
        self.fc_z_log_var = nn.Linear(self.channels[-1], latent_dim)

    def forward(self, x):
        emb = torch.cat([getattr(self, name)(x[name]) for name in self.in_heads.keys()], dim=-1)
        lat = self.seq_blocks(emb)
        z_mean, z_log_var = self.fc_z(lat), self.fc_z_log_var(lat)
        z = torch.distributions.Normal(z_mean, torch.exp(z_log_var / 2)).rsample()
        y = {name: getattr(self, name)(lat) for name in self.out_heads.keys()}
        return {'z': z, 'y': y, 'z_mean': z_mean, 'z_log_var': z_log_var}