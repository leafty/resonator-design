import torch

from torch import nn
from typing import Dict, List, Union
from models.blocks import ResBlockFC

class Encoder(nn.Module):
    def __init__(self, in_heads:Dict[str, nn.Module], out_heads:Dict[str, nn.Module], layer_widths:List[int], latent_dim:int, activation:str):
        """
        Initializes the encoder module of a conditional variational autoencoder.

        Parameters
        ----------
        in_heads : Dict[str, nn.Module]
            Dictionary of input heads, where the keys are strings and the values are PyTorch modules.
        out_heads : Dict[str, nn.Module]
            Dictionary of output heads, where the keys are strings and the values are PyTorch modules.
        layer_widths : List[int]
            List of integer values representing the widths of the hidden layers in the encoder.
        latent_dim : int
            Integer value representing the latent dimension of the model.
        activation : str
            String representing the activation function to be used in the hidden layers of the encoder.
        """
        super().__init__()
        self.in_heads = nn.ModuleDict(in_heads)
        self.out_heads = nn.ModuleDict(out_heads)

        in_channels = sum([head.channels[-1] for head in in_heads.values()])
        self.channels = [in_channels] + layer_widths
        self.blocks = [ResBlockFC(c, c_next, activation=activation) for c, c_next in zip(self.channels[:-1], self.channels[1:])]
        self.seq_blocks = nn.Sequential(*self.blocks)
        self.fc_z = nn.Linear(self.channels[-1], latent_dim)

    def get_custom_modules(self) -> dict:
        return dict()

    def set_custom_modules(self, modules:dict) -> None:
        pass

    def forward(self, x, return_lat:bool=False):
        """
        Forward pass of the encoder.

        Parameters
        ----------
        x : Dict[str, torch.Tensor]
            Dictionary of input tensors, where the keys are strings and the values are PyTorch tensors.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of output tensors, where the keys are strings and the values are PyTorch tensors.
        """
        emb = torch.cat([head(x[name]) for name, head in self.in_heads.items()], dim=-1)
        lat = self.seq_blocks(emb)
        z = self.fc_z(lat)
        y = {name: head(lat) for name, head in self.out_heads.items()}

        out = {'z': z, 'y': y}
        if return_lat:
            out['lat'] = lat
        return out

    def summary(self, detail:int=1) -> None:
        """
        Prints a summary of a PyTorch model, including the number of parameters, the layers, their names, and the dimensionality.
        
        Parameters:
        detail (int, optional): Controls the level of detail in the summary. 
            Set to 1 to print the name and dimensionality of each layer. Defaults to 1.
        """
        total_params = sum(p.numel() for p in self.parameters())
        print(f"    Number of parameters: {total_params}")
        print("    Layers:")
        for name, layer in self.named_children():
            print(f"      {name}: {layer.__class__.__name__}")
            if detail > 0:
                for param_name, param in layer.named_parameters():
                    print(f"        {param_name}: {param.shape}")


class VEncoder(Encoder):
    def __init__(self, in_heads:Dict[str, nn.Module], out_heads:Dict[str, nn.Module], layer_widths:List[int], latent_dim:int, activation:str):
        """
        Initializes the encoder module of a conditional variational autoencoder with additional parameters for the latent distribution.

        Parameters
        ----------
        in_heads : Dict[str, nn.Module]
            Dictionary of input heads, where the keys are strings and the values are PyTorch modules.
        out_heads : Dict[str, nn.Module]
            Dictionary of output heads, where the keys are strings and the values are PyTorch modules.
        layer_widths : List[int]
            List of integer values representing the widths of the hidden layers in the encoder.
        latent_dim : int
            Integer value representing the latent dimension of the model.
        activation : str
            String representing the activation function to be used in the hidden layers of the encoder.
        """
        super().__init__(in_heads, out_heads, layer_widths, latent_dim, activation)
        self.fc_z_log_var = nn.Linear(self.channels[-1], latent_dim)

    def get_custom_modules(self) -> dict:
        return super().get_custom_modules() | {'fc_z_log_var': self.fc_z_log_var}

    def set_custom_modules(self, modules:dict) -> None:
        super().set_custom_modules(modules)
        self.fc_z_log_var = modules.get('fc_z_log_var', self.fc_z_log_var)

    def forward(self, x, return_lat:bool=False):
        """
        Forward pass of the encoder.

        Parameters
        ----------
        x : Dict[str, torch.Tensor]
            Dictionary of input tensors, where the keys are strings and the values are PyTorch tensors.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of output tensors, where the keys are strings and the values are PyTorch tensors.
        """
        emb = torch.cat([head(x[name]) for name, head in self.in_heads.items()], dim=-1)
        lat = self.seq_blocks(emb)
        z_mean, z_log_var = self.fc_z(lat), self.fc_z_log_var(lat)
        z = torch.distributions.Normal(z_mean, torch.exp(z_log_var / 2)).rsample()
        y = {name: head(lat) for name, head in self.out_heads.items()}

        out = {'z': z, 'y': y, 'z_mean': z_mean, 'z_log_var': z_log_var}
        if return_lat:
            out['lat'] = lat
        return out

class DiscreteEncoder(nn.Module):
    def __init__(self, encoder:Union[Encoder, VEncoder], disc_latent_dims:List[int], tau:float=1.):
        """
        Initializes the encoder module of a conditional variational autoencoder with additional parameters for the latent distribution.

        Parameters
        ----------
        in_heads : Dict[str, nn.Module]
            Dictionary of input heads, where the keys are strings and the values are PyTorch modules.
        disc_latent_dims
        """
        super().__init__()
        self.encoder = encoder
        self.in_heads, self.out_heads = self.encoder.in_heads, self.encoder.out_heads
        self.fc_a = nn.ModuleList([nn.Linear(self.encoder.channels[-1], latent_dim) for latent_dim in disc_latent_dims])
        self.tau = tau

    def get_custom_modules(self) -> dict:
        return self.encoder.get_custom_modules() | {'fc_a': self.fc_a}

    def set_custom_modules(self, modules:dict) -> None:
        self.encoder.set_custom_modules(modules)
        self.fc_a = modules.get('fc_a', self.fc_a)

    def forward(self, x, return_lat:bool=False):
        """
        Forward pass of the encoder.

        Parameters
        ----------
        x : Dict[str, torch.Tensor]
            Dictionary of input tensors, where the keys are strings and the values are PyTorch tensors.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of output tensors, where the keys are strings and the values are PyTorch tensors.
        """
        out = self.encoder(x, return_lat=True)
        alphas = [fc_a(out['lat']) for fc_a in self.fc_a]
        alphas = [torch.nn.functional.gumbel_softmax(alpha, hard=False, tau=self.tau) for alpha in alphas]

        out['z'] = torch.concat([out['z']] + alphas, dim=-1)
        for i, alpha in enumerate(alphas):
            out['alpha_' + str(i)] = alpha

        if not return_lat:
            out.pop('lat')
        return out
