import torch

from torch import nn
from typing import Dict, List
from models.blocks import ResBlockFC

class Decoder(nn.Module):
    def __init__(self, in_heads:Dict[str, nn.Module], out_heads:Dict[str, nn.Module], layer_widths:List[int], latent_dim:int, activation:str):
        """
        Initializes the decoder module of the conditional (variational) autoencoder.

        Parameters
        ----------
        in_heads : Dict[str, nn.Module]
            Dictionary of input heads, where the keys are strings and the values are PyTorch modules.
        out_heads : Dict[str, nn.Module]
            Dictionary of output heads, where the keys are strings and the values are PyTorch modules.
        layer_widths : List[int]
            List of integer values representing the widths of the hidden layers in the decoder.
        latent_dim : int
            Integer value representing the latent dimension of the model.
        activation : str
            String representing the activation function to be used in the hidden layers of the decoder.
        """
        super().__init__()
        self.in_heads = nn.ModuleDict(in_heads)
        self.out_heads = nn.ModuleDict(out_heads)

        self.channels = [latent_dim + sum([head.channels[-1] for head in in_heads.values()])] + layer_widths
        self.blocks = [ResBlockFC(c, c_next, activation=activation) for c, c_next in zip(self.channels[:-1], self.channels[1:])]
        self.seq_blocks = nn.Sequential(*self.blocks)

    def get_custom_modules(self) -> dict:
        return dict()

    def set_custom_modules(self, modules:dict) -> None:
        pass

    def forward(self, inputs:Dict[str, Dict[str, torch.Tensor]], return_lat:bool=False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the decoder.

        Parameters
        ----------
        inputs : Dict[str, Dict[str, torch.Tensor]]
            Dictionary of input tensors, where the keys are strings and the values are PyTorch tensors.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of output tensors, where the keys are strings and the values are PyTorch tensors.
        """
        z, y = inputs['z'], inputs['y']

        lat = self.seq_blocks(torch.cat([z] + [head(y[name]) for name, head in self.in_heads.items()], dim=1))

        out = {'x': {name: head(lat) for name, head in self.out_heads.items()}}
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