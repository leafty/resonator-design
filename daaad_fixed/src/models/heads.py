import torch

from torch import nn
from typing import List, Tuple, Union
from models.blocks import Activation, ResBlockFC, ResBlock1D, ResBlock2D, ResBlock3D, CONV_2_PADDING, SelfAttn1D, SelfAttn2D, SelfAttn3D

SCALING_FACTOR = 2
MIN_SPATIAL_DIM = 3
MIN_PROD_SPATIAL_DIM = 64
DIM_CLASSES = {
    1: (ResBlock1D, SelfAttn1D, nn.Conv1d),
    2: (ResBlock2D, SelfAttn2D, nn.Conv2d),
    3: (ResBlock3D, SelfAttn3D, nn.Conv3d),
}


class InHeadFC(nn.Module):
    """
    Fully-Connected Feed-Forward Network for encoding an unstrucured, 1D feature.
    """
    def __init__(self, in_channels:int, latent_dims:List[int], activation:Union[nn.Module, str]):
        """
        Parameters:
        in_channels (int): The number of input channels.
        latent_dims (List[int]): A list of integers representing the dimensions of the latent space.
                                 The length of the sequence defines the number of layers.
        activation (Union[nn.Module, str]): The activation function to use.
        """
        super().__init__()
        self.channels = [in_channels] + latent_dims
        self.blocks = [ResBlockFC(c, c_next, activation=activation) for c, c_next in zip(self.channels[:-1], self.channels[1:])]
        self.seq_blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.seq_blocks(x)


class OutHeadFC(nn.Module):
    """
    Fully-Connected Feed-Forward Network for decoding an unstrucured, 1D feature.
    """
    def __init__(self, in_channels:int, latent_dims:List[int], activation:Union[nn.Module, str], out_activation:Union[nn.Module, str]=None):
        """
        Parameters:
        in_channels (int): The number of input channels.
        latent_dims (List[int]): A list of integers representing the dimensions of the latent space.
                                 The length of the sequence defines the number of layers.
        activation (Union[nn.Module, str]): The activation function to use.
        out_activation (Union[nn.Module, str]): The activation function to use in the last layer.
        """
        super().__init__()
        self.channels = [in_channels] + latent_dims
        self.blocks = [ResBlockFC(c, c_next, activation=activation) for c, c_next in zip(self.channels[:-2], self.channels[1:-1])]
        self.seq_blocks = nn.Sequential(*self.blocks)
        self.out_block = ResBlockFC(self.channels[-2], self.channels[-1], out_activation, dropout=False, batchnorm=False, residual=False)

    def forward(self, x):
        return self.out_block(self.seq_blocks(x))
        

class InHeadConv(nn.Module):
    """
    Convolutional Network for encoding spatially strucured data in 1, 2 or 3 dimensions.
    """
    def __init__(self, dim:int, in_shape:Tuple[int], latent_dims:List[int], activation:Union[nn.Module, str], attn_block_indices:List[int]):
        """
        Parameters:
        dim (int): The dimensionality of the data: 1, 2 or 3.
        in_shape (Tuple[int]): The number of input channels (in numpy style, i.e. (... spatial dimensions ..., channels)).
        latent_dims (List[int]): A list of integers representing the dimensions of the latent space.
                                 The length of the sequence defines the number of layers.
        activation (Union[nn.Module, str]): The activation function to use.
        attn_block_indices (List[int]): Sequence of indices where to insert an attention layer in the encoding blocks.
        """
        super().__init__()
        self.dim = dim
        self.in_shape = in_shape
        self.res_block_class, self.attn_block_class, _ = DIM_CLASSES[self.dim]

        self.attn_block_indices = attn_block_indices

        if latent_dims is None or len(latent_dims) == 0:
            latent_dims = self._auto_generate_latent_dims()

        self.channels = [self.in_shape[-1]] + latent_dims
        self.blocks = [
            self.res_block_class(c, c_next, scaling='down', activation=activation) if i not in self.attn_block_indices \
                else nn.Sequential(self.attn_block_class(c), self.res_block_class(c, c_next, scaling='down', activation=activation)) \
                    for i, (c, c_next) in enumerate(zip(self.channels[:-1], self.channels[1:]))
        ]
        self.seq_blocks = nn.Sequential(*self.blocks)
        self.global_pool = leaky_global_pool()

    def _auto_generate_latent_dims(self):
        """
            If no depth and widths of the CNN head are provided, this method adds layers with increasing width
            by multiplicative factor of 2 until either the product of all spatial dimensions is below
            MIN_PROD_SPATIAL_DIM, or the smallest dimension falls below MIN_SPATIAL_DIM.
        """
        channels = 8
        latent_dims = [channels]
        spatial_dims = list(self.in_shape[:-1])

        while torch.prod(torch.Tensor(spatial_dims)) >= MIN_PROD_SPATIAL_DIM and torch.Tensor(spatial_dims).min() >= MIN_SPATIAL_DIM:
            channels *=2
            latent_dims.append(channels)
            spatial_dims = [d // 2 for d in spatial_dims]
        
        return latent_dims

    def forward(self, x):
        return self.global_pool(self.seq_blocks(x.transpose(1, -1)))


class OutHeadConv(nn.Module):
    """
    Convolutional Network for decoding spatially strucured data in 1, 2 or 3 dimensions.
    """
    def __init__(self, dim:int, in_channels:int, target_shape:Tuple[int], latent_dims:List[int], activation:Union[nn.Module, str], out_activation:Union[nn.Module, str], attn_block_indices:List[int]):
        """
        Parameters:
        dim (int): The dimensionality of the data: 1, 2 or 3.
        in_channels (int): The number of input channels.
        target_shape (Tuple[int]): Final shape that the output data should have (in numpy style, i.e. (... spatial dimensions ..., channels)).
        latent_dims (List[int]): A list of integers representing the dimensions of the latent space.
                                 The length of the sequence defines the number of layers.
        activation (Union[nn.Module, str]): The activation function to use.
        out_activation (Union[nn.Module, str]): The activation function to use in the last layer.
        attn_block_indices (List[int]): Sequence of indices where to insert an attention layer in the encoding blocks.
        """
        super().__init__()
        self.dim = dim
        self.target_shape = target_shape
        self.res_block_class, self.attn_block_class, self.conv_class = DIM_CLASSES[self.dim]

        self.attn_block_indices = attn_block_indices

        if latent_dims is None or len(latent_dims) == 0:
            latent_dims = self._auto_generate_latent_dims()

        self.channels = latent_dims[:1] + latent_dims + [target_shape[-1]]
        self.paddings, self.latent_shape = self._get_paddings(target_shape, latent_dims)
        self.blocks = [
            self.res_block_class(c, c_next, scaling='up', activation=activation, up_padding=p) if i not in self.attn_block_indices \
                else nn.Sequential(self.attn_block_class(c), self.res_block_class(c, c_next, scaling='up', activation=activation, up_padding=p)) \
                    for i, (c, c_next, p) in enumerate(zip(self.channels[:-2], self.channels[1:-1], self.paddings))
        ]
        self.reshape_layer = self._create_reshape_layer(in_channels, self.latent_shape)
        self.seq_blocks = nn.Sequential(*self.blocks)
        self.out_conv = self.conv_class(self.channels[-2], self.channels[-1], kernel_size=1)
        self.out_af = Activation(out_activation)

    def _get_paddings(self, target_shape:Tuple[int], latent_dims:List[int]) -> List[List[int]]:
        """
            When performing downsampling on data with odd spatial dimensionality, the shape is divided by two and rounded down.
            Given the shape that the data should have after decoding, as well as the the number of layers in the head,
            this function determines, by how much each intermediate result must be cropped (negatively padded) to result
            in the original dimensionality.
        """
        paddings = []
        shape = target_shape[:-1]
        for _ in range(len(latent_dims)):
            paddings.append([CONV_2_PADDING if i % 2 == 0 or shape[i // 2] % 2 == 0 else CONV_2_PADDING - 1 for i in range(len(shape) * 2)])
            shape = [(s // 2) if s % 2 == 0 else (s // 2) + 1 for s in shape]
        return paddings[::-1], latent_dims[:1] + shape

    def _create_reshape_layer(self, in_channels:int, latent_shape:List[int]):
        """
            This method reshapes a one-dimensional vector into a shape that can be processed by this CNN head.
            If the vector does not have the right dimensionality to be directly reshaped, it is first passed 
            through a linear layer.
        """
        if in_channels == torch.tensor(latent_shape).prod():
            return nn.Sequential(nn.Unflatten(1, latent_shape))
        else:
            return nn.Sequential(nn.Linear(in_channels, torch.tensor(latent_shape).prod()),
                                 nn.Unflatten(1, latent_shape))

    def _auto_generate_latent_dims(self):
        """
            If no depth and widths of the CNN head are provided, this method adds layers with increasing width
            by multiplicative factor of 2 until either the product of all spatial dimensions is below
            MIN_PROD_SPATIAL_DIM, or the smallest dimension falls below MIN_SPATIAL_DIM.
        """
        channels = 8
        latent_dims = [channels]
        spatial_dims = list(self.target_shape[:-1])

        while torch.prod(torch.Tensor(spatial_dims)) >= MIN_PROD_SPATIAL_DIM and torch.Tensor(spatial_dims).min() >= MIN_SPATIAL_DIM:
            channels *= 2
            latent_dims.append(channels)
            spatial_dims = [d // 2 for d in spatial_dims]

        return latent_dims[::-1]
                                        
    def forward(self, zy):
        h = self.reshape_layer(zy)
        h = self.seq_blocks(h)
        return self.out_af(self.out_conv(h).transpose(1, -1))


class InHeadConv1D(InHeadConv):
    """
    Convolutional Network for encoding spatially strucured (temporal) data in 1 dimension.
    """
    def __init__(self, in_channels:int, latent_dims:List[int], activation:Union[nn.Module, str], attn_block_indices:List[int]=[]):
        """
        Parameters:
        in_channels (int): The number of input channels.
        latent_dims (List[int]): A list of integers representing the dimensions of the latent space.
                                 The length of the sequence defines the number of layers.
        activation (Union[nn.Module, str]): The activation function to use.
        attn_block_indices (List[int]): Sequence of indices where to insert an attention layer in the encoding blocks.
        """
        super().__init__(1, in_channels, latent_dims, activation, attn_block_indices)

class OutHeadConv1D(OutHeadConv):
    """
    Convolutional Network for decoding spatially strucured (temporal) data in 1 dimension.
    """
    def __init__(self, in_channels:int, target_shape:Tuple[int], latent_dims:List[int], activation:Union[nn.Module, str], out_activation:Union[nn.Module, str]=None, attn_block_indices:List[int]=[]):
        """
        Parameters:
        in_channels (int): The number of input channels.
        target_shape (Tuple[int]): Final shape that the output data should have.
        latent_dims (List[int]): A list of integers representing the dimensions of the latent space.
                                 The length of the sequence defines the number of layers.
        activation (Union[nn.Module, str]): The activation function to use.
        out_activation (Union[nn.Module, str]): The activation function to use in the last layer.
        attn_block_indices (List[int]): Sequence of indices where to insert an attention layer in the encoding blocks.
        """
        super().__init__(1, in_channels, target_shape, latent_dims, activation, out_activation, attn_block_indices)

class InHeadConv2D(InHeadConv):
    """
    Convolutional Network for encoding spatially strucured (image-like) data in 2 dimensions.
    """
    def __init__(self, in_channels:int, latent_dims:List[int], activation:Union[nn.Module, str], attn_block_indices:List[int]=[]):
        """
        Parameters:
        in_channels (int): The number of input channels.
        latent_dims (List[int]): A list of integers representing the dimensions of the latent space.
                                 The length of the sequence defines the number of layers.
        activation (Union[nn.Module, str]): The activation function to use.
        attn_block_indices (List[int]): Sequence of indices where to insert an attention layer in the encoding blocks.
        """
        super().__init__(2, in_channels, latent_dims, activation, attn_block_indices)

class OutHeadConv2D(OutHeadConv):
    """
    Convolutional Network for decoding spatially strucured (image-like) data in 2 dimensions.
    """
    def __init__(self, in_channels:int, target_shape:Tuple[int], latent_dims:List[int], activation:Union[nn.Module, str], out_activation:Union[nn.Module, str]=None, attn_block_indices:List[int]=[]):
        """
        Parameters:
        in_channels (int): The number of input channels.
        target_shape (Tuple[int]): Final shape that the output data should have.
        latent_dims (List[int]): A list of integers representing the dimensions of the latent space.
                                 The length of the sequence defines the number of layers.
        activation (Union[nn.Module, str]): The activation function to use.
        out_activation (Union[nn.Module, str]): The activation function to use in the last layer.
        attn_block_indices (List[int]): Sequence of indices where to insert an attention layer in the encoding blocks.
        """
        super().__init__(2, in_channels, target_shape, latent_dims, activation, out_activation, attn_block_indices)

class InHeadConv3D(InHeadConv):
    """
    Convolutional Network for encoding spatially strucured (MRI- or video-like) data in 3 dimensions.
    """
    def __init__(self, in_channels:int, latent_dims:List[int], activation:Union[nn.Module, str], attn_block_indices:List[int]=[]):
        """
        Parameters:
        in_channels (int): The number of input channels.
        latent_dims (List[int]): A list of integers representing the dimensions of the latent space.
                                 The length of the sequence defines the number of layers.
        activation (Union[nn.Module, str]): The activation function to use.
        attn_block_indices (List[int]): Sequence of indices where to insert an attention layer in the encoding blocks.
        """
        super().__init__(3, in_channels, latent_dims, activation, attn_block_indices)

class OutHeadConv3D(OutHeadConv):
    """
    Convolutional Network for decoding spatially strucured (MRI- or video-like) data in 3 dimensions.
    """
    def __init__(self, in_channels:int, target_shape:Tuple[int], latent_dims:List[int], activation:Union[nn.Module, str], out_activation:Union[nn.Module, str]=None, attn_block_indices:List[int]=[]):
        """
        Parameters:
        in_channels (int): The number of input channels.
        target_shape (Tuple[int]): Final shape that the output data should have.
        latent_dims (List[int]): A list of integers representing the dimensions of the latent space.
                                 The length of the sequence defines the number of layers.
        activation (Union[nn.Module, str]): The activation function to use.
        out_activation (Union[nn.Module, str]): The activation function to use in the last layer.
        attn_block_indices (List[int]): Sequence of indices where to insert an attention layer in the encoding blocks.
        """
        super().__init__(3, in_channels, target_shape, latent_dims, activation, out_activation, attn_block_indices)


def leaky_global_pool(alpha:float=0.1):
    """
    A function that performs a global pooling operation on the input tensor,
    combining the maximum and average (multiplied by a scaling factor) of all the spatial dimensions.
    Inspired by leaky relu activation supposed to improve the gradient flow.

    Parameters:
    x (torch.Tensor): The input tensor.
    alpha (float): The weight given to the average value (default: 0.1).

    Returns:
    torch.Tensor: The output tensor.
    """
    return lambda x: (1 - alpha) * x.amax(dim=list(range(2, 2+len(x.shape[2:])))) + alpha * x.mean(dim=list(range(2, 2+len(x.shape[2:]))))
