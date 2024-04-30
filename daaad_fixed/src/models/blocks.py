import torch

from torch import nn
from typing import Tuple, Union

SCALING_FACTOR = 2
CONV_2_KERNEL_SIZE = 5
CONV_2_PADDING = CONV_2_KERNEL_SIZE // 2
DIM_CLASSES = {
    1: (nn.Conv1d, nn.BatchNorm1d, nn.ReplicationPad1d, 'linear', nn.MaxPool1d),
    2: (nn.Conv2d, nn.BatchNorm2d, nn.ReplicationPad2d, 'bilinear', nn.MaxPool2d),
    3: (nn.Conv3d, nn.BatchNorm3d, nn.ReplicationPad3d, 'trilinear', nn.MaxPool3d),
}

class ResBlockFC(nn.Module):
    """
    Mdule implementing a fully-connected residual block.
    """
    def __init__(self, in_channels:int, out_channels:int, activation:Union[nn.Module, str]='leaky_relu', dropout:bool=True, batchnorm:bool=True, residual:bool=True):
        """
        Initialize the ResBlockFC object.

        Parameters:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        activation (Union[nn.Module, str]): The activation function to use (default: 'leaky_relu').
        dropout (bool): Whether to use dropout or not (default: True).
        batchnorm (bool): Whether to use batch normalization or not (default: True).
        residual (bool): Whether to use a residual connection or not (default: True).
        """
        super().__init__()
        self.batchnorm, self.dropout, self.residual = batchnorm, dropout, residual
        self.fc1 = nn.Linear(in_channels, out_channels)
        if dropout:
            self.dr = nn.Dropout1d(p=0.1)
        if batchnorm:
            self.bn = nn.BatchNorm1d(out_channels)
        if residual:
            self.fc2 = nn.Linear(out_channels, out_channels)
        self.af = Activation(activation)

    def forward(self, x):
        x = self.fc1(x)
        l = self.af(x)
        if self.dropout:
            l = self.dr(l)
        if self.batchnorm:
            l = self.bn(l)
        if self.residual:
            l = self.fc2(l)
            return x + l
        else:
            return l

class ResBlockConv(nn.Module):
    """
    Module implementing a convolutional residual block.
    """
    def __init__(self, dim:int, in_channels:int,  out_channels:int, scaling:str, activation:Union[nn.Module, str], up_padding:Tuple[int], dropout:bool):
        """
        Initialize the ResBlockConv object.

        Parameters:
        dim (int): The number of dimensions of the input data (1, 2, or 3).
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        scaling (str): The scaling mode to use. 
            One of 'down' (reduce dimensionality), 'up' (increase dimensionality), or None (no change in dimensionality).
        activation (Union[nn.Module, str]): The activation function to use.
        up_padding (Tuple[int]): The padding to use for upsampling. If shape was uneven when encoding (downsampling), tensor must
            also be cropped to uneven when decoding (upsampling).
        dropout (bool): Whether to use dropout or not.
        """
        super().__init__()
        self.dim = dim
        self.conv_class, self.batchnorm_class, self.replicationpad_class, self.ups_mode, _ = DIM_CLASSES[self.dim]
        self.scaling = scaling
        self.bn1 = self.batchnorm_class(out_channels)
        self.af = Activation(activation)
        self.up_padding = up_padding
        self.dropout = dropout
        if self.dropout:
            self.dr = nn.Dropout(p=0.1)

        if scaling in ['down']:
            # downsampling matrix
            self.conv1 = self.conv_class(in_channels, out_channels, kernel_size=3, stride=SCALING_FACTOR, padding=1, padding_mode='replicate')
            self.conv2 = self.conv_class(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate')
            self.shortcut = nn.Sequential(
                self.conv_class(in_channels, out_channels, kernel_size=1, stride=SCALING_FACTOR),
            )
        elif scaling in ['up']:
            # upsampling matrix
            self.conv1 = nn.Sequential(
                self.conv_class(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
                nn.Upsample(scale_factor=SCALING_FACTOR, mode=self.ups_mode)
            )
            self.conv2 = nn.Sequential(
                self.conv_class(out_channels, out_channels, kernel_size=CONV_2_KERNEL_SIZE),
                self.replicationpad_class(self.up_padding)
            )
            self.shortcut = nn.Sequential(
                nn.Upsample(scale_factor=SCALING_FACTOR),
                self.conv_class(in_channels, out_channels, kernel_size=1),
            )
        else:
            self.conv1 = self.conv_class(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate')
            self.shortcut = nn.Sequential()

    def forward(self, x0):
        shortcut = self.shortcut(x0)
        # necessary for upsampling: if shape was uneven when encoding, tensor must
        # also be cropped to uneven when decoding
        shortcut = nn.functional.pad(shortcut, [-int(p == (CONV_2_PADDING - 1)) for p in self.up_padding])
        
        x = self.af(self.conv1(x0))
        if self.dropout:
            x = self.dr(x)
            x = self.bn1(x)
        x = self.af(self.conv2(x))
        x = x + shortcut
        return x


class ResBlock1D(ResBlockConv):
    """
    Module implementing a 1D convolutional residual block.
    """
    def __init__(self, in_channels:int, out_channels:int, scaling:str=None, activation:Union[nn.Module, str]='leaky_relu', up_padding:Tuple[int]=(2, 2), dropout:bool=True):
        """
        Initialize the ResBlock1D object.

        Parameters:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        scaling (str): The scaling mode to use (default: None). 
            One of 'down' (reduce dimensionality), 'up' (increase dimensionality), or None (no change in dimensionality).
        activation (Union[nn.Module, str]): The activation function to use (default: 'leaky_relu').
        up_padding (Tuple[int]): The padding to use for upsampling (default: (2, 2)).
        dropout (bool): Whether to use dropout or not.
        """
        super().__init__(1, in_channels, out_channels, scaling, activation, up_padding, dropout)

        
class ResBlock2D(ResBlockConv):
    """
    Module implementing a 2D convolutional residual block.
    """
    def __init__(self, in_channels:int, out_channels:int, scaling:str=None, activation:Union[nn.Module, str]='leaky_relu', up_padding:Tuple[int]=[2]*4, dropout:bool=True):
        """
        Initialize the ResBlock2D object.

        Parameters:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        scaling (str): The scaling mode to use (default: None). 
            One of 'down' (reduce dimensionality), 'up' (increase dimensionality), or None (no change in dimensionality).
        activation (Union[nn.Module, str]): The activation function to use (default: 'leaky_relu').
        up_padding (Tuple[int]): The padding to use for upsampling (default: (2, 2, 2, 2)).
        dropout (bool): Whether to use dropout or not.
        """
        super().__init__(2, in_channels, out_channels, scaling, activation, up_padding, dropout)


class ResBlock3D(ResBlockConv):
    """
    Module implementing a 3D convolutional residual block.
    """
    def __init__(self, in_channels:int, out_channels:int, scaling:str=None, activation:Union[nn.Module, str]='leaky_relu', up_padding:Tuple[int]=[2]*6, dropout:bool=False):
        """
        Initialize the ResBlock3D object.

        Parameters:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        scaling (str): The scaling mode to use (default: None). 
            One of 'down' (reduce dimensionality), 'up' (increase dimensionality), or None (no change in dimensionality).
        activation (Union[nn.Module, str]): The activation function to use (default: 'leaky_relu').
        up_padding (Tuple[int]): The padding to use for upsampling (default: (2, 2, 2, 2, 2, 2)).
        dropout (bool): Whether to use dropout or not.
        """
        super().__init__(3, in_channels, out_channels, scaling, activation, up_padding, dropout)


class SelfAttn(nn.Module):
    """
    Module implementing a self-attention layer.
    """
    def __init__(self, dim:int, in_channels:int, eps:float=1e-12):
        """
        Initialize the SelfAttn object.

        Parameters:
        dim (int): The dimensionality of the input data (1, 2, or 3).
        in_channels (int): The number of input channels.
        eps (float): The epsilon value for the spectral normalization (default: 1e-12).
        """
        super(SelfAttn, self).__init__()
        self.dim = dim
        self.conv_class, _, _, _, _, self.maxpool_class = DIM_CLASSES[self.dim]
        self.in_channels = in_channels

        self.snconv1x1_theta = nn.utils.spectral_norm(
            self.conv_class(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, bias=False), 
            eps=eps
        )
        self.snconv1x1_phi = nn.utils.spectral_norm(
            self.conv_class(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, bias=False), 
            eps=eps
        )
        self.snconv1x1_g = nn.utils.spectral_norm(
            self.conv_class(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, bias=False), 
            eps=eps
        )
        self.snconv1x1_o_conv = nn.utils.spectral_norm(
            self.conv_class(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, bias=False), 
            eps=eps
        )

        self.positional_encoding = nn.parameter.Parameter(torch.normal(2, 3, size=(in_channels,)), requires_grad=True)
        self.maxpool = self.maxpool_class(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        batch_size, ch, dims = x.shape[0], x.shape[1], list(x.shape[2:])

        x_pos = x + torch.reshape(self.positional_encoding, (1, -1) + (1,) * len(dims))

        # Theta path
        theta = self.snconv1x1_theta(x_pos)
        theta = theta.view(batch_size, ch//8, -1)

        # Phi path
        phi = self.snconv1x1_phi(x_pos)
        phi = self.maxpool(phi)
        phi = phi.view(batch_size, ch//8, -1)

        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)

        # g path
        g = self.snconv1x1_g(x_pos)
        g = self.maxpool(g)
        g = g.view(batch_size, ch//2, -1)

        # Attn_g - o_conv
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(batch_size, ch//2, *dims)
        attn_g = self.snconv1x1_o_conv(attn_g)

        return x + self.gamma * attn_g

class SelfAttn1D(SelfAttn):
    """
    Module implementing a 1D self-attention layer.
    """
    def __init__(self, in_channels:int, eps:float=1e-12):
        """
        Initialize the SelfAttn1D object.

        Parameters:
        in_channels (int): The number of input channels.
        eps (float): The epsilon value for the spectral normalization (default: 1e-12).
        """
        super().__init__(1, in_channels, eps)

class SelfAttn2D(SelfAttn):
    """
    Module implementing a 2D self-attention layer.
    """
    def __init__(self, in_channels:int, eps:float=1e-12):
        """
        Initialize the SelfAttn2D object.

        Parameters:
        in_channels (int): The number of input channels.
        eps (float): The epsilon value for the spectral normalization (default: 1e-12).
        """
        super().__init__(2, in_channels, eps)

class SelfAttn3D(SelfAttn):
    """
    Module implementing a 3D self-attention layer.
    """
    def __init__(self, in_channels:int, eps:float=1e-12):
        """
        Initialize the SelfAttn3D object.

        Parameters:
        in_channels (int): The number of input channels.
        eps (float): The epsilon value for the spectral normalization (default: 1e-12).
        """
        super().__init__(3, in_channels, eps)


class Activation(nn.Module):
    """
    Module wrapping around an activation function. 
    It allows the user to specify an activation function as a string or as a PyTorch module. 
    If a string is provided, the Activation class will initialize the corresponding PyTorch module. 
    If a PyTorch module is provided, the Activation class will simply use that module as the activation function. 
    """
    def __init__(self, activation:Union[nn.Module, str]):
        """
        Initialize the Activation object.

        Parameters:
        activation (Union[nn.Module, str]): The activation function to use. This can be specified as a string
                                            representing one of the following activation functions: 'linear', 
                                            'relu', 'leaky_relu', 'swish', 'elu', 'sigmoid', 'tanh', 'softmax',
                                            'log_softmax'. Alternatively, a PyTorch activation module can be
                                            provided directly.
        """
        super().__init__()
        if isinstance(activation, str) or activation is None:
            if activation is None or activation == 'linear':
                self.activation = nn.Identity()
            elif activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'leaky_relu':
                self.activation = nn.LeakyReLU()
            elif activation == 'swish':
                self.activation = nn.Swish()
            elif activation == 'elu':
                self.activation = nn.ELU()
            elif activation == 'sigmoid':
                self.activation = nn.Sigmoid()
            elif activation == 'tanh':
                self.activation = nn.Tanh()
            elif activation == 'softmax':
                self.activation = nn.Softmax(dim=-1)
            elif activation == 'log_softmax':
                self.activation = nn.LogSoftmax(dim=-1)
            else:
                raise ValueError(f'{activation} not in the list of valid activation functions')
        elif isinstance(activation, nn.Module):
            self.activation = activation
        else:
            raise ValueError(f'Object type of activation not understood. \
                Should be nn.Module or str, but was {activation.__class__}')
    
    def forward(self, x):
        return self.activation(x)