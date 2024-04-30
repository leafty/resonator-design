import torch

from torch import nn
from src.learning_model.models.blocks import ResBlockFC, ResBlock1D, ResBlock2D, ResBlock3D

SCALING_FACTOR = 2

class InHeadFC(nn.Module):
    def __init__(self, in_channels:int, latent_dims:list, activation:str):
        super().__init__()
        self.channels = [in_channels] + latent_dims
        self.blocks = [ResBlockFC(c, c_next, activation=activation) for c, c_next in zip(self.channels[:-1], self.channels[1:])]
        self.seq_blocks = nn.Sequential(*self.blocks)
        self.activation = activation

    def forward(self, x):
        return self.seq_blocks(x)

class OutHeadFC(nn.Module):
    def __init__(self, in_channels:int, in_head:InHeadFC):
        super().__init__()
        self.channels = [in_channels] + in_head.channels[::-1]
        self.activations = [in_head.activation] * max(0, len(in_head.channels) - 2) + [None]
        self.blocks = [ResBlockFC(c, c_next, activation=a) for c, c_next, a in zip(self.channels[:-3], self.channels[1:-2], self.activations)]
        self.seq_blocks = nn.Sequential(*self.blocks)
        self.fc = nn.Linear(self.channels[-2], self.channels[-1])

    def forward(self, x):
        return self.fc(self.seq_blocks(x))
        

class InHeadConv1D(nn.Module):
    def __init__(self, in_channels:int, latent_dims:list, activation:str):
        super().__init__()        
        self.channels = [in_channels] + latent_dims
        self.blocks = [ResBlock1D(c, c_next, activation=activation) for c, c_next in zip(self.channels[:-1], self.channels[1:])]
        self.seq_blocks = nn.Sequential(*self.blocks)
        self.global_pool = LeakyGlobalPool1D()
        self.latent_shape = None

    def forward(self, x):
        if self.latent_shape is None:
            self.latent_shape = self.seq_blocks(x).shape
        return self.global_pool(self.seq_blocks(x))


class OutHeadConv1D(nn.Module):
    def __init__(self, in_channels:int, in_head:InHeadConv1D):
        super().__init__()
        self.in_head = in_head

        self.channels = [in_channels] + in_head.channels[1:][::-1]
        self.activations = [in_head.activation] * max(0, len(in_head.latent_dims) - 2) + [None]
        self.blocks = [ResBlock1D(c, c_next, activation=a) for c, c_next, a in zip(self.channels[:-3], self.channels[1:-2], self.activations)]
        self.seq_blocks = nn.Sequential(*self.blocks)
        self.out = nn.Conv1d(self.channels[-2], self.channels[-1], kernel_size=1)
        self.reshape_layer = None

    def _init_reshape_layer(self):
        if self.channels[0] == torch.tensor(self.in_head.latent_shape[1:]).prod():
            self.reshape_layer = nn.Sequential(nn.Unflatten(1, self.in_head.latent_shape[1:]))
        else:
            self.reshape_layer = nn.Sequential(nn.Linear(self.channels[0], torch.tensor(self.in_head.latent_shape[1:]).prod()),
                                     nn.Unflatten(1, self.in_head.latent_shape[1:]))

    def forward(self, x):
        if self.reshape_layer is None:
            self._init_reshape_layer()
        x = self.reshape_layer(x)
        x = self.seq_blocks(x)
        return self.out(x)


class InHeadConv2D(nn.Module):
    def __init__(self, in_channels:int, latent_dims:list, activation:str):
        super().__init__()
        self.channels = [in_channels] + latent_dims
        self.blocks = nn.Sequential(*[
            ResBlock2D(c, c_next, scaling='down', activation=activation) for c, c_next in zip(self.channels[:-1], self.channels[1:])
        ])
        self.global_pool = LeakyGlobalPool2D()
        self.latent_shape = None

    def forward(self, x):
        if self.latent_shape is None:
            self.latent_shape = self.seq_blocks(x).shape
        return self.global_pool(self.blocks(x))


class OutHeadConv2D(nn.Module):
    def __init__(self, in_channels:int, latent_dims:list, target_shape:tuple, activation:str):
        super().__init__()
        self.one_d_dim = target_shape[1] // (len(latent_dims) * SCALING_FACTOR)
        self.two_d_dim = target_shape[2] // (len(latent_dims) * SCALING_FACTOR)

        if in_channels % (self.one_d_dim * self.two_d_dim) == 0:
            self.reshaped_channels = in_channels // (self.one_d_dim * self.two_d_dim)
            self.emb = nn.Sequential()
        else:
            self.reshaped_channels = in_channels // (self.one_d_dim * self.two_d_dim) + 1
            self.reshaped_channels = in_channels * (in_channels // (self.one_d_dim * self.two_d_dim) + 1)
            self.emb = nn.Sequential(nn.Linear(in_channels, self.reshaped_channels))

        self.channels = [in_channels] + latent_dims
        self.blocks = nn.Sequential(*[
            ResBlock2D(c, c_next, scaling='up', activation=activation) for c, c_next in zip(self.channels[:-1], self.channels[1:])
        ])
        self.out = nn.Conv2d(latent_dims[-1], latent_dims[-1], kernel_size=1)

    def forward(self, x):
        x = self.emb(x)
        x = x.reshape((-1, self.one_d_dim, self.two_d_dim, self.reshaped_channels))
        x = self.blocks(x)
        return self.out(x)


class InHeadConv3D(nn.Module):
    def __init__(self, in_channels:int, latent_dims:list, activation:str):
        super().__init__()
        self.channels = [in_channels] + latent_dims
        self.blocks = nn.Sequential(*[
            ResBlock3D(c, c_next, scaling='down', activation=activation) for c, c_next in zip(self.channels[:-1], self.channels[1:])
        ])
        self.global_pool = LeakyGlobalPool3D()

    def forward(self, x):
        return self.global_pool(self.blocks(x))
        

class OutHeadConv3D(nn.Module):
    def __init__(self, in_channels:int, latent_dims:list, target_shape:tuple, activation:str):
        super().__init__()
        self.one_d_dim = target_shape[1] // (len(latent_dims) * SCALING_FACTOR)
        self.two_d_dim = target_shape[2] // (len(latent_dims) * SCALING_FACTOR)
        self.three_d_dim = target_shape[3] // (len(latent_dims) * SCALING_FACTOR)

        if in_channels % (self.one_d_dim * self.two_d_dim * self.three_d_dim) == 0:
            self.reshaped_channels = in_channels // (self.one_d_dim * self.two_d_dim)
            self.emb = nn.Sequential()
        else:
            self.reshaped_channels = in_channels // (self.one_d_dim * self.two_d_dim * self.three_d_dim) + 1
            self.reshaped_channels = in_channels * (in_channels // (self.one_d_dim * self.two_d_dim * self.three_d_dim) + 1)
            self.emb = nn.Sequential(nn.Linear(in_channels, self.reshaped_channels))

        self.channels = [in_channels] + latent_dims
        self.blocks = nn.Sequential(*[
            ResBlock3D(c, c_next, scaling='up', activation=activation) for c, c_next in zip(self.channels[:-1], self.channels[1:])
        ])
        self.out = nn.Conv3d(latent_dims[-1], latent_dims[-1], kernel_size=1)

    def forward(self, x):
        x = self.emb(x)
        x = x.reshape((-1, self.one_d_dim, self.two_d_dim, self.three_d_dim, self.reshaped_channels))
        x = self.blocks(x)
        return self.out(x)


class LeakyGlobalPool1D(nn.Module):
    def __init__(self, alpha:float=0.01):
        self.alpha = alpha

    def forward(self, x):
        return x.max(axis=-2) + self.alpha * x.mean(axis=-2)


class LeakyGlobalPool2D(nn.Module):
    def __init__(self, alpha:float=0.01):
        self.alpha = alpha

    def forward(self, x):
        return x.max(axis=-3).max(axis=-2) + self.alpha * x.mean(axis=-3).mean(axis=-2)


class LeakyGlobalPool3D(nn.Module):
    def __init__(self, alpha:float=0.01):
        self.alpha = alpha

    def forward(self, x):
        return x.max(axis=-4).max(axis=-3).max(axis=-2) + self.alpha * x.mean(axis=-4).mean(axis=-3).mean(axis=-2)
