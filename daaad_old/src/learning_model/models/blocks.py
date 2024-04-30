from torch import nn

SCALING_FACTOR = 2

class ResBlockFC(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, activation:str='leaky_relu'):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dr = nn.Dropout(p=0.1)
        self.af = Activation(activation)

    def forward(self, input):
        d1 = self.af(self.fc(input))
        d2 = self.dr(d1)
        return self.bn(d1 + d2)

class ResBlockConv(nn.Module):
    def __init__(self, out_channels:int, activation:str='leaky_relu'):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dr = nn.Dropout(p=0.1)
        self.af = Activation(activation)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = self.bn1(self.af(self.conv1(input)))
        input = self.dr(self.bn2(self.af(self.conv2(input))))
        input = input + shortcut
        return input

class ResBlock1D(ResBlockConv):
    def __init__(self, in_channels:int, out_channels:int, scaling:int, activation:str='leaky_relu'):
        super().__init__(out_channels, activation)

        if scaling in ['down']:
            # downsampling matrix
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=SCALING_FACTOR, padding=1)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=SCALING_FACTOR),
                nn.BatchNorm1d(out_channels)
            )
        elif scaling in ['up']:
            # upsampling matrix
            self.conv1 = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=SCALING_FACTOR)
            )
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.Upsample(scale_factor=SCALING_FACTOR, mode='linear'),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()


class ResBlock2D(ResBlockConv):
    def __init__(self, in_channels:int, out_channels:int, scaling:int, activation:str='leaky_relu'):
        super().__init__(out_channels, activation)

        if scaling in ['down']:
            # downsampling matrix
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=SCALING_FACTOR, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=SCALING_FACTOR),
                nn.BatchNorm2d(out_channels)
            )
        elif scaling in ['up']:
            # upsampling matrix
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=SCALING_FACTOR)
            )
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.Upsample(scale_factor=SCALING_FACTOR, mode='linear'),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)


class ResBlock3D(ResBlockConv):
    def __init__(self, in_channels:int, out_channels:int, scaling:int, activation:str='leaky_relu'):
        super().__init__(out_channels, activation)

        if scaling in ['down']:
            # downsampling matrix
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=SCALING_FACTOR, padding=1)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=SCALING_FACTOR),
                nn.BatchNorm3d(out_channels)
            )
        elif scaling in ['up']:
            # upsampling matrix
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=SCALING_FACTOR)
            )
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.Upsample(scale_factor=SCALING_FACTOR, mode='linear'),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)


class Activation(nn.Module):
    def __init__(self, activation):
        super().__init__()
        if isinstance(activation, str):
            if activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'leaky_relu':
                self.activation = nn.LeakyReLU()
            elif activation == 'swish':
                self.activation = nn.Swish()
            elif activation == 'elu':
                self.activation = nn.ELU()
            else:
                raise ValueError(f'{activation} not in the list of valid activation functions')
        else:
            self.activation = activation
    
    def forward(self, input):
        if self.activation is None:
            return input
        return self.activation(input)