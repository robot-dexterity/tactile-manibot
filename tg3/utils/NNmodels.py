from abc import ABC, abstractmethod
import numpy as np
import os 
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class NNModule(ABC, nn.Module):
    """ Abstract base class for CNN models. """

    def __repr__(self):      
        return "{} ({})".format(self.__class__.__name__, self.info)

    def __str__(self):
        return self.__repr__()

    def __enter__(self):
        return self
        
    def __exit__(self):
        self.close()

    @property
    def type(self):
        return self.__class__.__name__

    def weights_init_normal(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)

    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def _soft_bound(self, x):
        return self.xmin + F.softplus(x - self.xmin) - F.softplus(x - self.xmax)
                
    @abstractmethod
    def forward(self, x):
        pass


class CNN(NNModule):
    def __init__(
        self,
        in_dim,
        in_channels,
        out_dim,
        conv_layers=[16, 16, 16],
        conv_kernel_sizes=[5, 5, 5],
        fc_layers=[128, 128],
        activation='relu',
        apply_batchnorm=False,
        dropout=0.0,
        device='cpu'
    ):
        super().__init__()

        assert conv_layers and fc_layers, "conv_layers and fc_layers must contain values"
        assert len(conv_layers) == len(conv_kernel_sizes), "conv_layers and conv_kernel_sizes must be same length"
        
        self.device = device
        act_layer = nn.ReLU if activation == 'relu' else nn.ELU

        # CNN layers
        in_ch = in_channels
        cnn_modules = []
        for out_ch, ksize in zip(conv_layers, conv_kernel_sizes):
            cnn_modules += [
                nn.Conv2d(in_ch, out_ch, kernel_size=ksize, stride=1, padding=2),
                nn.BatchNorm2d(out_ch) if apply_batchnorm else nn.Identity(),
                act_layer(),
                nn.MaxPool2d(2, 2)
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_modules)

        # Flatten size
        with torch.no_grad():
            n_flatten = np.prod(self.cnn(torch.zeros((1, in_channels, *in_dim))).shape)

        # FC layers
        fc_modules = [nn.Linear(n_flatten, fc_layers[0]), act_layer()]
        for i in range(len(fc_layers) - 1):
            fc_modules += [nn.Linear(fc_layers[i], fc_layers[i + 1]), act_layer(), nn.Dropout(dropout)]
        fc_modules.append(nn.Linear(fc_layers[-1], out_dim))
        self.fc = nn.Sequential(*fc_modules)

    def forward(self, x):
        # x = (x>0.5).float() #..todo: kipp this line seems to fix sim2real, why?
        return self.fc(self.cnn(x.to(self.device)).view(x.size(0), -1))


class NatureCNN(NNModule):
    """ CNN commonly used in RL from DQN: Mnih, Volodymyr, et al. Nature 518.7540 (2015): 529-533. """

    def __init__(
        self,
        in_dim,
        in_channels,
        out_dim,
        fc_layers=[128, 128],
        dropout=0.0
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = np.prod(self.cnn(torch.zeros(1, in_channels, *in_dim)).shape)

        fc_layers = [nn.Linear(n_flatten, fc_layers[0]), nn.ReLU()]
        for i in range(len(fc_layers) - 1):
            fc_layers += [nn.Linear(fc_layers[i], fc_layers[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        fc_layers.append(nn.Linear(fc_layers[-2].out_features, out_dim))

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        return self.fc(self.cnn(x).view(x.size(0), -1))


class ResidualBlock(NNModule):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class ResNet(NNModule):
    def __init__(self, block, in_channels, out_dim, layers=[2, 2, 2, 2]):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0])
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, out_dim)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers += [block(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x) 

