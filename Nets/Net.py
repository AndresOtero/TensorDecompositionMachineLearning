from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F

from Nets.TRNetShared import TRNetShared
from Nets.TTNetParallel import FeatureMap
from Nets.TTNetShared import TTNetShared
from Utils.TensorTools import flat_divisions, flat_divisions_with_batch


class FullyConnected(nn.Module):
    def __init__(self, net_params):
        super(FullyConnected, self).__init__()
        self.amount_of_divisions = net_params.get_amount_of_divisions()
        self.m = net_params.get_m()
        self.n = net_params.get_n()
        self.fc1 = FeatureMap(self.n, self.m, net_params.get_amount_of_divisions(), net_params.get_batch_size())
        self.fc2 = nn.Linear(self.m * self.amount_of_divisions, 10)


    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, self.m * self.amount_of_divisions)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_number_of_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


class ConvolutionalNet(nn.Module):
    def __init__(self, net_params):
        super(ConvolutionalNet, self).__init__()

        self.conv_layer = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)

        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)

        return F.log_softmax(x, dim=1)

    def get_number_of_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


class ConvolutionalNetWithTT(nn.Module):
    def __init__(self, net_params):
        super(ConvolutionalNetWithTT, self).__init__()
        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.5),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.tt= TTNetShared(net_params)
        self.row = net_params.get_divides_in_row()
        self.col =net_params.get_divides_in_col()
        self.divisions =net_params.get_amount_of_divisions()
        self.n =net_params.get_n()

    def forward(self, x):
        batch_size, first_dim,second_dim,third_dim = x.size()
        x1 = self.conv_layer(x)
        x2 = x1.view(batch_size, self.divisions, self.n)
        x3 = self.tt(x2)
        return x3

    def get_number_of_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


class ConvolutionalNetWithTR(nn.Module):
    def __init__(self, net_params):
        super(ConvolutionalNetWithTR, self).__init__()
        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.5),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.tr = TRNetShared(net_params)
        self.row = net_params.get_divides_in_row()
        self.col = net_params.get_divides_in_col()
        self.divisions = net_params.get_amount_of_divisions()
        self.n = net_params.get_n()

    def forward(self, x):
        batch_size, first_dim, second_dim, third_dim = x.size()
        x1 = self.conv_layer(x)
        x2 = x1.view(batch_size, self.divisions, self.n)
        x3 = self.tr(x2)
        return x3

    def get_number_of_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])
