from __future__ import print_function

import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from Nets.TTNetParallel import FirstKernelTensorTrain, FeatureMap, TTKernel
from Utils import Constant
from Utils.RanksFactory import RanksFactory
from Utils.TensorTools import group_divisions
from torch.nn import Parameter


class FirstKernelSharedTensorTrain(nn.Module):
    def __init__(self, m, r_j):
        super(FirstKernelSharedTensorTrain, self).__init__()
        self.fc1 = nn.Linear(m, r_j, bias=False)
        self.m = m
        self.r_j = r_j

    def forward(self, tensor):
        transformed_tensor = self.fc1(tensor)
        return transformed_tensor


class KernelSharedTensorTrain(nn.Module):
    def __init__(self, first_rank, m, second_rank, init_value):
        super(KernelSharedTensorTrain, self).__init__()
        self.first_rank = first_rank
        self.m = m
        self.second_rank = second_rank
        self.weight = Parameter(torch.randn(first_rank, m, second_rank))
        self.init_value = init_value
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight,gain=self.init_value )

    def forward(self, input, state):
        x = torch.einsum('bj,bi->bji', [input, state])  # OuterProduct
        x = torch.einsum('ijk,bji->bk', [self.weight, x])  # Mutiply by core
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class TTNetShared(nn.Module):
    def __init__(self, net_params):
        super(TTNetShared, self).__init__()
        self.ranks = RanksFactory.create_tensor_train_shared_ranks(net_params)
        self.kernels = []
        self.m = net_params.get_m()
        self.n = net_params.get_n()
        self.amount_of_divisions = net_params.get_amount_of_divisions()
        self.batch_size = net_params.get_batch_size()
        self.feature_map = FeatureMap(self.n, self.m, self.amount_of_divisions, self.batch_size)
        self.amount_of_divisions = net_params.amount_of_divisions
        self.first_kernel = KernelSharedTensorTrain(self.ranks[Constant.FIRST], self.m, self.ranks[Constant.SECOND],
                                                    net_params.init_value)
        self.shared_kernel = KernelSharedTensorTrain(self.ranks[Constant.SECOND], self.m, self.ranks[Constant.THIRD],
                                                     net_params.init_value)
        self.last_kernel = KernelSharedTensorTrain(self.ranks[Constant.THIRD], self.m, self.ranks[Constant.LAST],
                                                   net_params.init_value)

    def forward(self, tensor):
        featured_tensor = self.feature_map(tensor)
        division_divided_tensors = group_divisions(featured_tensor, self.amount_of_divisions)
        batch_size = tensor.size()[Constant.FIRST]
        state = torch.ones(batch_size, 1)
        pad_input = torch.ones(batch_size, self.m)
        state = self.first_kernel(pad_input, state)
        times = division_divided_tensors.size()[Constant.FIRST]
        for t in range(0, times):
            division_divided_input = division_divided_tensors[t]
            state = self.shared_kernel(division_divided_input, state)
        pad_input = torch.ones(batch_size, self.m)
        state = self.last_kernel(pad_input, state)
        return F.log_softmax(state, dim=1)

    def extra_repr(self):
        return 'ranks={}'.format(
            self.ranks
        )

    def get_number_of_parameters(self):
        self.number = sum(p.numel() for p in self.parameters())
        return self.number


class TTNetSharedWithoutFeatureMap(nn.Module):
    def __init__(self, net_params):
        super(TTNetSharedWithoutFeatureMap, self).__init__()
        self.ranks = RanksFactory.create_tensor_train_shared_ranks(net_params)
        self.kernels = []
        self.m = net_params.get_m()
        self.n = net_params.get_n()
        self.amount_of_divisions = net_params.get_amount_of_divisions()
        self.batch_size = net_params.get_batch_size()
        self.amount_of_divisions = net_params.amount_of_divisions
        self.first_kernel = KernelSharedTensorTrain(self.ranks[Constant.FIRST], self.n, self.ranks[Constant.SECOND],
                                                    net_params.init_value)
        self.shared_kernel = KernelSharedTensorTrain(self.ranks[Constant.SECOND], self.n, self.ranks[Constant.THIRD],
                                                     net_params.init_value)
        self.last_kernel = KernelSharedTensorTrain(self.ranks[Constant.THIRD], self.n, self.ranks[Constant.LAST],
                                                   net_params.init_value)

    def forward(self, tensor):
        division_divided_tensors = tensor.transpose(0,1)
        batch_size = tensor.size()[Constant.FIRST]
        state = torch.ones(batch_size, 1)
        pad_input = torch.ones(batch_size, self.n)
        state = self.first_kernel(pad_input, state)
        times = division_divided_tensors.size()[Constant.FIRST]
        for t in range(0, times):
            division_divided_input = division_divided_tensors[t]
            state = self.shared_kernel(division_divided_input, state)
        pad_input = torch.ones(batch_size, self.n)
        state = self.last_kernel(pad_input, state)
        return state

    def extra_repr(self):
        return 'ranks={}'.format(
            self.ranks
        )

    def get_number_of_parameters(self):
        self.number = sum(p.numel() for p in self.parameters())
        return self.number