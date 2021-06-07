from __future__ import print_function

import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from Nets.TTNetParallel import FirstKernelTensorTrain, FeatureMap, TTKernel
from Nets.TTNetShared import KernelSharedTensorTrain
from Utils import Constant
from Utils.RanksFactory import RanksFactory
from Utils.TensorTools import group_divisions
from torch.nn import Parameter




class TTNetSharedParallelized(nn.Module):
    def __init__(self, net_params):
        super(TTNetSharedParallelized, self).__init__()
        self.ranks = RanksFactory.create_tensor_train_shared_parallel_ranks(net_params)
        self.kernels = []
        self.first_kernels=[]
        self.last_kernels=[]
        self.m = net_params.get_m()
        self.n = net_params.get_n()
        self.amount_of_divisions=net_params.get_amount_of_divisions()
        self.batch_size=net_params.get_batch_size()
        self.feature_map = FeatureMap(self.n, self.m, self.amount_of_divisions, self.batch_size)
        self.categories = net_params.categories
        self.first_kernel= KernelSharedTensorTrain(self.ranks[Constant.FIRST],self.m,self.ranks[Constant.SECOND],net_params.init_value)
        self.last_kernel= KernelSharedTensorTrain(self.ranks[Constant.THIRD],self.m,self.ranks[Constant.LAST],net_params.init_value)
        for r in range(Constant.FIRST, self.categories):
            first_kernel = KernelSharedTensorTrain(self.ranks[Constant.FIRST], self.m, self.ranks[Constant.SECOND],net_params.init_value)
            self.first_kernels.append(first_kernel)
            last_kernel = KernelSharedTensorTrain(self.ranks[Constant.THIRD], self.m, self.ranks[Constant.LAST],net_params.init_value)
            self.last_kernels.append(last_kernel)
            kernel = KernelSharedTensorTrain(self.ranks[Constant.SECOND], self.m, self.ranks[Constant.THIRD],net_params.init_value)
            self.kernels.append(kernel)
        self.first= nn.Sequential(*self.first_kernels)
        self.net = nn.Sequential(*self.kernels)
        self.last = nn.Sequential(*self.last_kernels)

    def forward(self, tensor):
        featured_tensor = self.feature_map(tensor)
        division_divided_tensors = group_divisions(featured_tensor, self.amount_of_divisions)
        states=[]
        for r in range(Constant.FIRST, self.categories):
            batch_size = tensor.size()[Constant.FIRST]
            state = torch.ones(batch_size, self.ranks[Constant.FIRST])
            input = torch.ones(batch_size, self.m)
            state=self.first_kernels[r](input,state)
            times = division_divided_tensors.size()[Constant.FIRST]
            for t in range(0,times):
                division_divided_input = division_divided_tensors[t]
                state = self.kernels[r](division_divided_input,state)
            input = torch.ones(batch_size, self.m)
            state=self.last_kernels[r](input,state)
            states.append(state)
        concatenated_states = torch.cat(states, Constant.DIVISION_DIMENSION)  # Concatenate columns into tensor
        return F.log_softmax(concatenated_states, dim=1)

    def extra_repr(self):
        return 'ranks={}'.format(
            self.ranks
        )