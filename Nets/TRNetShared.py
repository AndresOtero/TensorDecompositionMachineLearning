import torch

import torch.nn as nn
import torch.nn.functional as F

from Nets.TRNetSerialized import TRNetSerialized
from Nets.TTNetParallel import FirstKernelTensorTrain, FeatureMap, TTKernel
from Nets.TTNetShared import KernelSharedTensorTrain
from Utils import Constant
from Utils.RanksFactory import RanksFactory
from Utils.TensorTools import group_divisions
from torch.nn import Parameter


class LastKernelSharedTensorRing(nn.Module):
    def __init__(self, categories, first_rank, m, second_rank, init_value):
        super(LastKernelSharedTensorRing, self).__init__()
        self.categories = categories
        self.first_rank = first_rank
        self.m = m
        self.second_rank = second_rank
        self.weight = Parameter(torch.randn(categories, first_rank, m, second_rank))
        self.init_value = init_value
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight,gain=self.init_value )

    def forward(self, input, state):
        x = torch.einsum('bj,rbi->rbji', [input, state])  # OuterProduct
        x = torch.einsum('cijk,rbji->cbrk', [self.weight, x])  # Mutiply by core
        return x


class KernelSharedTensorRing(nn.Module):
    def __init__(self, first_rank, m, second_rank, init_value):
        super(KernelSharedTensorRing, self).__init__()
        self.first_rank = first_rank
        self.m = m
        self.second_rank = second_rank
        self.weight = Parameter(torch.randn(first_rank, m, second_rank))
        self.init_value = init_value
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight,gain=self.init_value )

    def forward(self, input, state):
        x = torch.einsum('bj,rbi->rbji', [input, state])  # OuterProduct
        x = torch.einsum('ijk,rbji->rbk', [self.weight, x])  # Mutiply by core
        return x


class TRNetShared(nn.Module):
    def __init__(self, net_params):
        super(TRNetShared, self).__init__()
        self.ranks = RanksFactory.create_tensor_ring_shared_ranks(net_params)
        self.kernels = []
        self.m = net_params.get_m()
        self.n = net_params.get_n()
        self.amount_of_divisions = net_params.get_amount_of_divisions()
        self.batch_size = net_params.get_batch_size()
        self.feature_map = FeatureMap(self.n, self.m, self.amount_of_divisions, self.batch_size)
        self.amount_of_divisions = net_params.amount_of_divisions
        self.categories = net_params.categories
        self.last_kernel = LastKernelSharedTensorRing(self.categories, self.ranks[Constant.SECOND], self.m,
                                                      self.ranks[Constant.THIRD], net_params.init_value)
        self.shared_kernel = KernelSharedTensorRing(self.ranks[Constant.SECOND], self.m, self.ranks[Constant.THIRD],
                                                    net_params.init_value)

    def forward(self, tensor):
        featured_tensor = self.feature_map(tensor)
        division_divided_tensors = group_divisions(featured_tensor, self.amount_of_divisions)
        batch_size = tensor.size()[Constant.FIRST]
        state = torch.ones(self.ranks[Constant.FIRST], batch_size, self.ranks[Constant.SECOND])
        times = division_divided_tensors.size()[Constant.FIRST]
        for t in range(0, times):
            division_divided_input = division_divided_tensors[t]
            state = self.shared_kernel(division_divided_input, state)
        pad_input = torch.ones(batch_size, self.m)
        state = self.last_kernel(pad_input, state)
        state = TRNetSerialized.calculate_traces_serialized(state)

        return F.log_softmax(state, dim=1)

    def extra_repr(self):
        return 'ranks={}'.format(
            self.ranks
        )

    def get_number_of_parameters(self):
      for p in self.parameters():
            print(p.numel())
      self.number = sum(p.numel() for p in self.parameters())
      return self.number


class TRNetSharedWithoutFeatureMap(nn.Module):
    def __init__(self, net_params):
        super(TRNetSharedWithoutFeatureMap, self).__init__()
        self.ranks = RanksFactory.create_tensor_ring_shared_ranks(net_params)
        self.kernels = []
        self.m = net_params.get_m()
        self.n = net_params.get_n()
        self.amount_of_divisions = net_params.get_amount_of_divisions()
        self.batch_size = net_params.get_batch_size()
        self.amount_of_divisions = net_params.amount_of_divisions
        self.categories = net_params.categories
        self.last_kernel = LastKernelSharedTensorRing(self.categories, self.ranks[Constant.SECOND], self.m,
                                                      self.ranks[Constant.LAST], net_params.init_value)
        self.shared_kernel = KernelSharedTensorRing(self.ranks[Constant.SECOND], self.n, self.ranks[Constant.THIRD],
                                                    net_params.init_value)

    def forward(self, tensor):
        division_divided_tensors = tensor.transpose(0,1)
        batch_size = tensor.size()[Constant.FIRST]
        state = torch.ones(self.ranks[Constant.FIRST], batch_size, self.ranks[Constant.SECOND])
        times = division_divided_tensors.size()[Constant.FIRST]
        for t in range(0, times):
            division_divided_input = division_divided_tensors[t]
            state = self.shared_kernel(division_divided_input, state)
        pad_input = torch.ones(batch_size, self.m)
        state = self.last_kernel(pad_input, state)
        state = TRNetSerialized.calculate_traces_serialized(state)

        return state

    def extra_repr(self):
        return 'ranks={}'.format(
            self.ranks
        )

    def get_number_of_parameters(self):
        self.number = sum(p.numel() for p in self.parameters())
        return self.number