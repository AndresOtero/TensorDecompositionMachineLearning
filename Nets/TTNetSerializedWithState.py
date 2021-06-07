import math

import torch

from torch import nn


from Nets.TTNetParallel import FeatureMap, TTKernel
from Utils import Constant
from Utils.RanksFactory import RanksFactory
from Utils.TensorTools import group_divisions
import torch.nn.functional as F


class KernelTensorTrainWithState(nn.Module):

    def __init__(self, first_rank, m, second_rank):
        super(KernelTensorTrainWithState, self).__init__()
        self.first_rank = first_rank
        self.m = m
        self.second_rank = second_rank
        self.fc1 = nn.Bilinear(first_rank, second_rank, m, bias=False)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input, state):
        # See the autograd section for explanation of what happens here.
        return self.fc1(state, input)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class TTNetSerializedWithState(nn.Module):
    def __init__(self, net_params):
        super(TTNetSerializedWithState, self).__init__()
        self.ranks = RanksFactory.create_tensor_ring_ranks(net_params)
        self.kernels = []
        self.m = net_params.get_embedding()
        self.n = net_params.get_n()
        self.amount_of_divisions = net_params.get_amount_of_divisions()
        self.batch_size = net_params.get_batch_size()
        self.feature_map = FeatureMap(self.n, self.m, self.amount_of_divisions, self.batch_size)
        self.amount_of_divisions = net_params.amount_of_divisions
        for r in range(Constant.SECOND, len(self.ranks)):
            r_i = self.ranks[r - 1]
            r_j = self.ranks[r]
            kernel = TTKernel(r_i, self.m, r_j)
            self.kernels.append(kernel)
        self.net = nn.Sequential(*self.kernels)

    def forward(self, tensor, state):
        last_dim = tensor.size()[-1]
        tensor = tensor.contiguous()
        tensor_reshaped = tensor.view(-1, last_dim)
        division_divided_tensors = group_divisions(tensor_reshaped, self.amount_of_divisions)
        division = 0
        for kernel in self.kernels:
            division_divided_input = division_divided_tensors[division]
            state = state.narrow(0, 0, division_divided_input.size()[0])
            state = kernel(state, division_divided_input)
            division += 1
        return F.log_softmax(state, dim=1)
