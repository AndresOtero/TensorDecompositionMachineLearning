from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F

from Nets.TTNetParallel import FirstKernelTensorTrain, FeatureMap, TTKernel
from Utils import Constant
from Utils.RanksFactory import RanksFactory
from Utils.TensorTools import group_divisions


class TTNetSerialized(nn.Module):
    def __init__(self, net_params):
        super(TTNetSerialized, self).__init__()
        self.ranks = RanksFactory.create_tensor_train_serial_ranks(net_params)
        self.first_kernel = FirstKernelTensorTrain(net_params.m, self.ranks[Constant.SECOND])
        self.kernels = []
        self.m = net_params.get_m()
        self.n = net_params.get_n()
        self.amount_of_divisions=net_params.get_amount_of_divisions()
        self.batch_size=net_params.get_batch_size()
        self.feature_map = FeatureMap(self.n, self.m, self.amount_of_divisions, self.batch_size)
        self.amount_of_divisions = net_params.amount_of_divisions
        for r in range(Constant.THIRD, len(self.ranks)):
            r_i = self.ranks[r - 1]
            r_j = self.ranks[r]
            kernel = TTKernel(r_i, self.m, r_j)
            self.kernels.append(kernel)
        self.net = nn.Sequential(*self.kernels)

    def forward(self, tensor):
        featured_tensor = self.feature_map(tensor)
        division_divided_tensors = group_divisions(featured_tensor, self.amount_of_divisions)
        division = 0
        kernel_input = self.first_kernel(division_divided_tensors[division])
        for kernel in self.kernels:
            division += 1
            division_divided_input = division_divided_tensors[division]
            kernel_input = kernel(kernel_input, division_divided_input)
        return F.log_softmax(kernel_input, dim=1)

    @staticmethod
    def get_number_of_parameters(n, m, r, d, categories):
        feature_map_size = m * n + m
        first_kernel_size = m * r
        middle_kernels_size = ((d - 2) * r * m * r)
        last_kernel_size = r * m * categories
        total = feature_map_size+first_kernel_size+middle_kernels_size+last_kernel_size
        #print("TTNetSerialized",total)
        return total
