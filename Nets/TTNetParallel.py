from __future__ import print_function

import torch

import torch.nn as nn
import torch.nn.functional as F

from Utils import Constant
from Utils.RanksFactory import RanksFactory
from Utils.TensorTools import group_divisions


class FirstKernelTensorTrain(nn.Module):
    def __init__(self, m, r_j):
        super(FirstKernelTensorTrain, self).__init__()
        self.fc1 = nn.Linear(m, r_j, bias=False)
        self.m = m
        self.r_j = r_j

    def forward(self, tensor):
        # X size is (64,1,16,49)
        transformed_tensor = self.fc1(tensor)
        return F.relu(transformed_tensor)


class FeatureMap(nn.Module):
    def __init__(self, n, m, amount_of_division, batch_size):
        super(FeatureMap, self).__init__()
        self.m = m
        self.n = n
        self.amount_of_division = amount_of_division
        self.batch_size = batch_size
        self.fc1 = nn.Linear(self.n, self.m)


    def forward(self, tensor):
        # X size is (64,1,16,49)
        last_dim=tensor.size()[-1]
        tensor=tensor.contiguous()
        tensor_reshaped = tensor.view(-1, last_dim)  # x1 size (1024,49)
        tensor_transformed = F.relu(self.fc1(tensor_reshaped))
        return tensor_transformed


class TTKernel(nn.Module):
    def __init__(self, r_i, m, r_j):
        super(TTKernel, self).__init__()
        self.fc1 = nn.Bilinear(r_i, m, r_j, bias=False)

    def forward(self, input_tensor_1, input_tensor_2):
        # X size is (64,1,49,16)
        tensor_transformed = self.fc1(input_tensor_1, input_tensor_2)
        return F.relu(tensor_transformed)
        #return  tensor_transformed

class TTNetParallel(nn.Module):
    def __init__(self, net_params):
        super(TTNetParallel, self).__init__()

        self.kernels = []
        self.dicc_kernels = {}
        self.ranks = RanksFactory.create_tensor_train_parallel_ranks(net_params)
        self.m = net_params.get_m()
        self.n = net_params.get_n()
        self.amount_of_divisions=net_params.get_amount_of_divisions()
        self.batch_size=net_params.get_batch_size()
        self.feature_map = FeatureMap(self.n, self.m, self.amount_of_divisions, self.batch_size)
        self.amount_of_divisions = net_params.amount_of_divisions
        for category in range(net_params.categories):
            first_kernel = FirstKernelTensorTrain(self.m, self.ranks[Constant.SECOND])
            self.dicc_kernels[category] = [first_kernel]
            self.kernels.append(first_kernel)
            for r in range(Constant.THIRD, len(self.ranks)):
                r_i = self.ranks[r - 1]
                r_j = self.ranks[r]
                kernel = TTKernel(r_i, self.m, r_j)
                self.dicc_kernels[category].append(kernel)
                self.kernels.append(kernel)
        self.net = nn.Sequential(*self.kernels)

    def forward(self, tensor):
        featured_tensor = self.feature_map(tensor)
        division_divided_tensors = group_divisions(featured_tensor, self.amount_of_divisions)
        outputs = []
        for category in self.dicc_kernels:
            division = 0
            amount_of_kernels = len(self.dicc_kernels[category])
            first_kernel = self.dicc_kernels[category][0]
            kernel_input = first_kernel(division_divided_tensors[division])
            for n_kernel in range(1, amount_of_kernels):
                division += 1
                division_divided_input = division_divided_tensors[division]
                kernel = self.dicc_kernels[category][n_kernel]
                kernel_input = kernel(kernel_input, division_divided_input)
            outputs.append(kernel_input)
        concatenated_outputs = torch.cat(outputs, 1)  # Concatenate columns into tensor
        return F.log_softmax(concatenated_outputs, dim=1)

    @staticmethod
    def get_number_of_parameters(n, m, r, d, categories):
        feature_map_size = m * n + m
        first_kernel_size = m * r
        middle_kernels_size = ((d - 2) * r * m * r)
        last_kernel_size = r * m
        total = feature_map_size+(first_kernel_size+middle_kernels_size+last_kernel_size)*categories
        #print("TTNetParallel",total)
        return total