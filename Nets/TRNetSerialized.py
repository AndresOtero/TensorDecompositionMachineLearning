from __future__ import print_function

import math

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Parameter

from Nets.CoreKernelTensorRing import CoreKernelTensorRing
from Nets.TTNetParallel import FeatureMap, FirstKernelTensorTrain, TTKernel
from Utils import Constant
from Utils.RanksFactory import RanksFactory
from Utils.TensorTools import group_divisions, dimension_trace

from torch import nn
from torch.nn import Parameter
import torch
import math


class KernelTensorRingWithCategory(nn.Module):

    def __init__(self, amount_of_categories, first_rank, m, second_rank):
        super(KernelTensorRingWithCategory, self).__init__()
        self.first_rank = first_rank
        self.m = m
        self.second_rank = second_rank
        self.weight = Parameter(torch.randn(amount_of_categories, first_rank, m, second_rank))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        #return F.relu(torch.matmul(input, self.weight))
        return torch.matmul(input, self.weight)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



class TRNetSerialized(nn.Module):
    def __init__(self, net_params):
        super(TRNetSerialized, self).__init__()

        self.kernels = []
        self.ranks = RanksFactory.create_tensor_ring_ranks(net_params)
        self.m = net_params.get_m()
        self.n = net_params.get_n()
        self.amount_of_divisions=net_params.get_amount_of_divisions()
        self.batch_size=net_params.get_batch_size()
        self.feature_map = FeatureMap(self.n, self.m, self.amount_of_divisions, self.batch_size)
        self.amount_of_divisions = net_params.get_amount_of_divisions()
        self.amount_of_categories = net_params.get_amount_of_categories()
        first_kernel = KernelTensorRingWithCategory(self.amount_of_categories, self.ranks[Constant.FIRST], self.m,
                                                    self.ranks[Constant.SECOND])
        self.kernels.append(first_kernel)
        for r in range(Constant.THIRD, len(self.ranks)):
            r_i = self.ranks[r - 1]
            r_j = self.ranks[r]
            kernel = CoreKernelTensorRing(r_i, self.m, r_j)
            self.kernels.append(kernel)

        self.net = nn.Sequential(*self.kernels)

    @staticmethod
    def get_kernel_products( kernels, division_divided_tensors):
        amount_of_kernels = len(kernels)
        kernel_products = []
        for n_kernel in range(0, amount_of_kernels):
            division = n_kernel
            division_divided_input = division_divided_tensors[division]

            kernel = kernels[n_kernel]

            kernel_product = kernel(division_divided_input)
            kernel_products.append(kernel_product)
        return kernel_products

    @staticmethod
    def foward_products_serialized( kernel_products):
        product = kernel_products[Constant.FIRST].transpose(1, 2)
        for k in range(1, len(kernel_products)):
            product_input = kernel_products[k].transpose(0, 1)
            product = torch.matmul(product, product_input)
        return product


    @staticmethod
    def calculate_traces_serialized(product):
        trace_result = torch.einsum('cbii->bc', product)
        return trace_result



    def forward(self, tensor):
        featured_tensor = self.feature_map(tensor)
        division_divided_tensors = group_divisions(featured_tensor, self.amount_of_divisions)
        outputs = []
        kernel_products = self.get_kernel_products(self.kernels, division_divided_tensors)

        product = self.foward_products_serialized(kernel_products)

        concatenated_trace_results = self.calculate_traces_serialized(product)

        return F.log_softmax(concatenated_trace_results, dim=1)


    @staticmethod
    def get_number_of_parameters(n, m, r, d, categories):
        feature_map_size = m * n + m
        first_kernel_size = categories * r * m * r
        middle_kernels_size = ((d - 1) * r * m * r)
        total = feature_map_size+first_kernel_size+middle_kernels_size
        #print("TRNetSerialized",total)
        return total

class TRNetSerializedCell(TRNetSerialized):
    def __init__(self, net_params):
        super().__init__(net_params)


    def forward(self, tensor):
        featured_tensor = self.feature_map(tensor)
        division_divided_tensors = group_divisions(featured_tensor, self.amount_of_divisions)
        kernel_products = super(TRNetSerializedCell,self).get_kernel_products(self.kernels, division_divided_tensors)
        product = super(TRNetSerializedCell,self).foward_products_serialized(kernel_products)
        concatenated_trace_results = super(TRNetSerializedCell,self).calculate_traces_serialized(product)
        return concatenated_trace_results
