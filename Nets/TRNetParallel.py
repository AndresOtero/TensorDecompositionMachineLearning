from __future__ import print_function

import math

import torch
from Utils.RanksFactory import RanksFactory

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Parameter

from Nets.CoreKernelTensorRing import CoreKernelTensorRing
from Nets.TTNetParallel import FeatureMap, FirstKernelTensorTrain, TTKernel
from Utils import Constant
from Utils.TensorTools import group_divisions, dimension_trace


class TRNetParallel(nn.Module):
    def __init__(self, net_params):
        super(TRNetParallel, self).__init__()

        self.kernels = []
        self.dicc_kernels = {}
        self.ranks = RanksFactory.create_tensor_ring_ranks(net_params)
        self.m = net_params.get_m()
        self.n = net_params.get_n()
        self.amount_of_divisions=net_params.get_amount_of_divisions()
        self.batch_size=net_params.get_batch_size()
        self.feature_map = FeatureMap(self.n, self.m, self.amount_of_divisions, self.batch_size)
        self.amount_of_divisions = net_params.amount_of_divisions
        net_list = []
        for category in range(net_params.categories):
            self.dicc_kernels[category] = []
            for r in range(Constant.SECOND, len(self.ranks)):
                r_i = self.ranks[r - 1]
                r_j = self.ranks[r]
                kernel = CoreKernelTensorRing(r_i, net_params.m, r_j)
                self.dicc_kernels[category].append(kernel)
                self.kernels.append(kernel)
            category_kernels = self.dicc_kernels[category]
            net = nn.Sequential(*category_kernels)
            net_list.append(net)
        self.net = nn.ModuleList(net_list)

    @staticmethod
    def __get_kernel_products(kernels, division_divided_tensors):
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
    def __foward_products(kernel_products):
        product = kernel_products[Constant.FIRST].transpose(0, 1)
        for k in range(1, len(kernel_products)):
            product_input = kernel_products[k].transpose(0, 1)
            product = torch.matmul(product, product_input)
        return product

    @staticmethod
    def __calculate_traces(product):
        trace_results = [torch.trace(square_matrix).unsqueeze(0) for square_matrix in product]
        concatenated_trace_results = torch.cat(trace_results, 0).unsqueeze(1)  # Concatenate columns into tensor
        return concatenated_trace_results

    def forward(self, tensor):
        featured_tensor = self.feature_map(tensor)
        division_divided_tensors = group_divisions(featured_tensor, self.amount_of_divisions)
        outputs = []
        for category in self.dicc_kernels:
            category_kernels = self.dicc_kernels[category]

            kernel_products = self.__get_kernel_products(category_kernels, division_divided_tensors)

            product = self.__foward_products(kernel_products)

            concatenated_trace_results = self.__calculate_traces(product)

            outputs.append(concatenated_trace_results)
        concatenated_outputs = torch.cat(outputs, 1)  # Concatenate columns into tensor
        return F.log_softmax(concatenated_outputs, dim=1)


    @staticmethod
    def get_number_of_parameters(n, m, r, d, categories):
        feature_map_size = m * n + m
        middle_kernels_size = (d * r * m * r) * categories
        total = feature_map_size+middle_kernels_size
        #print("TRNetParallel",total)
        return total