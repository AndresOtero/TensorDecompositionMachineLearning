import math

import torch

from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from Nets.CoreKernelTensorRing import CoreKernelTensorRing
from Nets.TRNetSerialized import TRNetSerialized
from Nets.TTNetParallel import FeatureMap
from Utils import Constant
from Utils.RanksFactory import RanksFactory
from Utils.TensorTools import group_divisions


class KernelTensorRingWithCategoryAndState(nn.Module):

    def __init__(self, amount_of_categories, first_rank, m, second_rank):
        super(KernelTensorRingWithCategoryAndState, self).__init__()
        self.first_rank = first_rank
        self.m = m
        self.second_rank = second_rank
        self.weight = Parameter(torch.randn(m, first_rank, m, m, second_rank))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input,state):
        # See the autograd section for explanation of what happens here.
        product_state=torch.matmul(state,self.weight).squeeze(3)
        return torch.matmul(input, product_state)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class TRNetSerializedWithState(nn.Module):

    def __init__(self, net_params):
        super(TRNetSerializedWithState, self).__init__()

        self.kernels = []
        self.ranks = RanksFactory.create_tensor_ring_ranks(net_params)
        self.m = net_params.get_m()
        self.n = net_params.get_n()
        self.amount_of_divisions = net_params.get_amount_of_divisions()
        self.batch_size = net_params.get_batch_size()
        self.feature_map = FeatureMap(self.n, self.m, self.amount_of_divisions, self.batch_size)
        self.amount_of_divisions = net_params.get_amount_of_divisions()
        self.amount_of_categories = net_params.get_amount_of_categories()
        first_kernel = KernelTensorRingWithCategoryAndState(self.amount_of_categories, self.ranks[Constant.FIRST],
                                                            self.m, self.ranks[Constant.SECOND])
        self.kernels.append(first_kernel)
        for r in range(Constant.THIRD, len(self.ranks)):
            r_i = self.ranks[r - 1]
            r_j = self.ranks[r]
            kernel = CoreKernelTensorRing(r_i, self.m, r_j)
            self.kernels.append(kernel)

        self.net = nn.Sequential(*self.kernels)

    def forward(self, tensor,state):
        featured_tensor = self.feature_map(tensor)
        division_divided_tensors = group_divisions(featured_tensor, self.amount_of_divisions)
        kernel_products = self.get_kernel_products(self.kernels, division_divided_tensors,state)

        product = TRNetSerialized.foward_products_serialized(kernel_products)

        concatenated_trace_results = TRNetSerialized.calculate_traces_serialized(product)

        return concatenated_trace_results

    @staticmethod
    def get_kernel_products( kernels, division_divided_tensors,state):
        amount_of_kernels = len(kernels)
        kernel_products = []
        division_divided_input = division_divided_tensors[0]
        first_kernel = kernels[0]
        kernel_product = first_kernel(division_divided_input,state)
        kernel_products.append(kernel_product)
        for n_kernel in range(1, amount_of_kernels):
            division = n_kernel
            division_divided_input = division_divided_tensors[division]

            kernel = kernels[n_kernel]

            kernel_product = kernel(division_divided_input)
            kernel_products.append(kernel_product)
        return kernel_products
