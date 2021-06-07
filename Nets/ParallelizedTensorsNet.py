import torch

from torch import nn
import torch.nn.functional as F

from Utils import Constant


class ParallelizedTensorNet(nn.Module):
    def __init__(self, net_params):
        super(ParallelizedTensorNet, self).__init__()
        self.n_parallel_nets = net_params.get_n_parallel_nets()
        net_list = []
        tensor_net = net_params.get_tensor_net()
        for i in range(self.n_parallel_nets):
            new_tensor_net = tensor_net(net_params)
            net_list.append(new_tensor_net)
        self.net = nn.ModuleList(net_list)
        self.net_list = net_list
        self.fc = nn.Linear(net_params.categories * self.n_parallel_nets, net_params.categories)

    def forward(self, tensor):
        parallel_tensors_list = []
        for n_net in range(self.n_parallel_nets):
            narrowed_tensor = tensor.narrow(Constant.FIRST_DIMENSION, n_net, Constant.SIZE_ONE_DIMENSION)
            parallel_net_tensor = self.net_list[n_net]
            parallel_tensor = parallel_net_tensor(narrowed_tensor)
            parallel_tensors_list.append(parallel_tensor)
        concatenated_tensor = torch.cat(parallel_tensors_list, 1)  # Concatenate columns into tensor
        final_tensor = F.relu(self.fc(concatenated_tensor))
        return F.log_softmax(final_tensor, dim=1)

    def get_number_of_parameters(self):
        self.number = sum(p.numel() for p in self.parameters())
        return self.number