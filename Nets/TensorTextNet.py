import math

import torch

from torch import nn, autograd

from Nets.TRNetSerializedWithState import TRNetSerializedWithState
from Nets.TTNetSerializedWithState import TTNetSerializedWithState
from Nets.TTNetShared import TTNetShared, TTNetSharedWithoutFeatureMap
from Nets.TRNetShared import TRNetSharedWithoutFeatureMap

from Utils import Constant
from Utils.PreProcessText import PreProcessText
import torch.nn.functional as F

from Utils.TensorTools import flat_divisions_with_batch


class TensorTextNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        input_dim = PreProcessText.get_instance().get_vocab_size()
        self.embedding = nn.Embedding(input_dim, net_params.get_embedding())
        PreProcessText.get_instance().configure_embeddings(self.embedding)
        self.tt_net = TTNetSharedWithoutFeatureMap(net_params)
        self.amount_of_divisions = net_params.get_amount_of_divisions()
        self.dropout = nn.Dropout(p=net_params.get_dropout())
        self._get_number_of_params()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.hidden, a=math.sqrt(5))


    def forward(self, text):
        embedded = self.dropout( self.embedding(text))
        batch_divison_size = embedded.size()[0]
        tensor_resized = embedded.view(batch_divison_size,self.amount_of_divisions, -1)
        output = self.tt_net(tensor_resized)
        return output

    def _get_number_of_params(self):
        for p in self.parameters():
            print(p.size(), p.numel)
        self.number = sum(p.numel() for p in self.parameters())

    def get_number_of_parameters(self):
        return self.number

class TensorRingTextNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        input_dim = PreProcessText.get_instance().get_vocab_size()
        self.embedding = nn.Embedding(input_dim, net_params.get_embedding())
        PreProcessText.get_instance().configure_embeddings(self.embedding)
        self.tt_net = TRNetSharedWithoutFeatureMap(net_params)
        self.amount_of_divisions = net_params.get_amount_of_divisions()
        self.dropout = nn.Dropout(p=net_params.get_dropout())
        self._get_number_of_params()


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.hidden, a=math.sqrt(5))


    def forward(self, text):
        embedded = self.dropout( self.embedding(text))
        batch_divison_size = embedded.size()[0]
        tensor_resized = embedded.view(batch_divison_size,self.amount_of_divisions, -1)
        output = self.tt_net(tensor_resized)
        return output

    def _get_number_of_params(self):
        for p in self.parameters():
            print(p.size(), p.numel)
        self.number = sum(p.numel() for p in self.parameters())

    def get_number_of_parameters(self):
        return self.number


class TensorTextNetConv(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        input_dim = PreProcessText.get_instance().get_vocab_size()
        pad_idx = PreProcessText.get_instance().get_padding_idx()
        self.embedding = nn.Embedding(input_dim, net_params.get_embedding(), pad_idx)
        PreProcessText.get_instance().configure_embeddings(self.embedding)
        self.tt_net = TTNetSharedWithoutFeatureMap(net_params)

        n_filters = 100
        filter_sizes = [3,4,5]

        self.amount_of_divisions = net_params.get_amount_of_divisions()

        self.dropout = nn.Dropout(p=net_params.get_dropout())

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, net_params.get_embedding()))
            for fs in filter_sizes
        ])

        self._get_number_of_params()



    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.hidden, a=math.sqrt(5))

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=2))

        batch_divison_size = embedded.size()[0]
        tensor_resized = cat.reshape(batch_divison_size,self.amount_of_divisions, -1)
        output = self.tt_net(tensor_resized)

        return output

    def _get_number_of_params(self):
        for p in self.parameters():
            print(p.size(), p.numel)
        self.number = sum(p.numel() for p in self.parameters())

    def get_number_of_parameters(self):
        return self.number

class TensorRingTextNetConv(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        input_dim = PreProcessText.get_instance().get_vocab_size()
        pad_idx = PreProcessText.get_instance().get_padding_idx()
        self.embedding = nn.Embedding(input_dim, net_params.get_embedding(), pad_idx)
        PreProcessText.get_instance().configure_embeddings(self.embedding)

        self.tt_net = TRNetSharedWithoutFeatureMap(net_params)

        n_filters = 100
        filter_sizes = [3,4,5]

        self.amount_of_divisions = net_params.get_amount_of_divisions()

        self.dropout = nn.Dropout(p=net_params.get_dropout())

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, net_params.get_embedding()))
            for fs in filter_sizes
        ])

        self._get_number_of_params()


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.hidden, a=math.sqrt(5))

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=2))

        batch_divison_size = embedded.size()[0]
        tensor_resized = cat.reshape(batch_divison_size,self.amount_of_divisions, -1)
        output = self.tt_net(tensor_resized)

        return output

    def _get_number_of_params(self):
        for p in self.parameters():
            print(p.size(), p.numel)
        self.number = sum(p.numel() for p in self.parameters())

    def get_number_of_parameters(self):
        return self.number