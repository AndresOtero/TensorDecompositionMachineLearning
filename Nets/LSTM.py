# Recurrent neural network (many-to-one)
import torch
import torch.nn.functional as F

from Nets.TTNetParallel import FeatureMap
from Utils.PreProcessText import PreProcessText
# Hyper-parameters
from torch import nn

from Utils.TensorTools import group_divisions

sequence_length = 28
INPUT_SIZE = 28
EMBEDDING_DIM = 100
HIDDEN_SIZE = 128
NUM_LAYER = 2
NUM_CLASS = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01


class LSTM(nn.Module):
    def __init__(self, net_params):
        super(LSTM, self).__init__()
        self.hidden_size = net_params.get_rank()
        self.num_layers = net_params.get_amount_of_divisions()
        self.m = net_params.get_m()
        self.n = net_params.get_n()
        self.amount_of_divisions = net_params.get_amount_of_divisions()
        self.batch_size = net_params.get_batch_size()
        self.feature_map = FeatureMap(self.n, self.m, self.amount_of_divisions, self.batch_size)
        self.lstm = nn.LSTM(self.m, self.hidden_size, self.amount_of_divisions, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, net_params.get_amount_of_categories())

    def forward(self, x):
        featured_tensor = self.feature_map(x)
        division_divided_tensors = group_divisions(featured_tensor, self.amount_of_divisions).transpose(0, 1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(division_divided_tensors,(h0, c0))

        out = self.fc(out[:, -1, :])
        return F.log_softmax(out, dim=1)

    def get_number_of_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


class RNN(nn.Module):
    def __init__(self, net_params):
        super(RNN, self).__init__()
        self.hidden_size = net_params.get_rank()
        self.num_layers = net_params.get_amount_of_divisions()
        self.m = net_params.get_m()
        self.n = net_params.get_n()
        self.amount_of_divisions = net_params.get_amount_of_divisions()
        self.batch_size = net_params.get_batch_size()
        self.feature_map = FeatureMap(self.n, self.m, self.amount_of_divisions, self.batch_size)
        self.rnn = nn.RNN(self.m, self.hidden_size, self.amount_of_divisions, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, net_params.get_amount_of_categories())

    def forward(self, x):
        featured_tensor = self.feature_map(x)
        division_divided_tensors = group_divisions(featured_tensor, self.amount_of_divisions).transpose(0, 1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.rnn(division_divided_tensors,
                          h0)

        out = self.fc(out[:, -1, :])
        return F.log_softmax(out, dim=1)

    def get_number_of_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1


class RNNText(nn.Module):
    def __init__(self, net_params, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_SIZE, output_dim=OUTPUT_DIM):
        super().__init__()
        input_dim = PreProcessText.get_instance().get_vocab_size()
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.rnn = nn.RNN(embedding_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        text = text.transpose(0, 1)
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

    def get_number_of_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])





class CNNText(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        vocab_size = PreProcessText.get_instance().get_vocab_size()
        embedding_dim = 50
        n_filters = 100
        filter_sizes = [3, 4, 5]
        output_dim = 1
        dropout = 0.5
        pad_idx = PreProcessText.get_instance().get_padding_index()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))

        return self.fc(cat)

    def get_number_of_parameters(self):
        return sum(p.numel() for p in self.parameters())