from Nets.TRNetShared import TRNetShared
from Nets.TTNetShared import TTNetShared
from Nets.Net import ConvolutionalNet, FullyConnected, ConvolutionalNetWithTT, ConvolutionalNetWithTR
from Nets.ParallelizedTensorsNet import ParallelizedTensorNet
from Nets.LSTM import LSTM, RNNText, RNN, CNNText
from Nets.TRNetParallel import TRNetParallel
from Nets.TRNetSerialized import TRNetSerialized
from Nets.TTNetParallel import TTNetParallel
from Nets.TTNetSerialized import TTNetSerialized
from Nets.TTNetSharedParallelized import TTNetSharedParallelized
from Nets.TensorTextNet import TensorTextNet,TensorTextNetConv,TensorRingTextNet,TensorRingTextNetConv

MODELS = dict(NET=ConvolutionalNet, FC=FullyConnected, LSTM=LSTM, RNN_TEXT=RNNText, TR_PARALLEL=TRNetParallel,
              TR_SERIALIZED=TRNetSerialized, TT_PARALLEL=TTNetParallel, TT_SERIALIZED=TTNetSerialized,
              PARALLELIZED_TENSOR_NET=ParallelizedTensorNet, TENSOR_TEXT_NET=TensorTextNet,
              TENSOR_RING_TEXT_NET=TensorRingTextNet , TT_SHARED=TTNetShared,
              TT_SHARED_PARALLELIZED_MODEL=TTNetSharedParallelized,
              TR_SHARED=TRNetShared, RNN=RNN, ConvTT=ConvolutionalNetWithTT, ConvTR=ConvolutionalNetWithTR,
              CNN_TEXT=CNNText, TENSOR_TEXT_NET_CONV=TensorTextNetConv, TENSOR_RING_TEXT_NET_CONV=TensorRingTextNetConv)

NET_MODEL = "NET"
FC_MODEL = "FC"
LSTM_MODEL = "LSTM"
RNN_MODEL= "RNN"
RNN_TEXT_MODEL = "RNN_TEXT"
TR_PARALLEL_MODEL = "TR_PARALLEL"
TR_SERIALIZED_MODEL = "TR_SERIALIZED"
TT_PARALLEL_MODEL = "TT_PARALLEL"
TT_SERIALIZED_MODEL = "TT_SERIALIZED"
TT_SHARED_MODEL = "TT_SHARED"
TT_SHARED_PARALLELIZED_MODEL="TT_SHARED_PARALLELIZED_MODEL"
PARALLELIZED_TENSOR_NET = "PARALLELIZED_TENSOR_NET"
TENSOR_TEXT_NET = "TENSOR_TEXT_NET"
TENSOR_RING_TEXT_NET = "TENSOR_RING_TEXT_NET"
TENSOR_TEXT_NET_CONV = "TENSOR_TEXT_NET_CONV"
TENSOR_RING_TEXT_NET_CONV =  "TENSOR_RING_TEXT_NET_CONV"
TR_SHARED_MODEL="TR_SHARED"
CNN_TT_NET="ConvTT"
CNN_TR_NET="ConvTR"
CNN_TEXT_NET = 'CNN_TEXT'
