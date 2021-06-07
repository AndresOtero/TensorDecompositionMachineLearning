from Utils import Constant, EnumDataset
from Utils.NetParams import NetParams


class NetParamsFactory:
    @staticmethod
    def create_mnist_net_params(model, learning_rate, optimizer, cuda_is_available, m, rank,batch_size):
        net_params = NetParams(model, EnumDataset.MNIST_FLAT_DIVISIONS, learning_rate, optimizer, cuda_is_available,
                               m=m, rank=rank, tensor_size=Constant.TENSOR_SIZE_MNIST,batch_size=batch_size)
        return net_params

    @staticmethod
    def create_cifar_net_params(model, learning_rate, optimizer, cuda_is_available, m, rank,batch_size):
        net_params = NetParams(model, EnumDataset.CIFAR_FLAT_DIVISIONS, learning_rate, optimizer, cuda_is_available,
                               m=m, rank=rank, tensor_size=Constant.TENSOR_SIZE_CIFAR,
                               divides_in_row=Constant.DIVIDES_IN_ROW_CIFAR,batch_size=batch_size)
        return net_params

    @staticmethod
    def create_cifar_parallel_net_params(model, learning_rate, optimizer, cuda_is_available, m, rank,batch_size):
        net_params = NetParams(model,learning_rate, EnumDataset.CIFAR_PARALLEL_DIVISIONS, optimizer, cuda_is_available,
                               m=m, rank=rank, tensor_size=Constant.TENSOR_SIZE_CIFAR_PARALLEL,batch_size=batch_size)
        return net_params

    @staticmethod
    def create_fashion_mnist_net_params(model, learning_rate, optimizer, cuda_is_available, m, rank,batch_size):
        net_params = NetParams(model, EnumDataset.FASHION_MNIST_FLAT_DIVISIONS, learning_rate, optimizer,
                               cuda_is_available,m=m, rank=rank, tensor_size=Constant.TENSOR_SIZE_MNIST,batch_size=batch_size)
        return net_params

    @staticmethod
    def create_kuzushiji_mnist_net_params(model, learning_rate, optimizer, cuda_is_available, m, rank,batch_size):
        net_params = NetParams(model, EnumDataset.KUZUSHIJI_MNIST_FLAT_DIVISIONS, learning_rate, optimizer,
                               cuda_is_available,m=m, rank=rank, tensor_size=Constant.TENSOR_SIZE_MNIST,batch_size=batch_size)
        return net_params

    @staticmethod
    def create_IMDB_net_params(model, learning_rate, optimizer, cuda_is_available, m, rank,batch_size):
        net_params = NetParams(model, EnumDataset.IMDB, learning_rate, optimizer,
                               cuda_is_available,m=m, rank=rank,batch_size=batch_size)
        return net_params
