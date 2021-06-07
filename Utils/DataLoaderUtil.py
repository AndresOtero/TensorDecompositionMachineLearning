import torch
from torchvision import datasets, transforms
#from torchtext import data
from torchtext import datasets as datasetsText
from Utils.MnistDataSet import FashionMnist, KuzushijiMnist
from Utils.TensorTools import flat_divisions,parallel_flat_division


class DataLoaderUtil:
    @staticmethod
    def get_train_loader_mnist(net_params):
        use_cuda = net_params.use_cuda
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        return torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=net_params.batch_size, shuffle=True, **kwargs)

    @staticmethod
    def get_test_loader_mnist(net_params):
        use_cuda = net_params.use_cuda
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        return torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=net_params.test_batch_size, shuffle=True, **kwargs)

    @staticmethod
    def get_train_loader_mnist_with_flat_divisions(net_params):
        use_cuda = net_params.use_cuda
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        return torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda tensor: flat_divisions(tensor,
                                                                net_params.get_divides_in_row(),
                                                                net_params.get_divides_in_col()))
                           ])),
            batch_size=net_params.batch_size, shuffle=True, **kwargs)

    @staticmethod
    def get_test_loader_mnist_with_flat_divisions(net_params):
        use_cuda = net_params.use_cuda
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        return torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda tensor: flat_divisions(tensor,
                                                                net_params.get_divides_in_row(),
                                                                net_params.get_divides_in_col()))
            ])),
            batch_size=net_params.test_batch_size, shuffle=True, **kwargs)

    @staticmethod
    def get_train_loader_fashion_mnist(net_params):
        use_cuda = net_params.use_cuda
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        return torch.utils.data.DataLoader(
            FashionMnist('../data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3205,)),
                           ])),
            batch_size=net_params.batch_size, shuffle=True, **kwargs)

    @staticmethod
    def get_test_loader_fashion_mnist(net_params):
        use_cuda = net_params.use_cuda
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        return torch.utils.data.DataLoader(
            FashionMnist('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3205,)),
            ])),
            batch_size=net_params.test_batch_size, shuffle=True, **kwargs)


    @staticmethod
    def get_train_loader_fashion_mnist_with_flat_divisions(net_params):
        use_cuda = net_params.use_cuda
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        return torch.utils.data.DataLoader(
            FashionMnist('../data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3205,)),
                transforms.Lambda(lambda tensor: flat_divisions(tensor,
                                                                net_params.get_divides_in_row(),
                                                                net_params.get_divides_in_col()))
                           ])),
            batch_size=net_params.batch_size, shuffle=True, **kwargs)

    @staticmethod
    def get_test_loader_fashion_mnist_with_flat_divisions(net_params):
        use_cuda = net_params.use_cuda
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        return torch.utils.data.DataLoader(
            FashionMnist('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3205,)),
                transforms.Lambda(lambda tensor: flat_divisions(tensor,
                                                                net_params.get_divides_in_row(),
                                                                net_params.get_divides_in_col()))
            ])),
            batch_size=net_params.test_batch_size, shuffle=True, **kwargs)

    @staticmethod
    def get_train_loader_kuzushiji_mnist(net_params):
        use_cuda = net_params.use_cuda
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        return torch.utils.data.DataLoader(
            KuzushijiMnist('../data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=net_params.batch_size, shuffle=True, **kwargs)

    @staticmethod
    def get_test_loader_kuzushiji_mnist(net_params):
        use_cuda = net_params.use_cuda
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        return torch.utils.data.DataLoader(
            KuzushijiMnist('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=net_params.test_batch_size, shuffle=True, **kwargs)

    @staticmethod
    def get_train_loader_kuzushiji_mnist_with_flat_divisions(net_params):
        use_cuda = net_params.use_cuda
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        return torch.utils.data.DataLoader(
            KuzushijiMnist('../data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda tensor: flat_divisions(tensor,
                                                                net_params.get_divides_in_row(),
                                                                net_params.get_divides_in_col()))
            ])),
            batch_size=net_params.batch_size, shuffle=True, **kwargs)

    @staticmethod
    def get_test_loader_kuzushiji_mnist_with_flat_divisions(net_params):
        use_cuda = net_params.use_cuda
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        return torch.utils.data.DataLoader(
            KuzushijiMnist('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda tensor: flat_divisions(tensor,
                                                                net_params.get_divides_in_row(),
                                                                net_params.get_divides_in_col()))
            ])),
            batch_size=net_params.test_batch_size, shuffle=True, **kwargs)

    @staticmethod
    def get_train_loader_cifar(net_params):
        use_cuda = net_params.use_cuda
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        return torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=True, download=True, transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])),
            batch_size=net_params.batch_size, shuffle=True, **kwargs)

    @staticmethod
    def get_test_loader_cifar(net_params):
        use_cuda = net_params.use_cuda
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        return torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])),
            batch_size=net_params.test_batch_size, shuffle=True, **kwargs)

    @staticmethod
    def get_train_loader_cifar_with_flat_divisions(net_params):
        use_cuda = net_params.use_cuda
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        return torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.Lambda(lambda tensor: tensor.reshape(1, 3, 1024)),
                transforms.Lambda(lambda tensor: flat_divisions(tensor,
                                                                net_params.get_divides_in_row(),
                                                                net_params.get_divides_in_col()))
            ])),
            batch_size=net_params.batch_size, shuffle=True, **kwargs)

    @staticmethod
    def get_test_loader_cifar_with_flat_divisions(net_params):
        use_cuda = net_params.use_cuda
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        return torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.Lambda(lambda tensor: tensor.reshape(1, 3, 1024)),
                transforms.Lambda(lambda tensor: flat_divisions(tensor,
                                                                net_params.get_divides_in_row(),
                                                                net_params.get_divides_in_col()))
            ])),
            batch_size=net_params.test_batch_size, shuffle=True, **kwargs)

    @staticmethod
    def get_train_loader_cifar_with_parallel_flat_divisions(net_params):
        use_cuda = net_params.use_cuda
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        return torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Lambda(lambda tensor: parallel_flat_division(tensor, net_params.get_n_parallel_nets(),
                                                                        net_params.get_divides_in_row(),
                                                                        net_params.get_divides_in_col()))
            ])),
            batch_size=net_params.batch_size, shuffle=True, **kwargs)

    @staticmethod
    def get_test_loader_cifar_with_parallel_flat_divisions(net_params):
        use_cuda = net_params.use_cuda
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        return torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Lambda(lambda tensor: parallel_flat_division(tensor,net_params.get_n_parallel_nets(),
                                                                net_params.get_divides_in_row(),
                                                                net_params.get_divides_in_col()))
            ])),
            batch_size=net_params.test_batch_size, shuffle=True, **kwargs)

    @staticmethod
    def get_imdb_data_train_data(text,label):
        train_data, test_data = datasetsText.IMDB.splits(text, label)
        return train_data,test_data