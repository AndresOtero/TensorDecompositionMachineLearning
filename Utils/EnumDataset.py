from Utils.DataLoaderUtil import DataLoaderUtil as DL
from Utils.PreProcessText import PreProcessText as PPT

DATASETS = {"MNIST": [DL.get_train_loader_mnist, DL.get_test_loader_mnist],
            "MNIST_FLAT_DIVISIONS": [DL.get_train_loader_mnist_with_flat_divisions,
                                     DL.get_test_loader_mnist_with_flat_divisions],
            "FASHION_MNIST": [DL.get_train_loader_fashion_mnist, DL.get_test_loader_fashion_mnist],
            "FASHION_MNIST_FLAT_DIVISIONS": [DL.get_train_loader_fashion_mnist_with_flat_divisions,
                                        DL.get_test_loader_fashion_mnist_with_flat_divisions],
            "KUZUSHIJI_MNIST": [DL.get_train_loader_kuzushiji_mnist, DL.get_test_loader_kuzushiji_mnist],
            "KUZUSHIJI_MNIST_FLAT_DIVISIONS": [DL.get_train_loader_kuzushiji_mnist,
                                    DL.get_test_loader_kuzushiji_mnist_with_flat_divisions],
            "CIFAR": [DL.get_train_loader_cifar, DL.get_test_loader_cifar],
            "CIFAR_FLAT_DIVISIONS": [DL.get_train_loader_cifar_with_flat_divisions,
                                DL.get_test_loader_cifar_with_flat_divisions],
            "CIFAR_PARALLEL_DIVISIONS": [DL.get_train_loader_cifar_with_parallel_flat_divisions,
                                         DL.get_test_loader_cifar_with_parallel_flat_divisions],
            "IMDB": [PPT.get_instance().pre_process_imdb_dataset],
            "IMDB_WITH_LENGTHS": [PPT.get_instance().pre_process_imdb_dataset_include_lengths]

            }
MNIST = "MNIST"
MNIST_FLAT_DIVISIONS = "MNIST_FLAT_DIVISIONS"
FASHION_MNIST = "FASHION_MNIST"
FASHION_MNIST_FLAT_DIVISIONS = "FASHION_MNIST_FLAT_DIVISIONS"
KUZUSHIJI_MNIST = "KUZUSHIJI_MNIST"
KUZUSHIJI_MNIST_FLAT_DIVISIONS = "KUZUSHIJI_MNIST_FLAT_DIVISIONS"
CIFAR = "CIFAR"
CIFAR_FLAT_DIVISIONS = "CIFAR_FLAT_DIVISIONS"
CIFAR_PARALLEL_DIVISIONS = "CIFAR_PARALLEL_DIVISIONS"
IMDB = "IMDB"
IMDB_WITH_LENGTHS="IMDB_WITH_LENGTHS"
