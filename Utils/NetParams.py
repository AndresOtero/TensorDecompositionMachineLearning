import torch

from Utils import Constant as C, EnumDataset, EnumModel, EnumTrainMethods


class NetParams:
    def __init__(self,
                 model,
                 dataset,
                 train_method,
                 learning_rate,
                 optimizer,
                 cuda_is_available,
                 m=C.M,
                 divides_in_row=C.DIVIDES_IN_ROW,
                 divides_in_col=C.DIVIDES_IN_COL,
                 rank=C.RANK,
                 batch_size=C.BATCH_SIZE,
                 test_batch_size=C.TEST_BATCH_SIZE,
                 epochs=C.EPOCHS,
                 momentum=C.MOMENTUM,
                 seed=C.SEED,
                 log_interval=C.LOG_INTERVAL,
                 save=C.SAVE_MODEL,
                 cuda=C.USE_CUDA,
                 categories=C.CATEGORIES,
                 log_while_training=C.LOG_WHILE_TRAINING,
                 tensor_size=C.TENSOR_SIZE,
                 n_parallel_nets=C.N_PARALLEL_NETS,
                 embedding=C.EMBEDDING_DIM,
                 fixed_length=C.FIXED_LENGTH,
                 init_value=C.INIT_VALUE,
                 rank_first_and_last=C.RANK_FIRST_AND_LAST,
                 dropout = C.DROPOUT,
                 tensor_net=None):
        if tensor_net is not None:
            self.dataset_label = dataset, tensor_net
            self.tensor_net = EnumModel.MODELS[tensor_net]
        else:
            self.dataset_label = dataset
        self.init_value=init_value
        self.save = save
        self.log_interval = log_interval
        self.train_method = train_method
        self.seed = seed
        torch.manual_seed(seed)
        self.momentum = momentum
        self.epochs = epochs
        self.m = m
        self.divides_in_row = divides_in_row
        self.divides_in_col = divides_in_col
        self.amount_of_divisions = divides_in_row * divides_in_col
        self.rank = rank
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.cuda = cuda
        self.categories = categories
        self.log_while_training = log_while_training
        self.use_cuda = cuda and cuda_is_available
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.tensor_size = tensor_size
        self.n = int(tensor_size / self.amount_of_divisions)
        self.n_parallel_nets = n_parallel_nets
        self.model_label = model
        model = EnumModel.MODELS[model]
        self.embedding = embedding
        self.fixed_length=fixed_length
        self.dropout = dropout
        self.rank_first_and_last=rank_first_and_last
        self.dataset = self._create_dataset(EnumDataset.DATASETS[dataset])
        self.model = model(self)

    def get_dropout(self):
        return self.dropout

    def get_fixed_length(self):
        return  self.fixed_length

    def get_embedding(self):
        return self.embedding

    def get_train_and_test_method(self):
        (train_method, test_method) = EnumTrainMethods.TRAIN_METHODS[self.train_method]
        return train_method, test_method

    def _create_dataset(self, dataset_methods):
        datasets = []
        for get_method in dataset_methods:
            dataset = get_method(self)
            if type(dataset) == list:
                datasets = dataset
            else:
                datasets.append(dataset)
        return datasets[0], datasets[1]

    def get_dataset(self):
        return self.dataset

    def get_model(self):
        return self.model

    def get_tensor_net(self):
        return self.tensor_net

    def get_n_parallel_nets(self):
        return self.n_parallel_nets

    def get_use_cuda(self):
        return self.use_cuda

    def get_device(self):
        return self.device

    def get_log_while_training(self):
        return self.log_while_training

    def get_amount_of_categories(self):
        return self.categories

    def save_model(self):
        return self.save

    def get_log_interval(self):
        return self.log_interval

    def get_seed(self):
        return self.seed

    def get_momentum(self):
        return self.momentum

    def get_epochs(self):
        return self.epochs

    def get_cuda(self):
        return self.cuda

    def get_m(self):
        return self.m

    def get_n(self):
        return self.n

    def get_divides_in_row(self):
        return self.divides_in_row

    def get_divides_in_col(self):
        return self.divides_in_col

    def get_amount_of_divisions(self):
        return self.amount_of_divisions

    def get_rank(self):
        return self.rank

    def get_learning_rate(self):
        return self.learning_rate

    def get_optimizer(self):
        return self.optimizer

    def get_batch_size(self):
        return self.batch_size

    def get_result_data_key(self):
        return self.model_label, self.dataset_label, self.divides_in_row, self.divides_in_col, self.rank, \
               self.m, self.optimizer, self.learning_rate, self.batch_size,self.init_value,self.model.get_number_of_parameters(),\
               self.get_dropout()

    def get_result_data_key_str(self):
        return self.model_label, self.dataset_label, str(self.divides_in_row), str(self.divides_in_col), str(self.rank), \
               str(self.m), str(self.optimizer), str(self.learning_rate), str(self.batch_size),str(self.init_value)\
            ,str(self.model.get_number_of_parameters()),str(self.get_dropout())

    def get_result_data_file_name(self):
        return str(self.model_label) + "-" + str(self.dataset_label) + "-" + str(self.divides_in_row) + "-" + str(
            self.divides_in_col) \
               + "-" + str(self.rank) + "-" + str(self.m) + "-" + str(self.optimizer) + "-" + str(self.learning_rate) \
               + "-" + str(self.batch_size) + "-" + str(self.init_value) + "-" + str(self.dropout)

    def get_result_data_list(self):
        return [self.model_label, self.dataset_label, self.divides_in_row, self.divides_in_col, self.rank, self.m,
                self.optimizer, self.learning_rate, self.batch_size,self.init_value,self.model.get_number_of_parameters(),
                self.dropout]

    def print_basic_params(self):
        print("Parameters:", "Model:", self.model_label, "; Dataset:", self.dataset_label, "; Divides in row:",
              self.divides_in_row, "; Divides in col:", self.divides_in_col, "; M:", self.m, "; Rank:", self.rank,
              "; Learning rate:", self.learning_rate, "; Optimizer:", self.optimizer,"; Init value:", self.init_value,
              "; Number of parameters:", self.model.get_number_of_parameters(),"; Dropout:",self.model.get_dropout())

    def print_n_params(self):
        print("Parameters:", "Model:", self.model_label, "; M:", self.m, "; Rank:", self.rank,
              "; Number of parameters:", self.model.get_number_of_parameters())

