
class RanksFactory:
    @staticmethod
    def create_tensor_ring_ranks(net_params):
        return [net_params.rank for x in range(net_params.get_amount_of_divisions() + 1)]

    @staticmethod
    def create_tensor_train_serial_ranks(net_params):
        return [1] + [net_params.rank for x in range(net_params.get_amount_of_divisions() - 1)] \
               + [net_params.categories]

    @staticmethod
    def create_tensor_train_shared_ranks(net_params):
        return [1] + [net_params.rank for x in range(2)] \
               + [net_params.categories]

    @staticmethod
    def create_tensor_ring_shared_ranks(net_params):
        return [net_params.rank_first_and_last]+[net_params.rank for x in range(2)]+[net_params.rank_first_and_last]

    @staticmethod
    def create_tensor_train_shared_parallel_ranks(net_params):
        return [1] + [net_params.rank for x in range(2)] \
               + [1]

    @staticmethod
    def create_tensor_train_parallel_ranks(net_params):
        return [1] + [net_params.rank for x in range(net_params.get_amount_of_divisions() - 1)] \
               + [1]