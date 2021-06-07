from torch import optim

from Utils import Constant


class OptimizerFactory:
    @staticmethod
    def create_optimizer(model, net_params):
        if net_params.optimizer == Constant.ADAM:
            optimizer = optim.Adam(model.parameters(), lr=net_params.get_learning_rate())  # , momentum=args.momentum)
        if net_params.optimizer == Constant.SGD:
            optimizer = optim.SGD(model.parameters(), lr=net_params.get_learning_rate(),
                                  momentum=net_params.get_momentum())
        return optimizer