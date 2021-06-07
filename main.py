from __future__ import print_function

import csv
from datetime import datetime
from random import randint

import torch
import torch.nn.functional as F

from Nets.ParallelizedTensorsNet import ParallelizedTensorNet
from Nets.TRNetSerialized import TRNetSerialized, TRNetSerializedCell
from Utils import Constant, EnumModel, EnumDataset, EnumTrainMethods
from Utils.NetParamsFactory import NetParamsFactory
from Utils.PreProcessText import PreProcessText
from Utils.DataLoaderUtil import DataLoaderUtil
from Utils.NetParams import NetParams
from Utils.OptimizerFactory import OptimizerFactory
from Utils.TimerUtil import TimerUtil


def run_net(net_params, result_data):
    # Training settings
    timerUtil = TimerUtil()
    torch.manual_seed(net_params.seed)

    device = net_params.device
    train_loader, test_loader = net_params.get_dataset()
    model = net_params.get_model()
    model.to(device)
    train_method, test_method = net_params.get_train_and_test_method()

    time_str = timerUtil.get_time_string()
    filename = net_params.get_result_data_file_name()
    csvfile = open("./Runs/" + time_str + "_" + str(filename) + '.csv', 'w')
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(("model_label", "dataset_label", "divides_in_row", "divides_in_col", "rank", "m", "optimizer",
                         "learning_rate", "batch_size", "init_value", "number of parameters", "Dropout" ,"loss",
                         "percentage", "epoch_time", "total_time"))

    optimizer = OptimizerFactory.create_optimizer(model, net_params)
    key = net_params.get_result_data_key()
    result_data[key] = []
    for epoch in range(1, net_params.get_epochs() + 1):
        timerUtil.start_epoch()
        train_method(model, device, train_loader, optimizer, epoch, net_params.get_log_while_training(),
                     net_params.get_log_interval())

        loss, percentage = test_method(model, device, test_loader)
        timerUtil.end_epoch()
        timerUtil.print_difference_epoch(epoch, Constant.EPOCHS)
        timerUtil.print_difference_from_start()
        result_data[key].append([loss, percentage, timerUtil.get_epoch_delta(), timerUtil.get_total_delta()])
        list_key = net_params.get_result_data_list()
        csv_writer.writerow(list_key + [loss, percentage, timerUtil.get_epoch_delta(), timerUtil.get_total_delta()])
        csvfile.flush()  # whenever you want

    if net_params.save_model():
        torch.save(model.state_dict(), Constant.MODEL_FILENAME)
    print(result_data)
    csvfile.close()
    del model
    del test_loader
    del train_loader


def main():
    print("Hello World!")


if __name__ == '__main__':
    main()
