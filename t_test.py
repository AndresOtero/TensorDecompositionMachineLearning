from scipy import stats
import plotOutput as pltOutput
from Utils import Constant, EnumModel, EnumDataset, EnumTrainMethods
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

TR_FASHION = [('TR_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '8', '64', 'Adam', '0.0001', '256', '1', '58496'),
              ('TR_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '16', '64', 'Adam', '0.0001', '256', '1',
               '121984'), (
                  'TR_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '32', '64', 'Adam', '0.0001', '256', '1',
                  '273536'), (
                  'TR_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '64', '64', 'Adam', '0.0001', '256', '1',
                  '674944'), (
                  'TR_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '90', '64', 'Adam', '0.0001', '256', '1',
                  '1097600'), (
                  'TR_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '95', '64', 'Adam', '0.0001', '256', '1',
                  '1188800'), (
                  'TR_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '100', '64', 'Adam', '0.0001', '256', '1',
                  '1283200')]
TT_FASHION = [
    ('TT_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '25', '64', 'Adam', '0.0001', '256', '1', '60800'),
    ('TT_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '38', '64', 'Adam', '0.0001', '256', '1', '122368'),
    ('TT_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '60', '64', 'Adam', '0.0001', '256', '1', '275840'),
    ('TT_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '97', '64', 'Adam', '0.0001', '256', '1', '673664'),
    ('TT_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '125', '64', 'Adam', '0.0001', '256', '1', '1091200'),
    ('TT_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '128', '64', 'Adam', '0.0001', '256', '1', '1141888'),
    ('TT_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '135', '64', 'Adam', '0.0001', '256', '1', '1264640')]
TR_MNIST = [('TR_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '8', '64', 'Adam', '0.0001', '256', '1', '58496'),
            ('TR_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '16', '64', 'Adam', '0.0001', '256', '1', '121984'),
            ('TR_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '32', '64', 'Adam', '0.0001', '256', '1', '273536'),
            ('TR_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '64', '64', 'Adam', '0.0001', '256', '1', '674944'),
            ('TR_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '90', '64', 'Adam', '0.0001', '256', '1', '1097600'),
            ('TR_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '95', '64', 'Adam', '0.0001', '256', '1', '1188800'),
            ('TR_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '100', '64', 'Adam', '0.0001', '256', '1', '1283200')]
TT_MNIST = [('TT_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '25', '64', 'Adam', '0.0001', '256', '1', '60800'),
            ('TT_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '38', '64', 'Adam', '0.0001', '256', '1', '122368'),
            ('TT_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '60', '64', 'Adam', '0.0001', '256', '1', '275840'),
            ('TT_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '97', '64', 'Adam', '0.0001', '256', '1', '673664'),
            ('TT_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '125', '64', 'Adam', '0.0001', '256', '1', '1091200'),
            ('TT_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '128', '64', 'Adam', '0.0001', '256', '1', '1141888'),
            ('TT_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '135', '64', 'Adam', '0.0001', '256', '1', '1264640')]

TR_FASHION_32 = [
    ('TR_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '8', '32', 'Adam', '0.0001', '256', '1', '29248'),
    ('TR_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '16', '32', 'Adam', '0.0001', '256', '1', '60992'),
    ('TR_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '32', '32', 'Adam', '0.0001', '256', '1', '136768'),
    ('TR_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '64', '32', 'Adam', '0.0001', '256', '1', '337472')]
TT_FASHION_32 = [
    ('TT_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '8', '32', 'Adam', '0.0001', '256', '1', '6464'),
    ('TT_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '16', '32', 'Adam', '0.0001', '256', '1', '15424'),
    ('TT_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '32', '32', 'Adam', '0.0001', '256', '1', '45632'),
    ('TT_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '64', '32', 'Adam', '0.0001', '256', '1', '155200')]

TT_FASHION_ALT = [
    ('TT_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '8', '64', 'Adam', '0.0001', '256', '1', '12928'),
    ('TT_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '16', '64', 'Adam', '0.0001', '256', '1', '30848'),
    ('TT_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '32', '64', 'Adam', '0.0001', '256', '1', '91264'),
    ('TT_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '64', '64', 'Adam', '0.0001', '256', '1', '310400'),
    ('TT_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '140', '64', 'Adam', '0.0001', '256', '1', '1356160'),
    ('TT_SHARED', 'FASHION_MNIST_FLAT_DIVISIONS', '4', '4', '166', '64', 'Adam', '0.0001', '256', '1', '1883648')]

TR_MNIST_32 = [('TR_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '8', '32', 'Adam', '0.0001', '256', '1', '29248'),
               ('TR_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '16', '32', 'Adam', '0.0001', '256', '1', '60992'),
               ('TR_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '32', '32', 'Adam', '0.0001', '256', '1', '136768'),
               ('TR_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '64', '32', 'Adam', '0.0001', '256', '1', '337472')]
TT_MNIST_32 = [('TT_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '8', '32', 'Adam', '0.0001', '256', '1', '6464'),
               ('TT_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '16', '32', 'Adam', '0.0001', '256', '1', '15424'),
               ('TT_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '32', '32', 'Adam', '0.0001', '256', '1', '45632'),
               ('TT_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '64', '32', 'Adam', '0.0001', '256', '1', '155200')]

TT_MNIST_ALT = [('TT_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '8', '64', 'Adam', '0.0001', '256', '1', '12928'),
                ('TT_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '16', '64', 'Adam', '0.0001', '256', '1', '30848'),
                ('TT_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '32', '64', 'Adam', '0.0001', '256', '1', '91264'),
                ('TT_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '64', '64', 'Adam', '0.0001', '256', '1', '310400'),
                ('TT_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '140', '64', 'Adam', '0.0001', '256', '1', '1356160'),
                ('TT_SHARED', 'MNIST_FLAT_DIVISIONS', '4', '4', '166', '64', 'Adam', '0.0001', '256', '1', '1883648')]

LSTM_MNIST_32 = [('LSTM', 'MNIST_FLAT_DIVISIONS', '2', '2', '8', '32', 'Adam', '0.0001', '256', '1', '9466'),
                 ('LSTM', 'MNIST_FLAT_DIVISIONS', '2', '2', '16', '32', 'Adam', '0.0001', '256', '1', '16202'),
                 ('LSTM', 'MNIST_FLAT_DIVISIONS', '2', '2', '32', '32', 'Adam', '0.0001', '256', '1', '40426'),
                 ('LSTM', 'MNIST_FLAT_DIVISIONS', '2', '2', '64', '32', 'Adam', '0.0001', '256', '1', '131882')]
LSTM_MNIST = [('LSTM', 'MNIST_FLAT_DIVISIONS', '2', '2', '8', '64', 'Adam', '0.0001', '256', '1', '16794'),
              ('LSTM', 'MNIST_FLAT_DIVISIONS', '2', '2', '16', '64', 'Adam', '0.0001', '256', '1', '24554'),
              ('LSTM', 'MNIST_FLAT_DIVISIONS', '2', '2', '32', '64', 'Adam', '0.0001', '256', '1', '50826'),
              ('LSTM', 'MNIST_FLAT_DIVISIONS', '2', '2', '64', '64', 'Adam', '0.0001', '256', '1', '146378')] + [
                 ('LSTM', 'MNIST_FLAT_DIVISIONS', '2', '2', '91', '64', 'Adam', '0.0001', '256', '1', '271604'),
                 ('LSTM', 'MNIST_FLAT_DIVISIONS', '2', '2', '149', '64', 'Adam', '0.0001', '256', '1', '678648')]
LSTM_Fashion_32 = [('LSTM', 'FASHION_MNIST_FLAT_DIVISIONS', '2', '2', '8', '32', 'Adam', '0.0001', '256', '1', '9466'),
                   (
                       'LSTM', 'FASHION_MNIST_FLAT_DIVISIONS', '2', '2', '16', '32', 'Adam', '0.0001', '256', '1',
                       '16202'),
                   (
                       'LSTM', 'FASHION_MNIST_FLAT_DIVISIONS', '2', '2', '32', '32', 'Adam', '0.0001', '256', '1',
                       '40426'),
                   ('LSTM', 'FASHION_MNIST_FLAT_DIVISIONS', '2', '2', '64', '32', 'Adam', '0.0001', '256', '1',
                    '131882')]
LSTM_Fashion = [('LSTM', 'FASHION_MNIST_FLAT_DIVISIONS', '2', '2', '8', '64', 'Adam', '0.0001', '256', '1', '16794'),
                ('LSTM', 'FASHION_MNIST_FLAT_DIVISIONS', '2', '2', '16', '64', 'Adam', '0.0001', '256', '1', '24554'),
                ('LSTM', 'FASHION_MNIST_FLAT_DIVISIONS', '2', '2', '32', '64', 'Adam', '0.0001', '256', '1', '50826'), (
                    'LSTM', 'FASHION_MNIST_FLAT_DIVISIONS', '2', '2', '64', '64', 'Adam', '0.0001', '256', '1',
                    '146378')] + [
                   ('LSTM', 'FASHION_MNIST_FLAT_DIVISIONS', '2', '2', '91', '64', 'Adam', '0.0001', '256',
                    '1', '271604'), (
                       'LSTM', 'FASHION_MNIST_FLAT_DIVISIONS', '2', '2', '149', '64', 'Adam', '0.0001', '256',
                       '1', '678648')]

RNN_MNIST_32 = [('RNN', 'MNIST_FLAT_DIVISIONS', '2', '2', '8', '32', 'Adam', '0.0001', '256', '1', '7162'),
                ('RNN', 'MNIST_FLAT_DIVISIONS', '2', '2', '16', '32', 'Adam', '0.0001', '256', '1', '8906'),
                ('RNN', 'MNIST_FLAT_DIVISIONS', '2', '2', '32', '32', 'Adam', '0.0001', '256', '1', '15082'),
                ('RNN', 'MNIST_FLAT_DIVISIONS', '2', '2', '64', '32', 'Adam', '0.0001', '256', '1', '38186')]
RNN_MNIST = [('RNN', 'MNIST_FLAT_DIVISIONS', '2', '2', '8', '64', 'Adam', '0.0001', '256', '1', '13722'),
             ('RNN', 'MNIST_FLAT_DIVISIONS', '2', '2', '16', '64', 'Adam', '0.0001', '256', '1', '15722'),
             ('RNN', 'MNIST_FLAT_DIVISIONS', '2', '2', '32', '64', 'Adam', '0.0001', '256', '1', '22410'),
             ('RNN', 'MNIST_FLAT_DIVISIONS', '2', '2', '64', '64', 'Adam', '0.0001', '256', '1', '46538')] + [
                ('RNN', 'MNIST_FLAT_DIVISIONS', '2', '2', '187', '64', 'Adam', '0.0001', '256', '1', '272735'),
                ('RNN', 'MNIST_FLAT_DIVISIONS', '2', '2', '302', '64', 'Adam', '0.0001', '256', '1', '675810')]
RNN_FASHION_32 = [('RNN', 'FASHION_MNIST_FLAT_DIVISIONS', '2', '2', '8', '32', 'Adam', '0.0001', '256', '1', '7162'),
                  ('RNN', 'FASHION_MNIST_FLAT_DIVISIONS', '2', '2', '16', '32', 'Adam', '0.0001', '256', '1', '8906'),
                  ('RNN', 'FASHION_MNIST_FLAT_DIVISIONS', '2', '2', '32', '32', 'Adam', '0.0001', '256', '1', '15082'),
                  ('RNN', 'FASHION_MNIST_FLAT_DIVISIONS', '2', '2', '64', '32', 'Adam', '0.0001', '256', '1', '38186')]
RNN_FASHION = [('RNN', 'FASHION_MNIST_FLAT_DIVISIONS', '2', '2', '8', '64', 'Adam', '0.0001', '256', '1', '13722'),
               ('RNN', 'FASHION_MNIST_FLAT_DIVISIONS', '2', '2', '16', '64', 'Adam', '0.0001', '256', '1', '15722'),
               ('RNN', 'FASHION_MNIST_FLAT_DIVISIONS', '2', '2', '32', '64', 'Adam', '0.0001', '256', '1', '22410'),
               ('RNN', 'FASHION_MNIST_FLAT_DIVISIONS', '2', '2', '64', '64', 'Adam', '0.0001', '256', '1', '46538')] + [
                  (
                      'RNN', 'FASHION_MNIST_FLAT_DIVISIONS', '2', '2', '187', '64', 'Adam', '0.0001', '256', '1',
                      '272735'),
                  (
                      'RNN', 'FASHION_MNIST_FLAT_DIVISIONS', '2', '2', '302', '64', 'Adam', '0.0001', '256', '1',
                      '675810')]

CONV_TR_CIFAR = [('ConvTR', 'CIFAR', '2', '2', '16', '32', 'Adam', '0.0001', '256', '1', '1219104'),
                 ('ConvTR', 'CIFAR', '2', '2', '16', '64', 'Adam', '0.0001', '256', '1', '1311296'),
                 ('ConvTR', 'CIFAR', '2', '2', '16', '128', 'Adam', '0.0001', '256', '1', '1495680')]
CONV_TT_CIFAR = [('ConvTT', 'CIFAR', '2', '2', '38', '32', 'Adam', '0.0001', '256', '1', '1219296'),
                 ('ConvTT', 'CIFAR', '2', '2', '38', '64', 'Adam', '0.0001', '256', '1', '1311680'),
                 ('ConvTT', 'CIFAR', '2', '2', '38', '128', 'Adam', '0.0001', '256', '1', '1496448')]
NET_CIFAR = [('NET', 'CIFAR', '2', '2', '0', '0', 'Adam', '0.0001', '256', '1', '1258346')]

TT_CONV_IMDB = [('TENSOR_TEXT_NET_CONV', 'IMDB', '3', '4', '50', '100', 'Adam', '0.0001', '250', '1', '1375400')]
TR_CONV_IMDB = [('TENSOR_RING_TEXT_NET_CONV', 'IMDB', '3', '4', '50', '100', 'Adam', '0.0001', '250', '1', '1377900')]
CONV_IMDB = [('CNN_TEXT', 'IMDB', '5', '1', '0', '50', 'Adam', '0.0001', '250', '1', '1310701')]


def loadRuns():
    paths_run = [Path('./Runs'), Path('./Runs_to_use')]
    # paths_run = [Path('./Runs')]

    paths = []
    for p in paths_run:
        subdirectories = [x for x in p.iterdir() if x.is_dir()]
        paths += [str(path.resolve()) for dirpath in subdirectories for path in dirpath.rglob('*.csv')]

    dicc = {}
    i = 0
    for path in paths:
        path_dicc = {}
        pltOutput.csv_reader(path, path_dicc)
        dicc[i] = path_dicc
        i += 1
    runs_dicc = {}

    for run in dicc:
        dicc_key = dicc[run].keys()
        if not list(dicc_key):
            continue
        run_key = list(dicc_key)[0]
        if run_key in runs_dicc:
            runs_dicc[run_key][len(runs_dicc[run_key])] = dicc[run][run_key]
        else:
            runs_dicc[run_key] = {0: dicc[run][run_key]}
    return runs_dicc


def GetLines(RunsKeys, Run_x, runs_dicc, params, minimum_run_number, run, x_axis_index=Constant.NET_PARAMETERS_INDEX,
             best_n_runs=1, last_run=False):
    for run_number in range(minimum_run_number):
        Run = runs_dicc[RunsKeys[run]][run_number]
        if not last_run:
            Run.sort(key=lambda item: -float(item[1]))
        else:
            Run.reverse()
        if (len(Run[0]) == 5):
            acc_index = 2
        else:
            acc_index = 1
        Run = np.array([Run[x][acc_index] for x in range(best_n_runs)]).astype(np.float)
        Run_x = np.concatenate((Run_x, Run))
        param = np.array([RunsKeys[run][x_axis_index] for x in range(best_n_runs)]).astype(np.float)
        params = np.concatenate((params, param))
    return Run_x, params


def GetLines3D(RunsKeys, Run_x, runs_dicc, params, minimum_run_number, run, x_axis_index=Constant.NET_M_INDEX,
               y_axis_index=Constant.NET_RANK_INDEX, best_n_runs=1, last_run=False):
    for run_number in range(minimum_run_number):
        Run = runs_dicc[RunsKeys[run]][run_number]
        if not last_run:
            Run.sort(key=lambda item: -float(item[1]))
        else:
            Run.reverse()
        Run = np.array([Run[x][1] for x in range(best_n_runs)]).astype(np.float)
        Run_x = np.concatenate((Run_x, Run))
        param = np.array(
            [(RunsKeys[run][x_axis_index], RunsKeys[run][y_axis_index]) for x in range(best_n_runs)]).astype(np.float)
        params = np.append(params, param)
    return Run_x, params


def GetRuns(RunsKeys, x_axis_index=Constant.NET_PARAMETERS_INDEX, best_n_runs=1, last_run=False, dropout=None):
    runs_dicc = loadRuns()
    run_x = np.array([]).astype(np.float)
    params = np.array([]).astype(np.float)
    for run in range(len(RunsKeys)):
        if dropout:
            keys_to_be_deleted = []
            for k in runs_dicc[RunsKeys[run]]:
                if float(runs_dicc[RunsKeys[run]][k][0][0]) != dropout:
                    keys_to_be_deleted.append(k)
            for k in keys_to_be_deleted:
                del runs_dicc[RunsKeys[run]][k]
        minimum_run_number = len(runs_dicc[RunsKeys[run]])
        run_x, params = GetLines(RunsKeys, run_x, runs_dicc, params, minimum_run_number, run, x_axis_index=x_axis_index,
                                 best_n_runs=best_n_runs, last_run=last_run)
    return run_x, params


def GetRuns3D(RunsKeys, x_axis_index=Constant.NET_PARAMETERS_INDEX, y_axis_index=Constant.NET_RANK_INDEX, best_n_runs=1,
              last_run=False):
    runs_dicc = loadRuns()
    run_x = np.array([]).astype(np.float)
    params = np.array([]).astype(np.float)
    for run in range(len(RunsKeys)):
        minimum_run_number = len(runs_dicc[RunsKeys[run]])
        run_x, params = GetLines3D(RunsKeys, run_x, runs_dicc, params, minimum_run_number, run,
                                   x_axis_index=x_axis_index, y_axis_index=y_axis_index, best_n_runs=best_n_runs,
                                   last_run=last_run)
    params1 = [params[x] for x in range(0, len(params), 2)]
    params2 = [params[x] for x in range(1, len(params), 2)]
    return run_x, params1, params2


def DoTTest(List_TR_Runs_Keys, List_TT_Runs_Keys):
    runs_dicc = loadRuns()
    TR_run_x = np.array([]).astype(np.float)
    TT_run_x = np.array([]).astype(np.float)
    TR_params = np.array([]).astype(np.float)
    TT_params = np.array([]).astype(np.float)

    for run in range(len(List_TR_Runs_Keys)):
        minimum_run_number = min([len(runs_dicc[List_TR_Runs_Keys[run]]), len(runs_dicc[List_TT_Runs_Keys[run]])])
        TR_run_x, TR_params = GetLines(List_TR_Runs_Keys, TR_run_x, runs_dicc, TR_params, minimum_run_number, run,
                                       x_axis_index=10, best_n_runs=1)
        TT_run_x, TT_params = GetLines(List_TT_Runs_Keys, TT_run_x, runs_dicc, TT_params, minimum_run_number, run,
                                       x_axis_index=10, best_n_runs=1)
    print(stats.ttest_ind(TR_run_x, TT_run_x))

    plt.plot(TR_params, TT_run_x, 'g^', TR_params, TR_run_x, 'bs')
    plt.show()


def main():
    DoTTest(TR_FASHION, TT_FASHION)
    DoTTest(TR_MNIST, TT_MNIST)
    DoTTest(CONV_TR_CIFAR, CONV_TT_CIFAR)
    DoTTest(TT_CONV_IMDB,TR_CONV_IMDB)

if __name__ == '__main__':
    main()
