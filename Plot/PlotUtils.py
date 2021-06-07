from Utils import Constant
from Plot.PlotHelper import *


def scatter_3d_run(run, label_title, zLimMin=0, zLimMax=0, legend="", plot_only_last=False):
    x = []
    y = []
    z = []
    z2 = []
    z3 = []
    for k in run:
        for r in run[k]:
            if zLimMax == 0 or (zLimMax > float(r[1]) > zLimMin):
                x.append(int(k[4]))  # rank
                y.append(int(k[5]))  # m
                z.append(float(r[1]))
                z2.append(float(r[2]))
                z3.append(float(r[3]))


    PlotHelper.scatter3d(x, y, z, label_x="rank", label_y="m", label_z="accuracy", title=label_title + "_accuracy",
                         show=True, legend=legend)
    PlotHelper.scatter3d(x, y, z2, label_x="rank", label_y="m", label_z="time", title=label_title + "_time", show=True,
                         legend=legend)
    PlotHelper.scatter3d(x, y, z3, label_x="rank", label_y="m", label_z="acumulated time",
                         title=label_title + "_acumulated_time", show=True, legend=legend)


def scatter_3d_multiple_runs(runs, color_dicc, offset, label_title, zLimMin=0, zLimMax=0):
    lines = []
    lines2 = []
    lines3 = []
    labels = []
    for k in runs:
        x = []
        y = []
        z = []
        z2 = []
        z3 = []
        color = []
        labels.append(k)
        for r in runs[k]:
            if zLimMax == 0 or (zLimMax > float(r[1]) > zLimMin):
                x.append(int(k[4])+offset[k])  # rank
                y.append(int(k[5]))  # m
                z.append(float(r[1]))
                z2.append(float(r[2]))
                z3.append(float(r[3]))
                color.append(color_dicc[k])
        lines.append([x, y, z, color])
        lines2.append([x, y, z2, color])
        lines3.append([x, y, z3, color])

    PlotHelper.scatter3dWithLines(lines, label_x="rank", label_y="m", label_z="accuracy",
                                  title=label_title + "_accuracy", show=True, legends=labels)
    PlotHelper.scatter3dWithLines(lines2, label_x="rank", label_y="m", label_z="time", title=label_title + "_time"
                                  , show=True, legends=labels)
    PlotHelper.scatter3dWithLines(lines3, label_x="rank", label_y="m", label_z="acumulated_time",
                                  title=label_title + "_acumulated_time"
                                  , show=True, legends=labels)


DIMENSION_LEARNING_RATE = 3


def get_runs(run, list_of_labels, dimension=DIMENSION_LEARNING_RATE):
    runs = {}
    for label in list_of_labels:
        not_key_of_the_run = [k for k in run if k[dimension] != label]
        dicc_copy = run.copy()
        dissposable = [dicc_copy.pop(key) for key in not_key_of_the_run]
        runs[label] = dicc_copy
    return runs


''' Search best runs'''
NUMBER_OF_EPOCHS = 10 - 1
ACCURACY_DIM = 1


def search_best_run(runs):
    best_run = None
    best_run_accuracy = 0
    for run in runs:
        accuracy_run = runs[run][NUMBER_OF_EPOCHS][ACCURACY_DIM]
        if accuracy_run > best_run_accuracy:
            best_run_accuracy = runs[run][NUMBER_OF_EPOCHS][
                ACCURACY_DIM]
            best_run = run
    return runs[best_run]


def get_lines_and_times(runs, line_dimension, time_dimension):
    lines = []
    times = []
    for run in runs:
        line = [r[line_dimension] for r in run]
        time = [r[time_dimension] for r in run]
        lines.append(line)
        times.append(time)
    return lines, times


"""
def get_accuracy_number_of_parameters(runs, number_of_parameters_functions, line_dimension):
    lines = []
    n_of_params = []
    texts = []
    for r in range(len(runs)):
        run = runs[r]
        number_of_parameters_func = number_of_parameters_functions[r]
        line = []
        n_of_param = []
        text=[]
        for k in run:
            if type(k)==type([]):
                s=k
                text.append("")
            else:
                s = run[k][9]
                text.append("m:" + str(k[1]) + ",r:" + str(k[0]))
            line.append(s[line_dimension])
            n_of_param.append(number_of_parameters_func(Constant.N, k[1], k[0], Constant.AMOUNT_OF_DIVISIONS,
                                                        Constant.CATEGORIES))
            #print("Acc",s[line_dimension],"n",Constant.N,"m", k[1],"r", k[0],"d", Constant.AMOUNT_OF_DIVISIONS,"c",
            #      Constant.CATEGORIES,"params",
            #      number_of_parameters_func(Constant.N, k[1], k[0], Constant.AMOUNT_OF_DIVISIONS,Constant.CATEGORIES) )
        lines.append(line)
        n_of_params.append(n_of_param)
        texts.append(text)
    return lines, n_of_params,texts
"""


def append_to_lists(single_execution, run_key, text, line, n_of_param, number_of_parameters_func):
    s = float(single_execution[Constant.NET_ACCURACY_INDEX])
    text.append(run_key[0])
    line.append(s)
    n_of_param.append(number_of_parameters_func)


def get_accuracy_number_of_parameters(runs, only_last_run, only_best_run, group_by_net):
    return get_lines(runs, only_last_run, only_best_run, group_by_net, Constant.NET_PARAMETERS_INDEX,
                     Constant.NET_PARAMETERS_INDEX_RUN, get_number_of_param_func)


def get_time_number_of_parameters(runs, only_last_run, only_best_run, group_by_net):
    return get_lines(runs, only_last_run, only_best_run, group_by_net, Constant.NET_PARAMETERS_INDEX,
                     Constant.NET_PARAMETERS_INDEX_RUN, get_time_func)


def get_time_func(single_execution, run_key, parameter_index, group_by_net):
    if not group_by_net:
        number_of_parameters_func = float(single_execution[3])
    else:
        number_of_parameters_func = float(single_execution[3])
    return number_of_parameters_func


def get_number_of_param_func(single_execution, run_key, parameter_index, group_by_net):
    if not group_by_net:
        number_of_parameters_func = int(run_key[parameter_index])
    else:
        number_of_parameters_func = int(single_execution[4])
    return number_of_parameters_func


def get_lines(runs, only_last_run,only_best_run, group_by_net, index_in_key, index_in_run, get_param_func):
    lines = []
    x_lines = []
    texts = []
    legends = []
    sorted_keys = sorted(runs, key=lambda kv: int(kv[index_in_key]))
    parameter_index = index_in_key
    if group_by_net:
        new_runs = {}
        for run_key in runs:
            if only_best_run:
                runs[run_key].sort(key=lambda item: float(item[index_in_run]))
            if not run_key[Constant.NET_TYPE_INDEX] in new_runs:
                new_runs[(run_key[Constant.NET_TYPE_INDEX])] = [runs[run_key][-1] + [run_key[index_in_key]]]
            else:
                new_runs[(run_key[Constant.NET_TYPE_INDEX])].append(runs[run_key][-1] + [run_key[index_in_key]])
        runs = new_runs
        for run_key in runs:
            runs[run_key].sort(key=lambda l: int(l[4]))
        sorted_keys = runs
        only_last_run = False
        only_best_run = False
        parameter_index = 1

    for run_key in sorted_keys:
        run = runs[run_key]
        legends.append(str(run_key))
        line = []
        n_of_param = []
        text = []
        if only_last_run or only_best_run:
            if only_best_run:
                run.sort(key=lambda item: float(item[index_in_run]))
            single_execution = run[-1]
            number_of_parameters_func = get_param_func(single_execution, run_key, parameter_index,
                                                                 group_by_net)
            append_to_lists(single_execution, run_key, text, line, n_of_param, number_of_parameters_func)
        else:
            for single_execution in run:
                if not group_by_net:
                    number_of_parameters_func = get_param_func(single_execution, run_key, parameter_index,
                                                                         group_by_net)
                else:
                    number_of_parameters_func = get_param_func(single_execution, run_key, parameter_index,
                                                                         group_by_net)
                append_to_lists(single_execution, run_key, text, line, n_of_param, number_of_parameters_func)
        lines.append(line)
        x_lines.append(n_of_param)
        texts.append(text)
    return lines, x_lines, texts, legends

def order_if_necessary(run_dicc,only_last_run,only_best_run):
    runs={}
    if not only_best_run and not only_last_run:
        return  run_dicc
    for run_key in run_dicc:
        if only_best_run:
            run_dicc[run_key].sort(key=lambda item: float(item[Constant.NET_PARAMETERS_INDEX_RUN]))
        runs[run_key]=[run_dicc[run_key][-1]]
    return runs
