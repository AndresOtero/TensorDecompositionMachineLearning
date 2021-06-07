import numpy
from Utils import EnumModel, EnumDataset, EnumTrainMethods
from Utils.NetParams import NetParams
from t_test import *
from Plot.PlotUtils import *
import plotOutput as pltOutput


runsTR, parametersTR = GetRuns(TR_FASHION, x_axis_index=Constant.NET_RANK_INDEX)
runsTT, parametersTT = GetRuns(TT_FASHION, x_axis_index=Constant.NET_RANK_INDEX)
runsTR_m, parametersTR_m = GetRuns(TR_FASHION, x_axis_index=Constant.NET_RANK_INDEX)
runsTT_m, parametersTT_m = GetRuns(TT_FASHION_ALT, x_axis_index=Constant.NET_RANK_INDEX)
runsTR_m_t, parametersTR_m_t = GetRuns(TR_FASHION)
runsTT_m_t, parametersTT_m_t = GetRuns(TT_FASHION_ALT)
l=[]
for t in [[(runsTR_m,parametersTR_m),(runsTT_m,parametersTT_m)],[(runsTR_m_t,parametersTR_m_t),(runsTT_m_t,parametersTT_m_t)]]:
    for r,p in t:
        dicc={}
        for x in range(len(p)):
            if p[x] not in dicc:
                dicc[p[x]]=r[x]
            else:
                if dicc[p[x]]<r[x]:
                    dicc[p[x]] = r[x]
        l.append(dicc)
for x in l:
    print(x)



runsTT, parametersTT = GetRuns(TT_CONV_IMDB,dropout=0.1)
runsTR, parametersTR = GetRuns(TR_CONV_IMDB,dropout=0.2)
runsCNN, parametersCNN = GetRuns(CONV_IMDB)

PlotHelper.scatter2d_different_x_axis([ runsTT,runsTR,runsCNN], [ parametersTT,parametersTR,parametersCNN],
                                      legends=['Tensor Train Compartido','Tensor Ring Compartido','CNN IMDB'], xLabel="Cantidad de parámetros",
                                      yLabel="Precisión", title="IMDB_scatter_best_run")
PlotHelper.interpolate2d_different_x_axis([ runsTT,runsTR,runsCNN], [ parametersTT,parametersTR,parametersCNN],
                                          legends=['Tensor Ring Compartido', 'Tensor Train Compartido','CNN IMDB'], xLabel="Cantidad de parámetros",
                                          yLabel="Precisión", title="IMDB_interpolate_best_run")
PlotHelper.interpolate_and_scatter_different_x_axis([ runsTT,runsTR,runsCNN], [ parametersTT,parametersTR,parametersCNN],
                                                    legends=['Tensor Ring Compartido', 'Tensor Train Compartido','CNN IMDB'],
                                                    xLabel="Cantidad de parámetros",
                                                    yLabel="Precisión",
                                                    title="IMDB_interpolate_scatter_best_run")

runsTR, parametersTR = GetRuns(TR_FASHION)
runsTT, parametersTT = GetRuns(TT_FASHION)

PlotHelper.scatter2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                      legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                      yLabel="Precisión", title="Fashion_MNIST_scatter_best_run")
PlotHelper.interpolate2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                          legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                          yLabel="Precisión", title="Fashion_MNIST_interpolate_best_run")
PlotHelper.interpolate_and_scatter_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                                    legends=['Tensor Ring Compartido', 'Tensor Train Compartido'],
                                                    xLabel="Cantidad de parámetros",
                                                    yLabel="Precisión",
                                                    title="Fashion_MNIST_interpolate_scatter_best_run")
runsTR, parametersTR = GetRuns(TR_FASHION)
runsTT, parametersTT = GetRuns(TT_FASHION)

PlotHelper.scatter2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                      legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                      yLabel="Precisión", title="Fashion_MNIST_scatter_best_run_log",set_x_log=True)
PlotHelper.interpolate2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                          legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                          yLabel="Precisión", title="Fashion_MNIST_interpolate_best_run_log",set_x_log=True)
PlotHelper.interpolate_and_scatter_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                                    legends=['Tensor Ring Compartido', 'Tensor Train Compartido'],
                                                    xLabel="Cantidad de parámetros",
                                                    yLabel="Precisión",
                                                    title="Fashion_MNIST_interpolate_scatter_best_run_log",set_x_log=True)




runsTR, parametersTR = GetRuns(TR_MNIST)
runsTT, parametersTT = GetRuns(TT_MNIST)

PlotHelper.scatter2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                      legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                      yLabel="Precisión", title="MNIST_scatter_best_run")
PlotHelper.interpolate2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                          legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                          yLabel="Precisión", title="MNIST_interpolate_best_run")
PlotHelper.interpolate_and_scatter_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                                    legends=['Tensor Ring Compartido', 'Tensor Train Compartido'],
                                                    xLabel="Cantidad de parámetros",
                                                    yLabel="Precisión", title="MNIST_interpolate_scatter_best_run")

runsTR, parametersTR = GetRuns(TR_FASHION, best_n_runs=3)
runsTT, parametersTT = GetRuns(TT_FASHION, best_n_runs=3)

PlotHelper.interpolate2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                          legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                          yLabel="Precisión", title="Fashion_MNIST_interpolate_best_3_runs")

PlotHelper.interpolate_and_scatter_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                                    legends=['Tensor Ring Compartido', 'Tensor Train Compartido'],
                                                    xLabel="Cantidad de parámetros",
                                                    yLabel="Precisión",
                                                    title="Fashion_MNIST_interpolate_scatter_best_3_runs")

runsTR, parametersTR = GetRuns(TR_MNIST, best_n_runs=3)
runsTT, parametersTT = GetRuns(TT_MNIST, best_n_runs=3)

PlotHelper.interpolate2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                          legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                          yLabel="Precisión", title="MNIST_interpolate_best_3_runs")

PlotHelper.interpolate_and_scatter_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                                    legends=['Tensor Ring Compartido', 'Tensor Train Compartido'],
                                                    xLabel="Cantidad de parámetros",
                                                    yLabel="Precisión", title="MNIST_interpolate_scatter_best_3_runs")

runsTR, parametersTR = GetRuns(TR_MNIST, last_run=True)
runsTT, parametersTT = GetRuns(TT_MNIST, last_run=True)

PlotHelper.scatter2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                      legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                      yLabel="Precisión", title="MNIST_scatter_last_run")
runsTR, parametersTR = GetRuns(TR_FASHION, last_run=True)
runsTT, parametersTT = GetRuns(TT_FASHION, last_run=True)

PlotHelper.scatter2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                      legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                      yLabel="Precisión", title="Fashion_MNIST_scatter_last_run")

runsTR, parametersTR = GetRuns(TR_FASHION, best_n_runs=3, last_run=True)
runsTT, parametersTT = GetRuns(TT_FASHION, best_n_runs=3, last_run=True)

PlotHelper.interpolate2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                          legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                          yLabel="Precisión", title="Fashion MNIST_interpolate_last_3_runs")

PlotHelper.interpolate_and_scatter_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                                    legends=['Tensor Ring Compartido', 'Tensor Train Compartido'],
                                                    xLabel="Cantidad de parámetros",
                                                    yLabel="Precisión",
                                                    title="Fashion MNIST_interpolate_scatter_last_3_runs")

runsTR, parametersTR = GetRuns(TR_MNIST, best_n_runs=3, last_run=True)
runsTT, parametersTT = GetRuns(TT_MNIST, best_n_runs=3, last_run=True)

PlotHelper.interpolate2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                          legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                          yLabel="Precisión", title="MNIST_interpolate_last_3_runs")

PlotHelper.interpolate_and_scatter_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                                    legends=['Tensor Ring Compartido', 'Tensor Train Compartido'],
                                                    xLabel="Cantidad de parámetros",
                                                    yLabel="Precisión", title="MNIST_interpolate_scatter_last_3_runs")

runsTR, parametersTR = GetRuns(TR_MNIST, x_axis_index=Constant.NET_RANK_INDEX)
TT_MNIST_RUNS = TT_MNIST_ALT + TT_MNIST
TT_MNIST_RUNS.sort(key=lambda item: -float(item[4]))
runsTT, parametersTT = GetRuns(TT_MNIST_RUNS, x_axis_index=Constant.NET_RANK_INDEX)

PlotHelper.scatter2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                      legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Rango",
                                      yLabel="Precisión", title="MNIST_best_run_m=64_all_runs_vs_ranks")
PlotHelper.interpolate_and_scatter_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                                    legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Rango",
                                                    yLabel="Precisión",
                                                    title="MNIST_interpolate_scatter_best_run_m=64_all_runs_vs_ranks")

PlotHelper.interpolate2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                          legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Rango",
                                          yLabel="Precisión", title="MNIST_interpolate_best_run_m=64_all_runs_vs_ranks")

runsTR, parametersTR = GetRuns(TR_MNIST, x_axis_index=Constant.NET_RANK_INDEX)
runsTT, parametersTT = GetRuns(TT_MNIST_ALT, x_axis_index=Constant.NET_RANK_INDEX)

PlotHelper.scatter2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                      legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Rango",
                                      yLabel="Precisión", title="MNIST_best_run_m=64_vs_ranks")
PlotHelper.interpolate_and_scatter_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                                    legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Rango",
                                                    yLabel="Precisión",
                                                    title="MNIST_interpolate_scatter_best_run_m=64_vs_ranks")

PlotHelper.interpolate2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                          legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Rango",
                                          yLabel="Precisión", title="MNIST_interpolate_best_run_m=64_vs_ranks")

runsTR, parametersTR = GetRuns(TR_MNIST)
runsTT, parametersTT = GetRuns(TT_MNIST_ALT)

PlotHelper.scatter2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                      legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                      yLabel="Precisión", title="MNIST_best_run_m=64")
PlotHelper.interpolate_and_scatter_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                                    legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                                    yLabel="Precisión",
                                                    title="MNIST_interpolate_scatter_best_run_m=64")

PlotHelper.interpolate2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                          legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                          yLabel="Precisión", title="MNIST_interpolate_best_run_m=64")
runsTR, parametersTR = GetRuns(TR_MNIST_32)
runsTT, parametersTT = GetRuns(TT_MNIST_32)
PlotHelper.scatter2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                      legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                      yLabel="Precisión", title="MNIST_best_run_m=32")
PlotHelper.interpolate_and_scatter_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                                    legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Rango",
                                                    yLabel="Precisión",
                                                    title="MNIST_interpolate_scatter_best_run_m=32")

PlotHelper.interpolate2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                          legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                          yLabel="Precisión", title="MNIST_interpolate_best_run_m=32")


runsTR, parametersTR = GetRuns(TR_MNIST_32, x_axis_index=Constant.NET_RANK_INDEX)
runsTT, parametersTT = GetRuns(TT_MNIST_32, x_axis_index=Constant.NET_RANK_INDEX)
PlotHelper.scatter2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                      legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Rango",
                                      yLabel="Precisión", title="MNIST_best_run_m=32_vs_ranks")
PlotHelper.interpolate_and_scatter_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                                    legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                                    yLabel="Precisión",
                                                    title="MNIST_interpolate_scatter_best_run_m=32_vs_ranks")

PlotHelper.interpolate2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                          legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Rango",
                                          yLabel="Precisión", title="MNIST_interpolate_best_run_m=32_vs_ranks")
runsTR, parametersTR = GetRuns(TR_MNIST_32)
runsTT, parametersTT = GetRuns(TT_MNIST_32)
PlotHelper.scatter2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                      legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                      yLabel="Precisión", title="MNIST_best_run_m=32")
PlotHelper.interpolate_and_scatter_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                                    legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                                    yLabel="Precisión",
                                                    title="MNIST_interpolate_scatter_best_run_m=32")

PlotHelper.interpolate2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                          legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                          yLabel="Precisión", title="MNIST_interpolate_scatter_best_run_m=32")




runsTR, parametersTR = GetRuns(TR_FASHION, x_axis_index=Constant.NET_RANK_INDEX)
TT_FASHION_RUNS = TT_FASHION_ALT + TT_FASHION
TT_FASHION_RUNS.sort(key=lambda item: -float(item[4]))
runsTT, parametersTT = GetRuns(TT_FASHION_RUNS, x_axis_index=Constant.NET_RANK_INDEX)

PlotHelper.scatter2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                      legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Rango",
                                      yLabel="Precisión", title="Fashion_best_run_m=64_all_runs_vs_ranks")
PlotHelper.interpolate_and_scatter_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                                    legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Rango",
                                                    yLabel="Precisión",
                                                    title="Fashion_interpolate_scatter_best_run_m=64_all_runs_vs_ranks")

PlotHelper.interpolate2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                          legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Rango",
                                          yLabel="Precisión",
                                          title="Fashion_interpolate_best_run_m=64_all_runs_vs_ranks")
runsTR, parametersTR = GetRuns(TR_FASHION, x_axis_index=Constant.NET_RANK_INDEX)
runsTT, parametersTT = GetRuns(TT_FASHION_ALT, x_axis_index=Constant.NET_RANK_INDEX)

PlotHelper.scatter2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                      legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Rango",
                                      yLabel="Precisión", title="Fashion_best_run_m=64_vs_ranks")
PlotHelper.interpolate_and_scatter_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                                    legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Rango",
                                                    yLabel="Precisión",
                                                    title="Fashion_interpolate_scatter_best_run_m=64_vs_ranks")

PlotHelper.interpolate2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                          legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Rango",
                                          yLabel="Precisión", title="Fashion_interpolate_best_run_m=64_vs_ranks")

runsTR, parametersTR = GetRuns(TR_FASHION)
runsTT, parametersTT = GetRuns(TT_FASHION)

PlotHelper.scatter2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                      legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                      yLabel="Precisión", title="Fashion_best_run_m=64_vs_ranks")
PlotHelper.interpolate_and_scatter_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                                    legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                                    yLabel="Precisión",
                                                    title="Fashion_interpolate_scatter_best_run_m=64")

PlotHelper.interpolate2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                          legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                          yLabel="Precisión", title="Fashion_interpolate_best_run_m=64")

runsTR, parametersTR = GetRuns(TR_FASHION_32)
runsTT, parametersTT = GetRuns(TT_FASHION_32)
PlotHelper.scatter2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                      legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                      yLabel="Precisión", title="Fashion_best_run_m=32")
PlotHelper.interpolate_and_scatter_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                                    legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                                    yLabel="Precisión",
                                                    title="Fashion_interpolate_scatter_best_run_m=32")

PlotHelper.interpolate2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                          legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Cantidad de parámetros",
                                          yLabel="Precisión", title="Fashion_interpolate_best_run_m=32")
runsTR, parametersTR = GetRuns(TR_FASHION_32, x_axis_index=Constant.NET_RANK_INDEX)
runsTT, parametersTT = GetRuns(TT_FASHION_32, x_axis_index=Constant.NET_RANK_INDEX)
PlotHelper.scatter2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                      legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Rango",
                                      yLabel="Precisión", title="Fashion_best_run_m=32_vs_ranks")
PlotHelper.interpolate_and_scatter_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                                    legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Rango",
                                                    yLabel="Precisión",
                                                    title="Fashion_interpolate_scatter_best_run_m=32_vs_ranks")

PlotHelper.interpolate2d_different_x_axis([runsTR, runsTT], [parametersTR, parametersTT],
                                          legends=['Tensor Ring Compartido', 'Tensor Train Compartido'], xLabel="Rango",
                                          yLabel="Precisión", title="Fashion_interpolate_best_run_m=32_vs_ranks")
z, x, y = GetRuns3D(TR_MNIST)
z_2, x_2, y_2 = GetRuns3D(TT_MNIST)
z_3, x_3, y_3 = GetRuns3D(TT_MNIST_32)
z_4, x_4, y_4 = GetRuns3D(TR_MNIST_32)
color = ['b', 'r', 'g', 'c']
legends = ['TR m=64', 'TT m=64', 'TT m=32', 'TR m=32']
PlotHelper.interpolate3d([x, x_2, x_3, x_4], [y, y_2, y_3, y_4], [z, z_2, z_3, z_4], color=color, legend=legends,
                         title='MNIST_Interpolate_3D', label_x="Parameters", label_y="Rank", label_z="accuracy")
PlotHelper.interpolate_scatter3d([x, x_2, x_3, x_4], [y, y_2, y_3, y_4], [z, z_2, z_3, z_4], color=color,
                                 legend=legends, title='MNIST_Interpolate_Scatter_3D',
                                 label_x="Parameters", label_y="Rank", label_z="accuracy")
PlotHelper.scatter3d([x, x_2, x_3, x_4], [y, y_2, y_3, y_4], [z, z_2, z_3, z_4], color=color, legend=legends,
                     title='MNIST_Scatter_3D', label_x="Parameters", label_y="Rank", label_z="accuracy")
runsTR, parametersTR = GetRuns(TR_FASHION, x_axis_index=Constant.NET_RANK_INDEX, best_n_runs=3)
runsTT, parametersTT = GetRuns(TT_FASHION, x_axis_index=Constant.NET_RANK_INDEX, best_n_runs=3)
runsTT_alt, parametersTT_alt = GetRuns(TT_FASHION_ALT, x_axis_index=Constant.NET_RANK_INDEX, best_n_runs=3)

runsTT = np.concatenate((runsTT, runsTT_alt))
parametersTT = np.concatenate((parametersTT, parametersTT_alt))
aux_runsTT = [(runsTT[x], parametersTT[x]) for x in range(len(runsTT))]
aux_runsTT.sort(key=lambda x: x[1])
runsTT, parametersTT = np.array([x[0] for x in aux_runsTT]), np.array([x[1] for x in aux_runsTT])
runsTT_32, parametersTT_32 = GetRuns(TT_FASHION_32, x_axis_index=Constant.NET_RANK_INDEX, best_n_runs=3)
runsTR_32, parametersTR_32 = GetRuns(TR_FASHION_32, x_axis_index=Constant.NET_RANK_INDEX, best_n_runs=3)

PlotHelper.scatter2d_different_x_axis([runsTR, runsTT, runsTT_32, runsTR_32],
                                      [parametersTR, parametersTT, parametersTT_32, parametersTR_32],
                                      legends=['Tensor Ring m=64', 'Tensor Train m=64', 'Tensor Train m=32',
                                               'Tensor Ring m=32'], xLabel="Rango",
                                      yLabel="Precisión", title="Fashion_MNIST_scatter_m_3_runs_vs_ranks")

PlotHelper.interpolate_and_scatter_different_x_axis([runsTR, runsTT, runsTT_32, runsTR_32],
                                                    [parametersTR, parametersTT, parametersTT_32, parametersTR_32],
                                                    legends=['Tensor Ring m=64', 'Tensor Train m=64',
                                                             'Tensor Train m=32', 'Tensor Ring m=32'], xLabel="Rango",
                                                    yLabel="Precisión",
                                                    title="Fashion_MNIST_scatter_interpolate_m_3_runs_vs_ranks")

PlotHelper.interpolate2d_different_x_axis([runsTR, runsTT, runsTT_32, runsTR_32],
                                          [parametersTR, parametersTT, parametersTT_32, parametersTR_32],
                                          legends=['Tensor Ring m=64', 'Tensor Train m=64', 'Tensor Train m=32',
                                                   'Tensor Ring m=32'], xLabel="Rango",
                                          yLabel="Precisión", title="Fashion_MNIST_interpolate_m_3_runs_vs_ranks")

runsTR, parametersTR = GetRuns(TR_FASHION)
runsTT, parametersTT = GetRuns(TT_FASHION)
runsTT_alt, parametersTT_alt = GetRuns(TT_FASHION_ALT)

runsTT = np.concatenate((runsTT, runsTT_alt))
parametersTT = np.concatenate((parametersTT, parametersTT_alt))
aux_runsTT = [(runsTT[x], parametersTT[x]) for x in range(len(runsTT))]
aux_runsTT.sort(key=lambda x: x[1])
runsTT, parametersTT = np.array([x[0] for x in aux_runsTT]), np.array([x[1] for x in aux_runsTT])
runsTT_32, parametersTT_32 = GetRuns(TT_FASHION_32)
runsTR_32, parametersTR_32 = GetRuns(TR_FASHION_32)

PlotHelper.scatter2d_different_x_axis([runsTR, runsTT, runsTT_32, runsTR_32],
                                      [parametersTR, parametersTT, parametersTT_32, parametersTR_32],
                                      legends=['Tensor Ring m=64', 'Tensor Train m=64', 'Tensor Train m=32',
                                               'Tensor Ring m=32'], xLabel="Cantidad de parámetros",
                                      yLabel="Precisión", title="Fashion_MNIST_scatter_m")

PlotHelper.interpolate_and_scatter_different_x_axis([runsTR, runsTT, runsTT_32, runsTR_32], [parametersTR, parametersTT,
                                                                                             parametersTT_32,
                                                                                             parametersTR_32],
                                                    legends=['Tensor Ring m=64', 'Tensor Train m=64',
                                                             'Tensor Ring m=32', 'Tensor Train m = 32'],
                                                    xLabel="Cantidad de parámetros", yLabel="Precisión",
                                                    title="Fashion_MNIST_interpolate_scatter_m")

PlotHelper.interpolate2d_different_x_axis([runsTR, runsTT, runsTT_32, runsTR_32], [parametersTR, parametersTT,
                                                                                   parametersTT_32, parametersTR_32],
                                          legends=['Tensor Ring m=64', 'Tensor Train m=64', 'Tensor Train m=32',
                                                   'Tensor Ring m=32'], xLabel="Cantidad de parámetros", yLabel="Precisión",
                                          title="Fashion_MNIST_interpolate_m")

runsTR, parametersTR = GetRuns(TR_MNIST, x_axis_index=Constant.NET_RANK_INDEX)
runsTT, parametersTT = GetRuns(TT_MNIST, x_axis_index=Constant.NET_RANK_INDEX)
runsTT_alt, parametersTT_alt = GetRuns(TT_MNIST_ALT, x_axis_index=Constant.NET_RANK_INDEX)

runsTT = np.concatenate((runsTT, runsTT_alt))
parametersTT = np.concatenate((parametersTT, parametersTT_alt))
aux_runsTT = [(runsTT[x], parametersTT[x]) for x in range(len(runsTT))]
aux_runsTT.sort(key=lambda x: x[1])
runsTT, parametersTT = np.array([x[0] for x in aux_runsTT]), np.array([x[1] for x in aux_runsTT])
runsTT_32, parametersTT_32 = GetRuns(TT_MNIST_32, x_axis_index=Constant.NET_RANK_INDEX)
runsTR_32, parametersTR_32 = GetRuns(TR_MNIST_32, x_axis_index=Constant.NET_RANK_INDEX)

PlotHelper.scatter2d_different_x_axis([runsTR, runsTT, runsTT_32, runsTR_32],
                                      [parametersTR, parametersTT, parametersTT_32, parametersTR_32],
                                      legends=['Tensor Ring m=64', 'Tensor Train m=64', 'Tensor Train m=32',
                                               'Tensor Ring m=32'], xLabel="Rango",
                                      yLabel="Precisión", title="MNIST_scatter_m_vs_ranks")

PlotHelper.interpolate_and_scatter_different_x_axis([runsTR, runsTT, runsTT_32, runsTR_32],
                                                    [parametersTR, parametersTT, parametersTT_32, parametersTR_32],
                                                    legends=['Tensor Ring m=64', 'Tensor Train m=64',
                                                             'Tensor Train m=32', 'Tensor Ring m=32'], xLabel="Rango",
                                                    yLabel="Precisión", title="MNIST_interpolate_scatter_m_vs_ranks")

PlotHelper.interpolate2d_different_x_axis([runsTR, runsTT, runsTT_32, runsTR_32],
                                          [parametersTR, parametersTT, parametersTT_32, parametersTR_32],
                                          legends=['Tensor Ring m=64', 'Tensor Train m=64', 'Tensor Train m=32',
                                                   'Tensor Ring m=32'], xLabel="Rango",
                                          yLabel="Precisión", title="MNIST_interpolate_m_vs_ranks")


runsTR, parametersTR = GetRuns(TR_MNIST)
runsTT, parametersTT = GetRuns(TT_MNIST)
runsTT_alt, parametersTT_alt = GetRuns(TT_MNIST_ALT)

runsTT = np.concatenate((runsTT, runsTT_alt))
parametersTT = np.concatenate((parametersTT, parametersTT_alt))
aux_runsTT = [(runsTT[x], parametersTT[x]) for x in range(len(runsTT))]
aux_runsTT.sort(key=lambda x: x[1])
runsTT, parametersTT = np.array([x[0] for x in aux_runsTT]), np.array([x[1] for x in aux_runsTT])
runsTT_32, parametersTT_32 = GetRuns(TT_MNIST_32)
runsTR_32, parametersTR_32 = GetRuns(TR_MNIST_32)

PlotHelper.scatter2d_different_x_axis([runsTR, runsTT, runsTT_32, runsTR_32],
                                      [parametersTR, parametersTT, parametersTT_32, parametersTR_32],
                                      legends=['Tensor Ring m=64', 'Tensor Train m=64', 'Tensor Train m=32',
                                               'Tensor Ring m=32'], xLabel="Cantidad de parámetros",
                                      yLabel="Precisión", title="MNIST_scatter_m")

PlotHelper.interpolate_and_scatter_different_x_axis([runsTR, runsTT, runsTT_32, runsTR_32],
                                                    [parametersTR, parametersTT, parametersTT_32, parametersTR_32],
                                                    legends=['Tensor Ring m=64', 'Tensor Train m=64',
                                                             'Tensor Train m=32', 'Tensor Ring m=32'], xLabel="Cantidad de parámetros",
                                                    yLabel="Precisión", title="MNIST_interpolate_scatter_m")

PlotHelper.interpolate2d_different_x_axis([runsTR, runsTT, runsTT_32, runsTR_32],
                                          [parametersTR, parametersTT, parametersTT_32, parametersTR_32],
                                          legends=['Tensor Ring m=64', 'Tensor Train m=64', 'Tensor Train m=32',
                                                   'Tensor Ring m=32'], xLabel="Cantidad de parámetros",
                                          yLabel="Precisión", title="MNIST_interpolate_m")


TR_PARAMETERS = TR_MNIST
TT_PARAMETERS = TT_MNIST + TT_MNIST_ALT
y1, x1 = np.array([x[Constant.NET_RANK_INDEX] for x in TR_PARAMETERS]).astype(float), np.array(
    [x[Constant.NET_PARAMETERS_INDEX] for x in TR_PARAMETERS]).astype(float)
y2, x2 = np.array([x[Constant.NET_RANK_INDEX] for x in TT_PARAMETERS]).astype(float), np.array(
    [x[Constant.NET_PARAMETERS_INDEX] for x in TT_PARAMETERS]).astype(float)
legends = ['TR', 'TT']
PlotHelper.interpolate_and_scatter_different_x_axis([x1, x2], [y1, y2], yLabel='Cantidad de parámetros', xLabel='Rango',
                                                    legends=legends, title="Scatter_Number_Of_Parameters_m=64_log",set_y_log=True)

TR_PARAMETERS = TR_MNIST_32
TT_PARAMETERS = TT_MNIST_32
y1, x1 = np.array([x[Constant.NET_RANK_INDEX] for x in TR_PARAMETERS]).astype(float), np.array(
    [x[Constant.NET_PARAMETERS_INDEX] for x in TR_PARAMETERS]).astype(float)
y2, x2 = np.array([x[Constant.NET_RANK_INDEX] for x in TT_PARAMETERS]).astype(float), np.array(
    [x[Constant.NET_PARAMETERS_INDEX] for x in TT_PARAMETERS]).astype(float)
legends = ['TR', 'TT']
PlotHelper.interpolate_and_scatter_different_x_axis([x1, x2], [y1, y2], yLabel='Cantidad de parámetros', xLabel='Rango',
                                                    legends=legends, title="Scatter_Number_Of_Parameters_m=32_log",set_y_log=True)

TR_PARAMETERS = TR_MNIST
TT_PARAMETERS = TT_MNIST + TT_MNIST_ALT
y1, x1 = np.array([x[Constant.NET_RANK_INDEX] for x in TR_PARAMETERS]).astype(float), np.array(
    [x[Constant.NET_PARAMETERS_INDEX] for x in TR_PARAMETERS]).astype(float)
y2, x2 = np.array([x[Constant.NET_RANK_INDEX] for x in TT_PARAMETERS]).astype(float), np.array(
    [x[Constant.NET_PARAMETERS_INDEX] for x in TT_PARAMETERS]).astype(float)
legends = ['TR', 'TT']
PlotHelper.interpolate_and_scatter_different_x_axis([x1, x2], [y1, y2], yLabel='Cantidad de parámetros', xLabel='Rango',
                                                    legends=legends, title="Scatter_Number_Of_Parameters_m=64")

TR_PARAMETERS = TR_MNIST_32
TT_PARAMETERS = TT_MNIST_32
y1, x1 = np.array([x[Constant.NET_RANK_INDEX] for x in TR_PARAMETERS]).astype(float), np.array(
    [x[Constant.NET_PARAMETERS_INDEX] for x in TR_PARAMETERS]).astype(float)
y2, x2 = np.array([x[Constant.NET_RANK_INDEX] for x in TT_PARAMETERS]).astype(float), np.array(
    [x[Constant.NET_PARAMETERS_INDEX] for x in TT_PARAMETERS]).astype(float)
legends = ['TR', 'TT']
PlotHelper.interpolate_and_scatter_different_x_axis([x1, x2], [y1, y2], yLabel='Cantidad de parámetros', xLabel='Rango',
                                                    legends=legends, title="Scatter_Number_Of_Parameters_m=32")

TR_PARAMETERS_32 = TR_MNIST_32
TT_PARAMETERS_32 = TT_MNIST_32
TR_PARAMETERS_64 = TR_MNIST
TT_PARAMETERS_64 = TT_MNIST + TT_MNIST_ALT
y1, x1 = np.array([x[Constant.NET_RANK_INDEX] for x in TR_PARAMETERS_32]).astype(float), np.array(
    [x[Constant.NET_PARAMETERS_INDEX] for x in TR_PARAMETERS_32]).astype(float)
y2, x2 = np.array([x[Constant.NET_RANK_INDEX] for x in TT_PARAMETERS_32]).astype(float), np.array(
    [x[Constant.NET_PARAMETERS_INDEX] for x in TT_PARAMETERS_32]).astype(float)
y3, x3 = np.array([x[Constant.NET_RANK_INDEX] for x in TR_PARAMETERS_64]).astype(float), np.array(
    [x[Constant.NET_PARAMETERS_INDEX] for x in TR_PARAMETERS_64]).astype(float)
y4, x4 = np.array([x[Constant.NET_RANK_INDEX] for x in TT_PARAMETERS_64]).astype(float), np.array(
    [x[Constant.NET_PARAMETERS_INDEX] for x in TT_PARAMETERS_64]).astype(float)
legends = ['Tensor Ring Compartido M=32', 'Tensor Train Compartido M=32','Tensor Ring Compartido M=64'
    ,'Tensor Train Compartido M=64']
PlotHelper.interpolate_and_scatter_different_x_axis([x1, x2, x3, x4], [y1, y2, y3, y4], yLabel='Cantidad de parámetros', xLabel='Rango',
                                                    legends=legends, title="Scatter_Number_Of_Parameters_MNIST",grid=True)
runsLSTM, parametersLSTM = GetRuns(LSTM_Fashion)
runsLSTM32, parametersLSTM32 = GetRuns(LSTM_Fashion_32)
runsRNN, parametersRNN = GetRuns(RNN_FASHION)
runsRNN32, parametersRNN32 = GetRuns(RNN_FASHION_32)
runsTR, parametersTR = GetRuns(TR_FASHION)
runsTT, parametersTT = GetRuns(TT_FASHION)
PlotHelper.scatter2d_different_x_axis([runsLSTM, runsLSTM32, runsRNN, runsRNN32, runsTT, runsTR],
                                      [parametersLSTM, parametersLSTM32, parametersRNN, parametersRNN32, parametersTT,
                                       parametersTR], legends=['LSTM', 'LSTM_32', 'RNN', 'RNN 32', 'TT', 'TR'],
                                      xLabel="Cantidad de parámetros",
                                      yLabel="Precisión", title="ALL_Fashion_MNIST_scatter_best_run")

PlotHelper.interpolate_and_scatter_different_x_axis([runsLSTM, runsLSTM32, runsRNN, runsRNN32, runsTT, runsTR],
                                                    [parametersLSTM, parametersLSTM32, parametersRNN, parametersRNN32,
                                                     parametersTT, parametersTR],
                                                    legends=['LSTM', 'LSTM_32', 'RNN', 'RNN 32', 'TT', 'TR'],
                                                    xLabel="Cantidad de parámetros",
                                                    yLabel="Precisión",
                                                    title="ALL_Fashion_MNIST_interpolate_scatter_best_run")

PlotHelper.interpolate2d_different_x_axis([runsLSTM, runsLSTM32, runsRNN, runsRNN32, runsTT, runsTR],
                                          [parametersLSTM, parametersLSTM32, parametersRNN, parametersRNN32,
                                           parametersTT, parametersTR],
                                          legends=['LSTM M=64', 'LSTM M=32', 'RNN M=64', 'RNN M=32', 'TT Compartido',
                                                   'TR Compartido'],
                                          xLabel="Cantidad de parámetros",
                                          yLabel="Precisión", title="ALL_Fashion_MNIST_interpolate_best_run")

runsLSTM, parametersLSTM = GetRuns(LSTM_MNIST)
runsLSTM32, parametersLSTM32 = GetRuns(LSTM_MNIST_32)
runsRNN, parametersRNN = GetRuns(RNN_MNIST)
runsRNN32, parametersRNN32 = GetRuns(RNN_MNIST_32)
runsTR, parametersTR = GetRuns(TR_MNIST)
runsTT, parametersTT = GetRuns(TT_MNIST)
PlotHelper.scatter2d_different_x_axis([runsLSTM, runsLSTM32, runsRNN, runsRNN32, runsTT, runsTR],
                                      [parametersLSTM, parametersLSTM32, parametersRNN, parametersRNN32, parametersTT,
                                       parametersTR], legends=['LSTM', 'LSTM_32', 'RNN', 'RNN 32', 'TT', 'TR'],
                                      xLabel="Cantidad de parámetros",
                                      yLabel="Precisión", title="ALL_MNIST_scatter_best_run")

PlotHelper.interpolate_and_scatter_different_x_axis([runsLSTM, runsLSTM32, runsRNN, runsRNN32, runsTT, runsTR],
                                                    [parametersLSTM, parametersLSTM32, parametersRNN, parametersRNN32,
                                                     parametersTT, parametersTR],
                                                    legends=['LSTM', 'LSTM_32', 'RNN', 'RNN 32', 'TT', 'TR'],
                                                    xLabel="Cantidad de parámetros",
                                                    yLabel="Precisión", title="ALL_MNIST_interpolate_scatter_best_run")

PlotHelper.interpolate2d_different_x_axis([runsLSTM, runsLSTM32, runsRNN, runsRNN32, runsTT, runsTR],
                                          [parametersLSTM, parametersLSTM32, parametersRNN, parametersRNN32,
                                           parametersTT, parametersTR],
                                          legends=['LSTM M=64', 'LSTM M=32', 'RNN M=64', 'RNN M=32', 'TT Compartido',
                                                   'TR Compartido'],
                                          xLabel="Cantidad de parámetros",
                                          yLabel="Precisión", title="ALL_MNIST_interpolate_best_run")

runsCONVTR, parametersCONVTR = GetRuns(CONV_TR_CIFAR)
runsCONVTT, parametersCONVTT = GetRuns(CONV_TT_CIFAR)
runsNET, parametersNET = GetRuns(NET_CIFAR)

PlotHelper.scatter2d_different_x_axis([runsCONVTR, runsCONVTT, runsNET],
                                      [parametersCONVTR, parametersCONVTT, parametersNET],
                                      legends=['CONV_TR', 'CONV_TT', 'NET'],
                                      xLabel="Cantidad de parámetros",
                                      yLabel="Precisión", title="CIFAR_scatter_best_run")

PlotHelper.interpolate_and_scatter_different_x_axis([runsCONVTR, runsCONVTT, runsNET],
                                      [parametersCONVTR, parametersCONVTT, parametersNET],
                                      legends=['CONV_TR', 'CONV_TT', 'NET'],
                                      xLabel="Cantidad de parámetros",
                                                    yLabel="Precisión",
                                                    title="CIFAR_interpolate_scatter_best_run")

PlotHelper.interpolate2d_different_x_axis([runsCONVTR, runsCONVTT, runsNET],
                                      [parametersCONVTR, parametersCONVTT, parametersNET],
                                      legends=['CONV_TR', 'CONV_TT', 'NET'],
                                      xLabel="Cantidad de parámetros",
                                          yLabel="Precisión", title="CIFAR_interpolate_scatter_best_run")

runsTR, parametersTR = GetRuns(CONV_TR_CIFAR, best_n_runs=3)
runsTT, parametersTT = GetRuns(CONV_TT_CIFAR, best_n_runs=3)
runsNET, parametersNET = GetRuns(NET_CIFAR, best_n_runs=3)

PlotHelper.scatter2d_different_x_axis([runsCONVTR, runsCONVTT, runsNET],
                                      [parametersCONVTR, parametersCONVTT, parametersNET],
                                      legends=['CONV_TR', 'CONV_TT', 'NET'],
                                      xLabel="Cantidad de parámetros",
                                      yLabel="Precisión", title="CIFAR_scatter_best_3_runs")

PlotHelper.interpolate_and_scatter_different_x_axis([runsCONVTR, runsCONVTT, runsNET],
                                      [parametersCONVTR, parametersCONVTT, parametersNET],
                                      legends=['CONV_TR', 'CONV_TT', 'NET'],
                                      xLabel="Cantidad de parámetros",
                                                    yLabel="Precisión",
                                                    title="CIFAR_interpolate_scatter_best_3_runs")

PlotHelper.interpolate2d_different_x_axis([runsCONVTR, runsCONVTT, runsNET],
                                      [parametersCONVTR, parametersCONVTT, parametersNET],
                                      legends=['CONV_TR', 'CONV_TT', 'NET'],
                                      xLabel="Cantidad de parámetros",
                                          yLabel="Precisión", title="CIFAR_interpolate_scatter_3_best_runs")
