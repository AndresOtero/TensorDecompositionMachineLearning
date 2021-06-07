import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp1d, CubicSpline, interpolate, PchipInterpolator, interp2d
from torch import linspace
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting
import numpy as np


class PlotHelper(object):
    @staticmethod
    def scatter2d_different_x_axis(list_of_lines, t, legends=[], xLimMin=0, xLimMax=0, yLimMin=0, yLimMax=0, xLabel="x",
                                yLabel="y",
                                title="", invert_x=False, show=False, texts=[],set_y_log=False,set_x_log=False):
        fig = plt.figure()
        ax = fig.gca()
        colors = cm.rainbow(np.linspace(0, 1, len(list_of_lines)))
        for line in range(len(list_of_lines)):
            color=colors[line]
            for r in range(len(t[line])):
                ax.scatter(t[line][r], list_of_lines[line][r],color=color)
                if texts:
                    ax.text(t[line][r]+1, list_of_lines[line][r]+0.05,texts[line][r], fontsize=7)
        ax.legend(legends)
        colors = iter(colors)
        for leg in ax.get_legend().legendHandles:
            leg.set_color(next(colors))
        if (xLimMax != 0):
            ax.set_xlim(xLimMin, xLimMax)
        if (yLimMax != 0):
            ax.set_ylim(yLimMin, yLimMax)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)

        ax.set_title(title)
        if invert_x:
            ax.invert_xaxis()
        if set_x_log:
            ax.set_xscale('log')
        if set_y_log:
            ax.set_yscale('log')
        plt.savefig("./img/" + title + ".png")
        plt.grid(True)
        if (show):
            plt.show()
        plt.close()
             
        
    @staticmethod
    def plot2d(list_of_lines, t, legends=[], xLimMin=0, xLimMax=0, yLimMin=0, yLimMax=0, xLabel="x", yLabel="y",
               title="", invert_x=False, show=False, texts=[]):
        fig = plt.figure()
        ax = fig.gca()
        for line in range(len(list_of_lines)):
            ax.plot(t[line], list_of_lines[line])
        ax.legend(legends, loc=1)
        if (xLimMax != 0):
            ax.set_xlim(xLimMin, xLimMax)
        if yLimMax != 0:
            ax.set_ylim(yLimMin, yLimMax)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)

        ax.set_title(title)
        if invert_x:
            ax.invert_xaxis()
        plt.savefig("./img/" + title + ".png")
        plt.grid(True)
        if (show):
            plt.show()
        plt.close()
        
        
    @staticmethod
    def plot2d_different_x_axis(list_of_lines, t, legends=[], xLimMin=0, xLimMax=0, yLimMin=0, yLimMax=0, xLabel="x",
                                yLabel="y",
                                title="", invert_x=False, show=False):
        fig = plt.figure()
        ax = fig.gca()
        for line in range(len(list_of_lines)):
            ax.plot(t[line], list_of_lines[line])
        ax.legend(legends, loc=0)
        if (xLimMax != 0):
            ax.set_xlim(xLimMin, xLimMax)
        if (yLimMax != 0):
            ax.set_ylim(yLimMin, yLimMax)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)

        ax.set_title(title)
        if invert_x:
            ax.invert_xaxis()
        plt.savefig("./img/" + title + ".png")
        plt.grid(True)
        if (show):
            plt.show()
        plt.close()
        
        
    @staticmethod
    def interpolate2d_different_x_axis(list_of_lines, t, legends=[], xLimMin=0, xLimMax=0, yLimMin=0, yLimMax=0, xLabel="x",
                                yLabel="y",
                                title="", invert_x=False, show=False,set_x_log=False,set_y_log=False):
        fig = plt.figure()
        ax = fig.gca()
        for line in range(len(list_of_lines)):
            x=t[line].astype(int)
            y=list_of_lines[line]
            new_x,new_y= PlotHelper.GetAverageOfUniqueValues(x, y)
            try:
                ax.plot(new_x, interp1d(new_x, new_y, kind='cubic')(new_x))
            except:
                if(len(new_x)==1):
                    ax.plot(new_x, new_y)
                else:
                    ax.plot(new_x, interp1d(new_x, new_y, kind='quadratic')(new_x))
        ax.legend(legends, loc=0)
        if (xLimMax != 0):
            ax.set_xlim(xLimMin, xLimMax)
        if (yLimMax != 0):
            ax.set_ylim(yLimMin, yLimMax)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)

        #ax.set_title(title)
        if invert_x:
            ax.invert_xaxis()
        if set_x_log:
            ax.set_xscale('log')
        if set_y_log:
            ax.set_yscale('log')
        plt.savefig("./img/" + title + ".png")
        plt.grid(True)
        if (show):
            plt.show()
        plt.close()
        
        
    @staticmethod
    def GetAverageOfUniqueValues(x,y):
        dicc={}
        for i in range(len(x)):
            x_value=x[i]
            y_value=y[i]
            if x_value not in dicc:
                dicc[x_value]= [y_value]
            else:
                dicc[x_value].append(y_value)
        new_x,new_y=[],[]
        for x in sorted(dicc.keys()):
            new_x.append(x)
            new_y.append(sum(dicc[x]) / len(dicc[x]))
        return new_x,new_y

    @staticmethod
    def GetAverageOfUniqueValues3D(x,y,z):
        dicc={}
        for i in range(len(x)):
            x_value=x[i]
            y_value=y[i]
            z_value=z[i]
            if (x_value,y_value) not in dicc:
                dicc[(x_value,y_value)]= [z_value]
            else:
                dicc[(x_value,y_value)].append(z_value)
        new_x,new_y,new_z=[],[],[]
        for x,y in sorted(dicc.keys()):
            new_x.append(x)
            new_y.append(y)
            new_z.append(sum(dicc[(x,y)]) / len(dicc[(x,y)]))
        return new_x,new_y,new_z

    @staticmethod
    def interpolate_and_scatter_different_x_axis(list_of_lines, t, legends=[], xLimMin=0, xLimMax=0, yLimMin=0, yLimMax=0,
                                       xLabel="x",
                                       yLabel="y",
                                       title="", invert_x=False, show=False, texts=[],set_y_log=False,set_x_log=False,
                                                 grid=False):
        fig = plt.figure()
        ax = fig.gca()
        colors = cm.rainbow(np.linspace(0, 1, len(list_of_lines)))
        iter_colors = iter(colors)

        for line in range(len(list_of_lines)):
            x = t[line].astype(int)
            y = list_of_lines[line]
            my_color = colors[line]
            new_x,new_y= PlotHelper.GetAverageOfUniqueValues(x, y)
            try:
                ax.plot(new_x, interp1d(new_x, new_y, kind='cubic')(new_x),color=next(iter_colors))
            except:
                if(len(new_x)==1):
                    ax.plot(new_x, new_y,color=next(iter_colors))
                else:
                    ax.plot(new_x, interp1d(new_x, new_y, kind='quadratic')(new_x),color=next(iter_colors))


        ax.legend(legends, loc=1)
        for line in range(len(list_of_lines)):
            color = colors[line]
            for r in range(len(t[line])):
                ax.scatter(t[line][r], list_of_lines[line][r], color=color)
                if texts:
                    ax.text(t[line][r] + 1, list_of_lines[line][r] + 0.05, texts[line][r], fontsize=7)
        ax.legend(legends)
        colors = iter(colors)
        for leg in ax.get_legend().legendHandles:
            leg.set_color(next(colors))
        if (xLimMax != 0):
            ax.set_xlim(xLimMin, xLimMax)
        if (yLimMax != 0):
            ax.set_ylim(yLimMin, yLimMax)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)

        #ax.set_title(title)
        if invert_x:
            ax.invert_xaxis()
        if set_x_log:
            ax.set_xscale('log')
        if set_y_log:
            ax.set_yscale('log')
        if(grid):
            plt.grid(True)
        plt.savefig("./img/" + title + ".png")

        if (show):
            plt.show()
        plt.close()
        


    @staticmethod
    def plot3d(x, y, z, xLimMin=0, xLimMax=0, yLimMin=0, yLimMax=0, label_x="x", label_y="y", label_z="z", title="",
               show=False):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(x, y, z)
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        ax.set_zlabel(label_z)
        ax.set_title(title)
        if (xLimMax != 0):
            ax.set_xlim(xLimMin, xLimMax)
        if (yLimMax != 0):
            ax.set_ylim(yLimMin, yLimMax)
        plt.savefig("../img/" + title + ".png")
        plt.grid(True)
        if (show):
            plt.show()
        plt.close()
        
        
    @staticmethod
    def scatter3d(x, y, z, xLimMin=0, xLimMax=0, yLimMin=0, yLimMax=0, label_x="x", label_y="y", label_z="z", label="",
                  color="", title="", show=False, legend=""):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        ax.set_zlabel(label_z)
        ax.set_title(title)
        if label:
            ax.set_label(label)
        if label:
            ax.set_label(label)
        for line in range(len(x)):
            x_line = x[line]
            y_line = y[line]
            z_line = z[line]
            color_line =color[line]
            if color_line:
                ax.scatter(x_line, y_line, z_line, c=color_line)
            else:
                ax.scatter(x_line, y_line, z_line)
        if (xLimMax != 0):
            ax.set_xlim(xLimMin, xLimMax)
        if (yLimMax != 0):
            ax.set_ylim(yLimMin, yLimMax)
        plt.savefig("./img/" + title + ".png")
        ax.legend(legend)
        plt.grid(True)
        if (show):
            plt.show()
        plt.close()
        
    @staticmethod
    def interpolate_scatter3d(x, y, z, xLimMin=0, xLimMax=0, yLimMin=0, yLimMax=0, label_x="x", label_y="y", label_z="z", label="",
                  color="", title="", show=False, legend=""):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        ax.set_zlabel(label_z)
        ax.set_title(title)
        if label:
            ax.set_label(label)
        for line in range(len(x)):
            x_line = x[line]
            y_line = y[line]
            z_line = z[line]
            color_line =color[line]
            new_x,new_y,new_z= PlotHelper.GetAverageOfUniqueValues3D(x_line, y_line,z_line)
            ax.plot(new_x, new_y,new_z,color_line)
            if color_line:
                ax.scatter(x_line, y_line, z_line, c=color_line)
            else:
                ax.scatter(x_line, y_line, z_line)
        if (xLimMax != 0):
            ax.set_xlim(xLimMin, xLimMax)
        if (yLimMax != 0):
            ax.set_ylim(yLimMin, yLimMax)
        plt.savefig("./img/" + title + ".png")
        ax.legend(legend)
        plt.grid(True)
        if (show):
            plt.show()
        plt.close()
        
    @staticmethod
    def interpolate3d(x, y, z, xLimMin=0, xLimMax=0, yLimMin=0, yLimMax=0, label_x="x", label_y="y",
                              label_z="z", label="",
                              color="", title="", show=False, legend=""):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        ax.set_zlabel(label_z)
        ax.set_title(title)
        if label:
            ax.set_label(label)
        for line in range(len(x)):
            x_line = x[line]
            y_line = y[line]
            z_line = z[line]
            color_line = color[line]
            new_x, new_y, new_z = PlotHelper.GetAverageOfUniqueValues3D(x_line, y_line, z_line)
            ax.plot(new_x, new_y, new_z, color_line)
        if (xLimMax != 0):
            ax.set_xlim(xLimMin, xLimMax)
        if (yLimMax != 0):
            ax.set_ylim(yLimMin, yLimMax)
        plt.savefig("./img/" + title + ".png")
        ax.legend(legend)
        plt.grid(True)
        if (show):
            plt.show()
        plt.close()
        
    @staticmethod
    def scatter3dWithLines(lines, xLimMin=0, xLimMax=0, yLimMin=0, yLimMax=0, label_x="x", label_y="y", label_z="z",
                           label="", title="", show=False, legends=""):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        ax.set_zlabel(label_z)
        ax.set_title(title)
        if label:
            ax.set_label(label)
        for line in lines:
            x = line[0]
            y = line[1]
            z = line[2]
            if len(line) == 4:
                color = line[3]
                ax.scatter(x, y, z, c=color)
            else:
                ax.scatter(x, y, z)
        if legends:
            ax.legend(legends, loc=0)
        if (xLimMax != 0):
            ax.set_xlim(xLimMin, xLimMax)
        if (yLimMax != 0):
            ax.set_ylim(yLimMin, yLimMax)
        plt.savefig("./img/" + title + ".png")
        plt.grid(True)
        if (show):
            plt.show()
        plt.close()
                    
    @staticmethod
    def plotHeatMapWithLines(x, y, z, line1=[], line2=[], legends=[], xLimMin=0, xLimMax=0, yLimMin=0, yLimMax=0,
                             label_x="x", label_y="y", title="", show=False):
        # PlotHelper.scatter3d(r_matrix[:,0],r_matrix[:,1],r_matrix[:,2])
        fig = plt.figure()
        ax = fig.gca()
        ax.legend(legends, loc=0)
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        # ax.imshow(r_matrix,interpolation='nearest',extent=[t[0],t[-1],c_array[0],c_array[-1]])
        cntr = ax.contourf(x, y, z, cmap=cm.RdYlBu_r)
        fig.colorbar(cntr, ax=ax)
        if (xLimMax != 0):
            ax.set_xlim(xLimMin, xLimMax)
        if (yLimMax != 0):
            ax.set_ylim(yLimMin, yLimMax)
        ax.plot(line1, color="k", linewidth=1, linestyle='dashed')
        ax.plot(line2, color="m", linewidth=1, linestyle='dashed')
        ax.set_title(title)
        plt.savefig("./img/" + title + ".png")
        plt.grid(True)
        if (show):
            plt.show()