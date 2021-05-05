import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyjamalib
import scipy.signal,scipy.stats

class DataPlot:
    """Integrates all functions to perform data 
    processing to calculate the joint angle.

    See Also
    --------
    Developed by T.F Almeida in 25/03/2021
    
    For more information see:
    https://github.com/tuliofalmeida/pyjama    
    """
    def pyjama_subplot(data,time,box_data=None,title='Title',x_label='Time (s)',
                    y_label='Y Label',data_name=None,labels=None,box_text_a=None,
                    box_text_b=None,colors=None,grid=True,ret=False):
        
        """Plot the data according to the size of 
        the array in subplots. It allows to plot a box 
        with small information along with the data.

        Parameters
        ----------
        data: ndarray or tuple
            Variable with all datas (e.g.
            x = [data1,data2]).
        time: ndarray
            Time vector of data acquisition.
        box_data: ndarray or tuple
            Data you want to put in a separate 
            box. If empty, it will not plot anything.
        title: str
            Plot title.
        x_label: str
            Plot x-axis name.
        y_label: str
            Plot y-axis name.
        data_name: str list
            Title name of each subplot.
        labels: str list
            Labels of subplots according with the 
            size of input data array/tuple.
        box_text_b: str
            Text that you want to be displayed 
            before the data plotted in the box
        box_text_a: str
            Text that you want to be displayed 
            after the data plotted in the box
        colors: str list
            Name of the colors that you want the 
            data to be plotted (in order), 
            remembering that they must be names 
            supported by matplotlib
        grid: bool
            Plot grids in each subplot        
        ret: bool
            If true, the function will return to 
            the fig for a variable.

        Returns
        -------
        Subplots according with the data input size
        ret: True
            the function will return to 
            the fig for a variable.

        See Also
        --------
        Developed by T.F Almeida in 25/03/2021

        For more information see:
        https://github.com/tuliofalmeida/pyjama    
        """ 
        size = data[0].shape[1] 
        plt.tight_layout()
        textstr=[]

        fig, axs = plt.subplots(size,figsize=(18, 12))
        fig.suptitle(title, fontsize = 28)

        if type(data_name) == type(None):
            data_name = []
            for ç in range(size):    
                data_name.append('Data '+str(ç))

        if type(labels) == type(None):
            labels = []
            for ç in range(size):    
                labels.append('Data '+str(ç))

        if type(colors) == type(None):
            colors = []
            x = ['black', 'red', 'blue', 'green', 'orange', 'yellow',
                'gray', 'purple', 'pink', 'brown', 'cyan']
            for ç in range(size):    
                colors.append(x[ç])

        if type(box_data) != type(None):
            props = dict(boxstyle='round', facecolor='gray', alpha=0.3)
            box = True
            for ç in range(size):
                textstr.append(box_text_b +str(box_data[ç])+box_text_a)
                
        for ç in range(size):
            for i in range(len(data)):
                axs[ç].plot(time, data[i][:,ç],label=labels[i],color=colors[i])
                axs[ç].set_title(data_name[ç], fontsize=20) 
                if box:
                    axs[ç].text(-2.8, 1.25, textstr[ç], fontsize=14,
                    verticalalignment='top', bbox=props)

        for ax in axs.flat:
            ax.set(xlabel=x_label, ylabel=y_label)
            ax.yaxis.label.set_size(20)
            ax.xaxis.label.set_size(20)
            ax.label_outer()
            ax.legend(loc="upper right",fontsize=14)
            if grid:
                ax.grid()
            

        plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.4,hspace=0.4)

        if ret:
            return fig
    