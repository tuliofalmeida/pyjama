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
                    box_text_b=None,h_box=None,colors=None,grid=True,fig_size=(18,12),
                    title_size=28,subtitle_size=20,box_size=14,legend_size=14,
                    label_size=20,dpi=300,save_fig=False,ret=False):
        
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
            It was developed to receive short data 
            accompanied by a short text for the label, 
            such as error metrics (MPE, RMSE, MSE).
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
        h_box: float
            Constant to determine the box height.
        colors: str list
            Name of the colors that you want the 
            data to be plotted (in order), 
            remembering that they must be names 
            supported by matplotlib.
        grid: bool
            Plot grids in each subplot
        fig_size: tuple
            Image size values (x, y).
        title_size: int
            Title font size.
        subtitle_size: int
            Subtitle font size.
        box_size: int
            Box font size.
        legend_size: int
            Legend font size.
        label_size: int
            Axis font size.
        dpi: int
            Dots per inch, enhances the details 
            of the image.  
        save_fig: bool
            If true, the plot will be saved in 
            content or script folder with a 
            resolution of 300 dpi as default.    
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
        top_boxes = np.zeros((len(data),size))
        top_box_max = np.zeros(size)
        box = False
        
        fig, axs = plt.subplots(size,figsize=fig_size)
        fig.suptitle(title, fontsize = title_size)

        if type(data_name) == type(None):
            data_name = []
            for ç in range(size):    
                data_name.append('Data '+str(ç+1))
                
        if type(labels) == type(None):
            labels = []
            for ç in range(size):    
                labels.append('Data '+str(ç+1))

        if type(h_box) == type(None):
            h_box = .35

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

        for i in range(len(data)):
            for ç in range(size):
                top_boxes[i][ç] = max(data[i][:,ç])
                top_box_max[ç] = max(top_boxes[:,ç])

        for ç in range(size):
            for i in range(len(data)):
                axs[ç].plot(time, data[i][:,ç],label=labels[i],color=colors[i])
                axs[ç].set_title(data_name[ç], fontsize=subtitle_size) 
                if box:
                    axs[ç].text(-2.7, top_box_max[ç]+(top_box_max[ç]*h_box), textstr[ç], fontsize=box_size,
                    verticalalignment='top', bbox=props,horizontalalignment='left')       

        for ax in axs.flat:
            ax.set(xlabel=x_label, ylabel=y_label)
            ax.yaxis.label.set_size(label_size)
            ax.xaxis.label.set_size(label_size)
            ax.label_outer()
            ax.legend(loc="upper right",fontsize=legend_size)
            if grid:
                ax.grid()
            
        plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.4,hspace=0.4)
        
        if save_fig:
            name_fig = title + '.png'
            plt.savefig(name_fig, dpi=dpi)

        if ret:
            return fig
    