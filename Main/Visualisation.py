import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.animation as animation
import matplotlib as mpl
import os
from datetime import datetime
import numpy as np

# An example of automatically creating a folder to save things into
temp_dir = "C:\\Users\\20183003\\Desktop\\Final_Project_Code\\temp"
results_dir = "C:\\Users\\20183003\\Desktop\\Final_Project_Code\\results"

name = None

# Create a folder to save the results in
if name is None:
    name = str(datetime.now().strftime("%m_%d_%H_%M_%S"))
    
res_dir = os.path.join(results_dir, name)
os.makedirs(res_dir, exist_ok=name is not None)
# A subfolder specifically for figures, for example
res_dir_figs = os.path.join(res_dir, "figs")
os.makedirs(res_dir_figs, exist_ok=name is not None)

class Visualization:
    """General visualization class"""

    def __init__(self) -> None:

        # For saving images this can be helpful
        # globally changes these settings so that one doesnt have to do this manually
        # for all plots
        
        # if we want to save plots and or videos
        self.save_plot = True
        self.save_vids = True
        if self.save_plot:
            # mpl.rcParams["xtick.labelsize"] = 20
            # mpl.rcParams["ytick.labelsize"] = 20
            # mpl.rcParams["xtick.major.size"] = 16
            # mpl.rcParams["ytick.major.size"] = 16
            # mpl.rcParams["axes.titlesize"] = 25
            mpl.rcParams["figure.figsize"] = (20, 6.7)

            self.fig_list = []
        
        # For videos it is always nice to have them reasonably large
        self.video_figsize = (20,10)

        if self.save_vids:
            self.writervideo = animation.FFMpegWriter(fps=5)

    # The show function creates one figure, which makes use of several plot functions
    # In my actual code I have a bunch of predefined show functions
    def show_plot_wab_evol(self,w,a,b, pts=np.array([[]])):
        fig, axs = plt.subplots(ncols=3)
        self.plot_wa_evol(axs[0], w, a, pts)
        self.plot_wb_evol(axs[1], w, b, pts)
        self.plot_ab_evol(axs[2], a, b, pts)

        # Save the plot to a list so that we can save all of them at the end
        if self.save_plot:
            self.fig_list.append((fig, "PlaceholderName.pdf"))

    def plot_wa_evol(self, ax: Axes,  w, a, pts=np.array([[]])):
        w_pts=np.arange(0.3,3,0.01)
        
        ax.plot(w.T, a.T, label="plot", linestyle='--',color='b')
        ax.plot(w_pts,1/w_pts*4/(5+4*0.001+4*0.001**2))
        if np.size(pts!=0):
            ax.plot(pts[:,0],pts[:,1],'*')
            
        ax.grid()
        #ax.legend()
        ax.set_xlabel('w')
        ax.set_ylabel('a')
        ax.set_title("Training paths, (w,a)-plane")
        
    def plot_wb_evol(self, ax: Axes,  w, b, pts=np.array([[]])):
        w_pts=np.arange(0.3,3,0.01)
        
        ax.plot(w.T, b.T, label="plot", linestyle='--',color='b')
        ax.plot(w_pts,-1/2*w_pts)
        if np.size(pts!=0):
            ax.plot(pts[:,0],pts[:,2],'*')
            
        ax.grid()
        #ax.legend()
        ax.set_xlabel('w')
        ax.set_ylabel('b')
        ax.set_title("Training paths, (w,b)-plane")
        
    def plot_ab_evol(self, ax: Axes,  a, b, pts=np.array([[]])):
        # In my case x and y are saved in some other python class, but you could for example
        # also make them function arguments
        w_pts=np.arange(0.3,3,0.01)
        
        ax.plot(a.T, b.T, label="plot", linestyle='--',color='b')
        ax.plot(1/w_pts*4/(5+4*0.001+4*0.001**2),-1/2*w_pts)
        if np.size(pts!=0):
            ax.plot(pts[:,1],pts[:,2],'*')
            
        ax.grid()
        #ax.legend()
        ax.set_xlabel('a')
        ax.set_ylabel('b')
        ax.set_title("Training paths, (a,b)-plane")

    # I save all of them at the end so that you can change settings for all of them at once, 
    # like making the background transparent
    def save_fig(self):
        if self.save_plot:
            for fig, name in self.fig_list:
                fig.savefig(os.path.join(res_dir_figs, name), transparent=False)

    # Same for this one
    def save_video(self, name, ani):
        ani.save(os.path.join(res_dir_figs, name), writer=self.writervideo)


# Helper functions to make it prettier
# This is usefull for the imshow command, as you dont want ticks on the x-y axis there
def imshow_axes(ax: Axes):
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])