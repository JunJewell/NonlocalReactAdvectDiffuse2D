#For the 2 species, 2D, nonlocal reaction-advection-diffusion system
#Class takes input of solution data and system parameters
# and has methods that plot and save graphs or animations of density heatmaps,

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker

import matplotlib
from matplotlib_style import matplotlib_style

    

class plotter():
    
    def __init__(self, L_x, L_y, mesh_points_x, mesh_points_y, solution_y, solution_t, 
                  file_tag, use_PGF=True, save=True, raster_heatmap=None, num_heatmap_ticks=6):
        
        self.mesh_points_x = mesh_points_x
        self.mesh_points_y = mesh_points_y
        
        self.u_v_over_time = solution_y.reshape(2, self.mesh_points_x, self.mesh_points_y, solution_t.size)
        self.solution_t = solution_t
        
        self.x_points = np.linspace(0, L_x, self.mesh_points_x)
        self.y_points = np.linspace(0, L_y, self.mesh_points_y)
        
        
        plt.rcParams.update(matplotlib_style(**{"text.usetex":True})) 
        self.use_PGF = use_PGF
        if (self.use_PGF):
            matplotlib.use('pgf')
        
        
        #rastering the heatmap significantly reduces the final file size
        self.raster_heatmap = raster_heatmap
        self.save = save
        #string to add to the start of the saved graph file, to distinguish it
        self.file_tag = file_tag
        
        self.num_heatmap_ticks = num_heatmap_ticks
    
 
    #Create a heatmap of the density at the a specific time for either u(x,t) or v(x,t)
    #Default is at the final time. u_not_v=True to plot u, and u_not_v=False to plot v.
    def heatmap(self, u_not_v = True, time_index=-1):
        
        if (u_not_v):
            species_index = 0
            species_name = "u"
        else:
            species_index = 1
            species_name = "v"
        
        fig, ax = plt.subplots()
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '5%', '5%')
        pcm = ax.pcolormesh(self.x_points, self.y_points,
                            self.u_v_over_time[species_index,:,:,time_index].T,
                            cmap="magma", rasterized=self.raster_heatmap)
        cb = fig.colorbar(pcm, cax=cax)
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(self.num_heatmap_ticks))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(self.num_heatmap_ticks))
        
        if (self.save):
            fig.savefig("%s_heatmap_%s_t=%.2f.eps" %(self.file_tag, species_name,
                                                  self.solution_t[time_index]))
            
    
    #Create mp4 of heatmap of density changing over time
    def animation(self, u_not_v = True, sample_rate=100):

        if (u_not_v):
            species_index = 0
            species_name = "u"
        else:
            species_index = 1
            species_name = "v"        
        
        fig_anim, ax_anim = plt.subplots(1,1, figsize=(14,12))
        
        #take a frame every "sample_rate" number of time points
        frame_indices = np.arange(0, self.solution_t.size-1, sample_rate, dtype=int)
        
        #create first frame
        div = make_axes_locatable(ax_anim)
        cax = div.append_axes('right', '5%', '5%')   
        pcm = ax_anim.pcolormesh(self.x_points, self.y_points, 
                                 self.u_v_over_time[species_index, :,:,0].T,
                                 cmap="magma", rasterized=self.raster_heatmap)
        cb = fig_anim.colorbar(pcm, cax=cax)
        
        ax_anim.xaxis.set_major_locator(ticker.MaxNLocator(self.num_heatmap_ticks))
        ax_anim.yaxis.set_major_locator(ticker.MaxNLocator(self.num_heatmap_ticks))
        
        #create all other frames
        def heat_map_frame(i):
            #clear axis
            ax_anim.cla()
            ax_anim.set_title("Time = %0.3f" %(self.solution_t[i]))
            ax_anim.set_xlabel("x")
            ax_anim.set_ylabel("y")
            
            pcm = ax_anim.pcolormesh(self.x_points, self.y_points,
                                     self.u_v_over_time[species_index, :,:,i].T,
                                     cmap="magma", rasterized=self.raster_heatmap)
            #colourbar will update scaling during the animation
            fig_anim.colorbar(pcm, cax=cax)
            ax_anim.set_aspect('equal')
            ax_anim.xaxis.set_major_locator(ticker.MaxNLocator(self.num_heatmap_ticks))
            ax_anim.yaxis.set_major_locator(ticker.MaxNLocator(self.num_heatmap_ticks))
            
        anim = animation.FuncAnimation(fig=fig_anim, func=heat_map_frame,
                                       frames=frame_indices, repeat=True,
                                       repeat_delay=5000, init_func=None)
        
        if (self.save):
            f = r"%s_anim_%s.mp4" %(self.file_tag, species_name)
            writervideo = animation.FFMpegWriter(fps=10) 
            anim.save(f, writer=writervideo)
    