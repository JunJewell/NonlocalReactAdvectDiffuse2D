#For the 1 species, 2D, nonlocal reaction-advection-diffusion system
#Class takes input of solution data and system parameters
# and has methods that plot and save graphs or animations of density heatmaps,
# and graphs of the dispersion relation comparing theory and simulation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker

import matplotlib
from matplotlib_style import matplotlib_style

from scipy.fftpack import fft2, ifft2
from scipy.fftpack import fftfreq, fftshift

from dispersion_relation import dispersion
    

class plotter():
    
    def __init__(self, rho, U, mu, xi, kernel_name, L_x, L_y, mesh_points_x, mesh_points_y, solution_y, solution_t, 
                  file_tag, use_PGF=True, save=True, raster_heatmap=None, num_heatmap_ticks=6):
    
        
        #for use in dispersion relation plot
        self.rho = rho
        self.U = U
        self.mu = mu
        self.xi = xi
        self.kernel_name = kernel_name
        
        #for use in all plots
        self.mesh_points_x = mesh_points_x
        self.mesh_points_y = mesh_points_y
        self.L_x = L_x
        self.L_y = L_y
        
        self.u_over_time = solution_y.reshape(self.mesh_points_x, self.mesh_points_y, solution_t.size)
        self.solution_t = solution_t
        
        self.x_points = np.linspace(0, self.L_x, self.mesh_points_x)
        self.y_points = np.linspace(0, self.L_y, self.mesh_points_y)
        
        
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
    
     
    
    #Create a heatmap of the density at the a specific time. Default is final time
    def heatmap(self, time_index=-1):
        

        fig, ax = plt.subplots()
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '5%', '5%')
        pcm = ax.pcolormesh(self.x_points, self.y_points,
                            self.u_over_time[:,:,time_index].T,
                            cmap="magma", rasterized=self.raster_heatmap)
        cb = fig.colorbar(pcm, cax=cax)
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(self.num_heatmap_ticks))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(self.num_heatmap_ticks))
        
        if (self.save):
            fig.savefig("%s_heatmap_t=%.2f.eps" %(self.file_tag,
                                                     self.solution_t[time_index]))
            


    #Create mp4 of heatmap of density changing over time
    def animation(self, sample_rate=100):

        fig_anim, ax_anim = plt.subplots(1,1, figsize=(14,12))
        
        #take a frame every "sample_rate" number of time points
        frame_indices = np.arange(0, self.solution_t.size-1, sample_rate, dtype=int)
        
        #create first frame
        div = make_axes_locatable(ax_anim)
        cax = div.append_axes('right', '5%', '5%')   
        pcm = ax_anim.pcolormesh(self.x_points, self.y_points, 
                                 self.u_over_time[:,:,0].T,
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
                                     self.u_over_time[:,:,i].T,
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
            f = r"%s_anim.mp4" %(self.file_tag)
            writervideo = animation.FFMpegWriter(fps=10) 
            anim.save(f, writer=writervideo)
    
    
    #Creates a graph of the dispersion relation relating growth rate, lambda,
    #with spatial mode, k. Includes a lineplot from linear stability theory
    #and scatter points from simulation
    def dispersion_relation(self, sample_t=1, xlims=None, ylims=None):
        
        #sample_t is the index of the time point at which to sample the change in Fourier mode amplitude
        #should be early in the simulation so still approximately in the linear regime
        
        #Calculate Fourier transform of u(x,t) at t=0 and t=time[sample_t]
        #taking away U from u(x,t) because that only gives a delta function at 0,0
        fourier_initial = fft2(self.u_over_time[:,:,0]-self.U, shape=self.u_over_time[:,:,0].shape)
        fourier_initial = fftshift(fourier_initial)
        
        fourier_later = fft2(self.u_over_time[:,:,sample_t]-self.U, shape=self.u_over_time[:,:,0].shape)
        fourier_later = fftshift(fourier_later)
        
        #Mode numbers
        N_x = fftshift(fftfreq(fourier_initial.shape[0], self.L_x/self.mesh_points_x))*self.L_x
        N_y = fftshift(fftfreq(fourier_initial.shape[1], self.L_y/self.mesh_points_y))*self.L_y
        
        #Log of amplitude at sample_t normalised by initial amplitude
        log_normalised = np.log(fourier_later/fourier_initial)

        
        #Create array of k values that each have corresponding lambda values
        numerical_k_points = []
        numerical_lambda_points = []
        mode_index = np.array(np.where(log_normalised))
        for index_pair in mode_index.T:
            
            numerical_k_points.append(2*np.pi*np.sqrt( np.square(N_x[index_pair[0]]/self.L_x) + np.square(N_y[index_pair[1]]/self.L_y)))
            
            #real part of lambda taken (any imaginary part would have been small and caused by numerical error)
            numerical_lambda_points.append((log_normalised[index_pair[0], index_pair[1]]/self.solution_t[sample_t]).real)
        
        numerical_k_points = np.array(numerical_k_points)
        numerical_lambda_points = np.array(numerical_lambda_points)
        
        
        
        #Theory points based on linear stability analysis
        if ylims is None:
            theory_k_points = np.linspace(0.01, 
                                          2*np.pi*np.sqrt( np.square(self.mesh_points_x/self.L_x) +
                                                          np.square(self.mesh_points_y/self.L_y)),
                                          1000)
        
        else:
            theory_k_points = np.linspace(0.01, xlims[1] ,1000)
        
        theory_lambda_points = dispersion(theory_k_points,
                                                       self.rho, self.U,
                                                       self.mu, self.xi,
                                                       self.kernel_name).dispersion_relation()        
        
        
        fig_disp, ax_disp = plt.subplots(1,1, figsize=(10,8))
        ax_disp.plot(theory_k_points, theory_lambda_points, label="Theory")
        ax_disp.scatter(numerical_k_points, numerical_lambda_points, 
                        label="Simulation", marker='.', c='r', s=50)
        ax_disp.axhline(y=0, c='black', linestyle='solid')
        ax_disp.set_xlabel("k")
        ax_disp.set_ylabel("$\lambda$")
        ax_disp.legend()
        
        if (xlims is None and ylims is None):
            #setting useful y and x limits using heuristics
            max_x_index = np.where(np.sign(theory_lambda_points[:-1]) != np.sign(theory_lambda_points[1:]))[0] + 1
            if(max_x_index.size == 0):
                max_x_index=20
            else:
                max_x_index = max_x_index[1]
            max_x_index = np.min([theory_k_points.size - 1, int(1.6*max_x_index)])
            ax_disp.set_xlim(-0.1, theory_k_points[max_x_index])
            max_y = np.max([np.max(theory_lambda_points), np.max(numerical_lambda_points)])
            ax_disp.set_ylim(theory_lambda_points[max_x_index], max_y + 0.5*abs(max_y))
        else:
            ax_disp.set_xlim(xlims)
            ax_disp.set_ylim(ylims)
        
        if (self.save):
            fig_disp.savefig("%s_dispersionRelation.eps" %(self.file_tag))