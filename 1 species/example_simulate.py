#For the 1 species, 2D, nonlocal reaction-advection-diffusion system
#Example of running a simulation using `simulation_class' and saving data,
# and plotting with `plot_class'

import numpy as np

from simulation_class import nonlocal_react_advect_diffuse_system
from plot_class import plotter

#Define parameters
U = 0.5
rho = 1
#note: for top-hat kernel xi should correspond to an integer number of meshpoints
xi = 0.4
mu = 100
L_x = 2
L_y = 2
kernel_name = "tophat"

mesh_points_x = 25 #More mesh points should be used than this - this is just for a quick example
mesh_points_y = int(mesh_points_x*(L_y/L_x)) #not used in simulation
time_span = (0, 10)
#which time values to store the solution values
time_evaluations = np.linspace(time_span[0], time_span[1], 10000)
#include one very early time point used for calculating linear growth modes
time_evaluations = np.insert(time_evaluations, 1, 1e-6)

RNG_seed = 15

#save text file specifying all the parameters
parameter_string = "kernel=%s \nU=%.10f \nrho=%.10f \nxi=%.10f \nmu=%.10f \nL_x=%.10f \nL_y=%.10f \nmesh_points_x=%d \nmesh_points_y=%d \ninitial_time=%.10f \nfinal_time=%.10f \nRNG_Seed=%d" %(kernel_name, U, rho, xi, mu, L_x, L_y, mesh_points_x, mesh_points_y, time_span[0], time_span[1], RNG_seed)
with open("parameters.txt", "w") as text_file:
    text_file.write(parameter_string)
    
    
#Create instance of nonlocal_react_advect_diffuse_system
system = nonlocal_react_advect_diffuse_system(U, rho, xi, mu, L_x, L_y, kernel_name,
                                      mesh_points_x, time_span,
                                      time_evaluations,
                                      integrator_method="BDF", seed=RNG_seed)

# Running simulation, checking and saving progress
print("Beginning simulation")
solution = system.simulate_with_progress("data_u.csv" , "data_t.csv")


# Plotting data
print("Beginning plotting")
plotter = plotter( rho=rho, U=U, mu=mu, xi=xi, kernel_name=kernel_name,
                       L_x=L_x, L_y=L_y, mesh_points_x=mesh_points_x,
                       mesh_points_y=mesh_points_y, solution_y=solution.y,
                       solution_t=solution.t, file_tag=r"",
                       use_PGF=False, raster_heatmap=True, num_heatmap_ticks=6)

print("Plotting and saving heatmap")
plotter.heatmap()
print("Plotting and saving dispersion relation")
plotter.dispersion_relation()
print("Creating and saving animation")
plotter.animation(sample_rate=100)

print("Everything finished")