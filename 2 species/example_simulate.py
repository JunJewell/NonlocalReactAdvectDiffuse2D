#For the 2 species, 2D, nonlocal reaction-advection-diffusion system
#Example of running a simulation using `simulation_class' and saving data,
# and plotting with `plot_class'

import numpy as np

from simulation_class import nonlocal_react_advect_diffuse_system
from plot_class import plotter

#Define parameters
U = 0.25
V = 0.25
D = 1
#note: for top-hat kernel xi should correspond to an integer number of meshpoints
xi_matrix = np.array([ [0.4,0.8], [0.4,0.8] ])
mu_matrix = np.array([ [100,0], [0, -5000] ])
L_x = 2
L_y = 2
kernel_name_matrix = np.array([ ["tophat","tophat"], ["tophat","tophat"] ])

mesh_points_x = 25
mesh_points_y = int(mesh_points_x*(L_y/L_x)) #not used in simulation, just for plots
time_span = (0, 5)

#which time values to store the solution values
time_evaluations = np.linspace(time_span[0], time_span[1], 10000)
time_evaluations = np.insert(time_evaluations, 1, 1e-6)

RNG_seed = 1

#save text file specifying all the parameters
parameter_string = ("kernel(uu,uv,vu,vv)=%s,%s,%s,%s \n"
                    "U=%.10f \nV=%.10f \nD=%.10f \n"
                    "xi(uu,uv,vu,vv)=%.10f,%.10f,%.10f,%.10f \n"
                    "mu(uu,uv,vu,vv)=%.10f,%.10f,%.10f,%.10f \n"
                    "L_x=%.10f \nL_y=%.10f \nmesh_points_x=%d \n"
                    "mesh_points_y=%d \ninitial_time=%.10f \n"
                    "final_time=%.10f \nRNG_Seed=%d" 
                    %(kernel_name_matrix[0,0], kernel_name_matrix[0,1], kernel_name_matrix[1,0], kernel_name_matrix[1,1],
                      U, V, D,
                      xi_matrix[0,0], xi_matrix[0,1], xi_matrix[1,0], xi_matrix[1,1], 
                      mu_matrix[0,0], mu_matrix[0,1], mu_matrix[1,0], mu_matrix[1,1],
                      L_x, L_y, mesh_points_x, mesh_points_y, time_span[0], time_span[1], RNG_seed)
                    )
with open("parameters.txt", "w") as text_file:
    text_file.write(parameter_string)
    
#Create instance of nonlocal_react_advect_diffuse_system
system = nonlocal_react_advect_diffuse_system(U, V, D, xi_matrix, mu_matrix, 
                                      L_x, L_y, kernel_name_matrix, 
                                      mesh_points_x, time_span,
                                      time_evaluations, seed=RNG_seed)

# Running simulation, checking and saving progress
print("Beginning simulation")
solution = system.simulate_with_progress("data_u_v.csv" , "data_t.csv")


# Plotting data
print("Beginning plotting")
plotter = plotter( L_x=L_x, L_y=L_y, mesh_points_x=mesh_points_x,
                       mesh_points_y=mesh_points_y, solution_y=solution.y,
                       solution_t=solution.t, file_tag=r"",
                       use_PGF=True, raster_heatmap=True, num_heatmap_ticks=6)

print("Plotting and saving heatmaps")
plotter.heatmap(u_not_v=True)
plotter.heatmap(u_not_v=False)

print("Creating and saving animation for u")
plotter.animation(u_not_v=True, sample_rate=200)
print("Creating and saving animation for v")
plotter.animation(u_not_v=False, sample_rate=200)

print("Everything finished")