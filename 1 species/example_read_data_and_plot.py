#For the 1 species, 2D, nonlocal reaction-advection-diffusion system
#Example of reading in data of u(x,t) and t values and parameters,
# and plotting and saving graphs using `plot_class'


import numpy as np

from plot_class import plotter

############   READ IN DATA    ######################

parameter_file_name = r"parameters.txt" 
u_file_name = r"data_u.csv" 
t_file_name = r"data_t.csv" 

#added to the start of the graph file name, can leave blank as r""
output_file_tag = r"A"

#Reading in parameters
print("Reading in parameters...")
with open(parameter_file_name) as param_file:
    parameters_string = param_file.read()

#All the values we need are between '=' and '\n'
#change all '\n' to '='
parameters_string = parameters_string.replace('\n','=')
#remove all white space
parameters_string = ''.join(parameters_string.split())
#take everything in between the '='
parameters_list = parameters_string.split(sep='=')
#only take every other element so we only take parameter values not names
parameters_list = parameters_list[1::2]

kernel_name = parameters_list[0]
U = float(parameters_list[1])
rho = float(parameters_list[2])
xi = float(parameters_list[3])
mu = float(parameters_list[4])
L_x = float(parameters_list[5])
L_y = float(parameters_list[6])
mesh_points_x = int(parameters_list[7])
mesh_points_y = int(parameters_list[8])
time_span = [float(parameters_list[9]), float(parameters_list[10])]
RNG_Seed = float(parameters_list[11])


#Reading in u and t data
print("Reading in u and t data...")
solution_y = np.genfromtxt(u_file_name, delimiter=',')
solution_t = np.genfromtxt(t_file_name, delimiter=',')



##########   PLOT FIGURES   ####################


print("Beginning plotting...")
plotter = plotter( rho=rho, U=U, mu=mu, xi=xi, kernel_name=kernel_name,
                       L_x=L_x, L_y=L_y, mesh_points_x=mesh_points_x,
                       mesh_points_y=mesh_points_y, solution_y=solution_y,
                       solution_t=solution_t, file_tag=output_file_tag,
                       use_PGF=False, raster_heatmap=True, num_heatmap_ticks=6)

print("heatmaps...")
#plot heatmap of u at final time   
plotter.heatmap()   
#plot heatmap of u at 4th time point                                  
plotter.heatmap(time_index=4)

print("dispersion relation...")
#plot dispersion relation with x,y limits calculated automatically
plotter.dispersion_relation()
#plot dispersion relation specifying x,y limits
plotter.dispersion_relation(xlims=[-0.1,8], ylims=[-60,10])

print("animation...")
plotter.animation(sample_rate=10)

print("Finished.")
