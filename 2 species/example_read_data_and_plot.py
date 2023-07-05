#For the 2 species, 2D, nonlocal reaction-advection-diffusion system
#Example of reading in data of u(x,t) and t values and parameters,
# and plotting and saving graphs using `plot_class'


import numpy as np

from plot_class import plotter

############   READ IN DATA    ######################
parameter_file_name = r"parameters.txt" 
u_v_file_name = r"data_u_v.csv" 
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

kernel_name_matrix = np.full((2,2), None)
kernel_name_matrix[0,0] = parameters_list[0].split(sep=',')[0]
kernel_name_matrix[0,1] = parameters_list[0].split(sep=',')[1]
kernel_name_matrix[1,0] = parameters_list[0].split(sep=',')[2]
kernel_name_matrix[1,1] = parameters_list[0].split(sep=',')[3]


U = float(parameters_list[1])
V = float(parameters_list[2])
D = float(parameters_list[3])

xi_matrix = np.full((2,2), None)
xi_matrix[0,0] = float(parameters_list[4].split(sep=',')[0])
xi_matrix[0,1] = float(parameters_list[4].split(sep=',')[1])
xi_matrix[1,0] = float(parameters_list[4].split(sep=',')[2])
xi_matrix[1,1] = float(parameters_list[4].split(sep=',')[3])

mu_matrix = np.full((2,2), None)
mu_matrix[0,0] = float(parameters_list[5].split(sep=',')[0])
mu_matrix[0,1] = float(parameters_list[5].split(sep=',')[1])
mu_matrix[1,0] = float(parameters_list[5].split(sep=',')[2])
mu_matrix[1,1] = float(parameters_list[5].split(sep=',')[3])

L_x = float(parameters_list[6])
L_y = float(parameters_list[7])
mesh_points_x = int(parameters_list[8])
mesh_points_y = int(parameters_list[9])
time_span = [float(parameters_list[10]), float(parameters_list[11])]
RNG_Seed = float(parameters_list[12])

#Reading in u and t data
print("Reading in u and t data...")
solution_y = np.genfromtxt(u_v_file_name, delimiter=',')
solution_t = np.genfromtxt(t_file_name, delimiter=',')


##########   PLOT FIGURES   ####################
print("Beginning plotting...")
plotter = plotter( L_x=L_x, L_y=L_y, mesh_points_x=mesh_points_x,
                       mesh_points_y=mesh_points_y, solution_y=solution_y,
                       solution_t=solution_t, file_tag=output_file_tag,
                       use_PGF=True, raster_heatmap=True, num_heatmap_ticks=6)

print("heatmaps...")
#plotting heatmap of u at final time
plotter.heatmap(u_not_v=True)
#plotting heatmap of u at 10th time point
plotter.heatmap(u_not_v=True, time_index=10)
#plotting heatmap of v at final time
plotter.heatmap(u_not_v=False)


print("animation u...")
plotter.animation(u_not_v=True, sample_rate=100)
print("animation v...")
plotter.animation(u_not_v=False, sample_rate=100)

print("Finished.")
