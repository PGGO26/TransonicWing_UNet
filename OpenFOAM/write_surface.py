import numpy as np
x_min, x_max = -0.15, 1.65
boundary_y_min, boundary_y_max = 0, 1
Upper_z_min   , Upper_z_max    = 0.05, 0.5
Lower_z_min   , Lower_z_max    = 0.01, -0.5
res = 192

x_values = np.linspace(x_min, x_max, res)
y_values = np.linspace(boundary_y_min, boundary_y_max, res)
Upper_z_values  = np.linspace(Upper_z_min , Upper_z_max , res)
Lower_z_values  = np.linspace(Lower_z_min , Lower_z_max , res)

write_path_Upper_Surface = 'system/include/Upper_Surface'
with open(write_path_Upper_Surface, 'w') as file:
    file.write("pts\n(\n")
    for Upper_z in Upper_z_values:
        for y in y_values:
            y *= 1.5
            for x in x_values:
                file.write(f"({x} {y} {Upper_z})\n")
    file.write(");")
print(f'Write the surface points in {write_path_Upper_Surface} file.')

write_path_Lower_Surface = 'system/include/Lower_Surface'
with open(write_path_Lower_Surface, 'w') as file:
    file.write("pts\n(\n")
    for Lower_z in Lower_z_values:
        for y in y_values:
            y *= 1.5
            for x in x_values:
                file.write(f"({x} {y} {Lower_z})\n")
    file.write(");")
print(f'Write the surface points in {write_path_Lower_Surface} file.')

## InternalCloud
y_values_AIP = [0.2, 0.65, 0.8, 0.9, 0.96]
z_min_AIP, z_max_AIP = -0.9, 0.9
res_AIP = 1030

x_values_AIP = np.linspace(x_min, x_max, res_AIP)
z_values_AIP = np.linspace(z_min_AIP, z_max_AIP, res_AIP)
write_path_AIP = 'system/include/AIP'
with open(write_path_AIP, 'w') as file:
    file.write("pts\n(\n")
    for AIP_z in z_values_AIP:
        for AIP_y in y_values_AIP:
            AIP_y *= 1.1963
            for AIP_x in x_values_AIP:
                file.write(f"({AIP_x} {AIP_y} {AIP_z})\n")
    file.write(");")
print(f'Write the surface points in {write_path_AIP} file.')