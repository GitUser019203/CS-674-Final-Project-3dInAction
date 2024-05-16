# python "G:\My Drive\Colab Notebooks\COMPSCI 674\Final Project\Visualizing_Point_Clouds\ply_animation.py" 0.02 50 "G:\My Drive\Colab Notebooks\COMPSCI 674\Final Project\DFAUST\downloads\scans\50020\chicken_wings"
# python "G:\My Drive\Colab Notebooks\COMPSCI 674\Final Project\Visualizing_Point_Clouds\ply_animation.py" 0.03 54 "G:\My Drive\Colab Notebooks\COMPSCI 674\Final Project\Visualizing_Point_Clouds\frames_a01_s01_e01_sdepth"
# python "G:\My Drive\Colab Notebooks\COMPSCI 674\Final Project\Visualizing_Point_Clouds\ply_animation.py" 0.03 34 "G:\My Drive\Colab Notebooks\COMPSCI 674\Final Project\Visualizing_Point_Clouds\frames_a02_s01_e01_sdepth"

from open3d.io import read_point_cloud as get_pc_from_ply_file
from open3d.visualization import draw_geometries_with_custom_animation as open_windows, draw as make_animation 
from pathlib import Path
from os import chdir as modify_current_working_directory, getcwd as get_current_working_directory
from sys import argv as arguments_cmd_line

# Store the path of the folder containing the scans of the action specified on the command line in a variable.
scan_folder = arguments_cmd_line[3]

# Get all the point cloud frame files of the specified action and store them in a variable.
ply_files = Path(scan_folder).iterdir()

# Change the current working directory into the folder containing the scans.
modify_current_working_directory(scan_folder)

# Initialize a variable to hold the names of the PLY files.
ply_file_names = []

# Load all the PLY file names into a list and sort them by their order in the point cloud video.
for ply_file in ply_files:
	if ".ply" in ply_file.name:
		ply_file_names.append(ply_file)
ply_file_names.sort()

# Store the time each frame lasts and the total number of frames per clip
# in separate variables.
time_per_frame = float(arguments_cmd_line[1])
total_frames_in_clip = int(arguments_cmd_line[2])

# Initialize a variable to hold the point cloud frames.
geometry_objects = []

# Load the point cloud frames of the animation.
for ply_file_name in ply_file_names:
	total_geometries = len(geometry_objects)
	if total_geometries > total_frames_in_clip:
		break
	print('Loading point cloud frame {}.'.format(total_geometries))
	geometry_objects.append({'name': ply_file_name.name,'geometry': get_pc_from_ply_file(ply_file_name.name), 'time': time_per_frame * total_geometries})

# Make the animation.
make_animation(geometry_objects, animation_time_step = time_per_frame, animation_duration = time_per_frame * total_geometries)

