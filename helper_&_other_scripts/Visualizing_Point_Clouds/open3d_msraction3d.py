## cd "G:\My Drive\Colab Notebooks\COMPSCI 674\Final Project\MSRAction3DFPS\MSRAction3D_fps"
## python "G:\My Drive\Colab Notebooks\COMPSCI 674\Final Project\Visualizing_Point_Clouds\open3d_msraction3d.py" a01_s01_e01_sdepth.npy

import open3d as Open3D
import open3d.core as Open3DCore
import numpy as NumPy
from os import chdir as modify_current_working_directory, getcwd as get_current_working_dir, makedirs as create_directory
from plyfile import PlyElement as make_ply_element
from plyfile import PlyData as make_ply_file
from numpy.lib import recfunctions
import sys

# Convert the data in a .npy file to a
# numpy array.
def object_to_tensor(x):
    tensor = []
    for i in range(x.shape[0]):
        tensor.append(x[i][None])
    return NumPy.concatenate(tensor, axis = 0)

# Read a depth map sequence from the MSR-Action3D Dataset into a variable.
modify_current_working_directory(r"G:\My Drive\Colab Notebooks\COMPSCI 674\Final Project\MSRAction3DFPS\MSRAction3D_fps")
depth_map_sequence = object_to_tensor(NumPy.load(sys.argv[1], allow_pickle = True))
print('Depth Map Sequence ==> dtype: {}, shape: {}'.format(depth_map_sequence.dtype, depth_map_sequence.shape))

# Store the total number of depth maps in the depth map sequence in a variable.
total_num_depth_maps = depth_map_sequence.shape[0]

# Create a new directory for the output PLY files, if it doesn't exist already.
create_directory('frames_{}'.format(sys.argv[1].replace('.npy', '')), exist_ok = True)

# Change the current working directory.
modify_current_working_directory('frames_{}'.format(sys.argv[1].replace('.npy', '')))

# Loop over the depth maps in the depth map sequence. 
for index, depth_map in enumerate(depth_map_sequence):
    # Structure the first depth map of the sequence and store the structured depth map in a variable.
    depth_map= recfunctions.unstructured_to_structured(depth_map, dtype = [('x', float), ('y', float), ('z', float)])
    print('Structured Depth Map Shape: {}'.format(depth_map.shape))

    # Flip the y-coordinates of the point clouds if required.
    if "MSRAction3D" in sys.argv[1]:
        depth_map['y'] *= -1

    # Create the vertices of the depth map to write to a PLY file.
    vertices = make_ply_element.describe(depth_map, 'vertex')

    # Create a ply file.
    ply_file = make_ply_file([vertices], text = True)

    # Write the PLY file to the storage.
    ply_file.write(sys.argv[1].replace('.npy', '_frame_{}.ply'.format(index)))