################################################################################
# Imports
################################################################################

import os
import re
from glob import glob

import numpy as np
from scipy.interpolate import griddata
# from numba import jit

################################################################################
# Misc
################################################################################

def sorted_alphanumeric(l):
    '''
    Sort a list of strings with numbers
    @param l: The list
    @return The sorted list
    '''
    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(l, key=alphanum_key)

################################################################################
# ChildReader
################################################################################

class ChildReader(object):

    def __init__(self, base_directory, base_name):

        self.base_directory = base_directory
        self.base_name = os.path.join(self.base_directory, base_name)

    def read_nodes(self,
                   realization_nb=None,
                   read_boundary_flag=True,
                   read_edge_id=True):

        time_slices = []
        nodes = []

        file_suffix = ''
        if realization_nb is not None:
            file_suffix = '_' + str(realization_nb)
        with open(self.base_name + file_suffix + '.nodes') as node_file, \
             open(self.base_name + file_suffix + '.z') as z_file:

            node_line = node_file.readline()
            z_line = z_file.readline()

            while node_line and z_line:

                time_slice = float(node_line.rstrip('\n'))
                node_line = node_file.readline()
                z_line = z_file.readline()
                number_nodes = int(node_line.rstrip('\n'))

                nb_data = 3
                if read_boundary_flag == True:
                    nb_data += 1
                if read_edge_id == True:
                    nb_data += 1
                temp_nodes = np.empty((number_nodes, nb_data))
                for i in range(number_nodes):
                    node_line = node_file.readline()
                    node_line = node_line.rstrip('\n').split(' ')
                    z_line = z_file.readline()
                    temp_nodes[i, 0] = float(node_line[0])
                    temp_nodes[i, 1] = float(node_line[1])
                    temp_nodes[i, 2] = float(float(z_line.rstrip('\n')))
                    if read_boundary_flag == True:
                        temp_nodes[i, 3] = int(node_line[3])
                    if read_edge_id == True:
                        temp_nodes[i, 4] = int(node_line[2])

                time_slices.append(time_slice)
                nodes.append(temp_nodes)

                node_line = node_file.readline()
                z_line = z_file.readline()

        return nodes, time_slices
        
    def read_triangles(self,
                       realization_nb=None,
                       return_array=True):

        time_slices = []
        triangles = []

        file_suffix = ''
        if realization_nb is not None:
            file_suffix = '_' + str(realization_nb)
        with open(self.base_name + file_suffix + '.tri') as file:

            line = file.readline()

            while line:

                time_slice = float(line.rstrip('\n'))
                line = file.readline()
                number_triangles = int(line.rstrip('\n'))

                temp_triangles = np.empty((number_triangles, 3), dtype=np.int)
                for i in range(number_triangles):
                    line = file.readline()
                    line = line.rstrip('\n').split(' ')
                    temp_triangles[i] = [int(i) for i in line[:3]]

                time_slices.append(time_slice)
                triangles.append(temp_triangles)

                line = file.readline()

        return triangles, time_slices
        
    def read_output_file(self,
                         file_type,
                         realization_nb=None,
                         return_array=True):
        
        if file_type[0] != '.':
            file_type = '.' + file_type

        time_slices = []
        output = []

        file_suffix = ''
        if realization_nb is not None:
            file_suffix = '_' + str(realization_nb)
        with open(self.base_name + file_suffix + file_type) as file:

            line = file.readline()

            while line:

                time_slice = float(line.rstrip('\n'))
                line = file.readline()
                number_elements = int(line.rstrip('\n'))

                temp_output = []
                for i in range(number_elements):
                    line = file.readline()
                    temp_output.append(float(line))

                time_slices.append(time_slice)
                output.append(temp_output)

                line = file.readline()

        if return_array == True:
            return np.array(output), time_slices
        else:
            return output, time_slices
        
    def read_file(self,
                  file_name,
                  is_size=False,
                  return_array=True):

        output = []

        with open(os.path.join(self.base_directory, file_name)) as file:

            line = file.readline()
            if is_size == True:
                number_elements = int(line.rstrip('\n'))
                line = file.readline()

            while line:
                output.append(float(line))
                line = file.readline()

        if return_array == True:
            return np.array(output)
        else:
            return output

    def read_layers(self, file_nb, realization_nb=None, to_array=False):
        
        layers = []
        max_nb_layers = 0

        file_suffix = ''
        if realization_nb is not None:
            file_suffix = '_' + str(realization_nb)
        with open(self.base_name + file_suffix + '.lay' + file_nb) as layer_file:
        
            layer_line = layer_file.readline()
            time_slice = float(layer_line.rstrip('\n'))
            layer_line = layer_file.readline()
            nb_nodes = int(layer_line.rstrip('\n'))

            for node in range(nb_nodes):
                
                layer_line = layer_file.readline()
                nb_layers = int(layer_line.rstrip('\n'))
                if max_nb_layers < nb_layers:
                    max_nb_layers = nb_layers
                    
                node_layers = []
                
                for layer in range(nb_layers):
                    
                    layer_line = layer_file.readline()
                    layer_line = layer_line.rstrip('\n').split(' ')
                    creation_time = float(layer_line[0])
                    activation_time = float(layer_line[1])
                    exposure_time = float(layer_line[2])
                    layer_line = layer_file.readline()
                    layer_line = layer_line.rstrip('\n').split(' ')
                    thickness = float(layer_line[0])
                    erodibility = float(layer_line[1])
                    is_regolith = int(layer_line[2]) == 1
                    layer_line = layer_file.readline()
                    layer_line = layer_line.rstrip(' \n').split(' ')
                    grain_size_fractions = [float(i)/thickness for i in layer_line]
                    
                    node_layers.append([thickness,
                                        creation_time,
                                        activation_time,
                                        exposure_time,
                                        erodibility,
                                        is_regolith] + grain_size_fractions)
                    
                layers.append(np.array(node_layers))
                
        if to_array == True:
            array_layers = np.full((nb_nodes, max_nb_layers, len(layers[0][0])), np.nan)
            for i, node in enumerate(layers):
                for j, layer in enumerate(node):
                    array_layers[i, j] = layer
                if node.shape[0] < max_nb_layers:
                    for j in range(node.shape[0], max_nb_layers):
                        array_layers[i, j] = node[-1]
            return array_layers
                
        return layers

    def read_channel_map(self, file_nb, realization_nb=None):

        file_suffix = ''
        if realization_nb is not None:
            file_suffix = '_' + str(realization_nb)
        with open(self.base_name + file_suffix + '.channelmap' + file_nb) as map_file:

            map_line = map_file.readline()
            time_slice = float(map_line.rstrip('\n'))
            map_line = map_file.readline()
            nb_cells = int(map_line.rstrip('\n'))
            map_line = map_file.readline()
            width = int(map_line.rstrip('\n')) - 1
            map_line = map_file.readline()
            height = int(map_line.rstrip('\n')) - 1
            
            channel_map = np.full((4, height, width), np.nan)

            for j in range(height):
                for i in range(width):

                    map_line = map_file.readline()
                    # x, y, z, drainage
                    map_line = map_line.rstrip(' \n').split(' ')
                    channel_map[:, j, i] = [float(k) for k in map_line]
                
        return channel_map

    def read_lithology(self, file_nb, realization_nb=None):

        file_suffix = ''
        if realization_nb is not None:
            file_suffix = '_' + str(realization_nb)
        with open(self.base_name + file_suffix + '.litho' + file_nb) as file:

            line = file.readline()
            time_slice = float(line.rstrip('\n'))
            line = file.readline()
            nb_cells = int(line.rstrip('\n'))
            line = file.readline()
            width = int(line.rstrip('\n')) - 1
            line = file.readline()
            height = int(line.rstrip('\n')) - 1
            
            lithology = []

            for j in range(height):
                for i in range(width):
                    
                    line = file.readline()
                    nb_layers = int(line.rstrip('\n'))
                    layers = np.empty((nb_layers, 4))
                    for l in range(nb_layers):

                        line = file.readline()
                        # thickness, texture, last time, deposition time
                        line = line.rstrip(' \n').split(' ')
                        layers[l] = [float(k) for k in line]
                        
                    lithology.append(layers)
                
        return lithology

    @staticmethod    
    # @jit(nopython=True)
    def interpolate_lithology_grid(grid,
                                   data,
                                   channel_map,
                                   lithology,
                                   basement_layers):

        for j in range(grid.shape[2]):
            for i in range(grid.shape[3]):

                n = j*channel_map.shape[2] + i
                l = 0
                layer_nb = len(lithology[n]) + basement_layers
                z_interface = channel_map[2, j, i]

                for k in range(grid.shape[1] - 1, -1, -1):

                    if grid[0, k, j, i] <= z_interface and l < layer_nb:

                        while not (grid[0, k, j, i] <= z_interface
                                   and grid[0, k, j, i] > z_interface - lithology[n][l][0]):
                            z_interface -= lithology[n][l][0]
                            l += 1

                        if l < layer_nb:
                            data[:, k, j, i] = lithology[n][l]

        return np.concatenate((grid, data))

    def build_lithology_grid(self,
                             z_min,
                             z_max,
                             z_step,
                             basement_layers=-1,
                             file_nb=None,
                             realization_nb=None):

        if file_nb is None:
            lithology_files = glob(self.base_name + '*.litho*')
            lithology_files = sorted_alphanumeric(lithology_files)
            file_nb = lithology_files[-1].split('.')[-1][5:]

        channel_map = self.read_channel_map(file_nb,
                                            realization_nb=realization_nb)
        lithology = self.read_lithology(file_nb,
                                        realization_nb=realization_nb)

        x = channel_map[0, 0]
        y = channel_map[1, :, 0]
        z = np.arange(z_min, z_max, z_step)
        grid = np.array(np.meshgrid(z, y, x, indexing='ij'))
        data = np.full((4,) + grid.shape[1:], np.nan)

        lithology = ChildReader.interpolate_lithology_grid(grid,
                                                           data,
                                                           channel_map,
                                                           lithology,
                                                           basement_layers)

        return lithology
