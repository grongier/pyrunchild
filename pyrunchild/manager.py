"""CHILD files manager"""

# LICENCE GOES HERE


import os
from glob import glob
import numpy as np
from scipy import interpolate
# from numba import jit

from .utils import sorted_alphanumeric


################################################################################
# DataManager

class DataManager:
    """
    Base class to read and write files for CHILD.
    
    Parameters
    ----------
    base_directory : str
        Directory to read and write files.
    base_name : str
        Base name of CHILD's files.
    """
    def __init__(self, base_directory, base_name):

        self.base_directory = base_directory
        os.makedirs(self.base_directory, exist_ok=True)

        self.base_name = os.path.join(self.base_directory, base_name)

    def read_nodes(self,
                   realization=None,
                   read_boundary_flag=True,
                   read_edge_id=True):
        """
        Reads the .nodes and .z files created by a CHILD simulation.
        """
        time_slices = []
        nodes = []

        file_suffix = ''
        if realization is not None:
            file_suffix = '_' + str(realization)
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
        
    def read_triangles(self, realization=None):
        """
        Reads the .tri file created by a CHILD simulation.
        """
        time_slices = []
        triangles = []

        file_suffix = ''
        if realization is not None:
            file_suffix = '_' + str(realization)
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
        
    def read_output_file(self, file_type, realization=None, return_array=False):
        """
        Reads the any output file created by a CHILD simulation.
        """
        if file_type[0] != '.':
            file_type = '.' + file_type

        time_slices = []
        output = []

        file_suffix = ''
        if realization is not None:
            file_suffix = '_' + str(realization)
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
                output.append(np.array(temp_output))

                line = file.readline()

        if return_array == True:
            return np.array(output), time_slices
        else:
            return output, time_slices
        
    def read_file(self, file_name, is_size=False, return_array=True):
        """
        Reads any file.
        """
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

    def read_layers(self, file_nb=None, realization=None, to_array=False):
        """
        Reads a .lay* file created by a CHILD simulation.
        """
        layers = []
        max_nb_layers = 0

        if file_nb is None:
            layers_files = glob(self.base_name + '*.lay*')
            layers_files = sorted_alphanumeric(layers_files)
            if len(layers_files) > 0:
                file_nb = layers_files[-1].split('.')[-1][3:]
            else:
                return None
        file_suffix = ''
        if realization is not None:
            file_suffix = '_' + str(realization)
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
                    
                    node_layers.append([creation_time,
                                        activation_time,
                                        exposure_time,
                                        thickness,
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

    def read_channel_map(self, file_nb=None, realization=None):
        """
        Reads a .channelmap* file created by a CHILD simulation.
        """
        if file_nb is None:
            channel_map_files = glob(self.base_name + '*.channelmap*')
            channel_map_files = sorted_alphanumeric(channel_map_files)
            if len(channel_map_files) > 0:
                file_nb = channel_map_files[-1].split('.')[-1][10:]
            else:
                return None
        file_suffix = ''
        if realization is not None:
            file_suffix = '_' + str(realization)
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

    def read_top_layer(self, height=None, width=None, file_nb=None, realization=None):
        """
        Reads a .top* file created by a CHILD simulation.
        """
        if file_nb is None:
            top_files = glob(self.base_name + '*.top*')
            top_files = sorted_alphanumeric(top_files)
            if len(top_files) > 0:
                file_nb = top_files[-1].split('.')[-1][3:]
            else:
                return None
        file_suffix = ''
        if realization is not None:
            file_suffix = '_' + str(realization)
        if height is None or width is None:
            with open(self.base_name + file_suffix + '.channelmap' + file_nb) as map_file:
                map_line = map_file.readline()
                map_line = map_file.readline()
                map_line = map_file.readline()
                width = int(map_line.rstrip('\n')) - 1
                map_line = map_file.readline()
                height = int(map_line.rstrip('\n')) - 1
                
        with open(self.base_name + file_suffix + '.top' + file_nb) as top_file:

            top_line = top_file.readline()
            time_slice = float(top_line.rstrip('\n'))
            top_line = top_file.readline()

            top_layer = np.full((18, height, width), np.nan)

            for i in range(width):
                for j in range(height):

                    top_line = top_file.readline()
                    # x, y, z, mind, axisd, Ct1, Rt1, D1, DA1, DA2, DA3, DA4
                    top_line = top_line.rstrip(' \n').split(' ')
                    if len(top_line) == 18:
                        top_layer[:, j, i] = [float(k) for k in top_line]

        return top_layer

    def read_lithology(self, file_nb=None, realization=None):
        """
        Reads a .litho* file created by a CHILD simulation.
        """
        if file_nb is None:
            lithology_files = glob(self.base_name + '*.litho*')
            lithology_files = sorted_alphanumeric(lithology_files)
            if len(lithology_files) > 0:
                file_nb = lithology_files[-1].split('.')[-1][5:]
            else:
                return None
        file_suffix = ''
        if realization is not None:
            file_suffix = '_' + str(realization)
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

    def extract_boundary_map(self, boundary=0, file_nb=None, realization=None):
        """
        Extracts a layer boundary from a .litho* file created by a CHILD simulation.
        """
        boundary_map = self.read_channel_map(file_nb=file_nb,
                                             realization=realization)[:3]
        
        lithology = self.read_lithology(file_nb=file_nb,
                                        realization=realization)
        for v in range(boundary_map.shape[1]):
            for u in range(boundary_map.shape[2]):
                n = v*boundary_map.shape[2] + u
                for i in range(lithology[n].shape[0] - 1, boundary - 1, -1):
                    boundary_map[2, v, u] -= lithology[n][-(i + 1), 0]

        return boundary_map

    @staticmethod
    # @jit(nopython=True)
    def _interpolate_lithology_columns(z_cells,
                                       z_points,
                                       channel_map,
                                       lithology,
                                       basement_layers):
        """
        Interpolates the layer columns of a .litho* file created by a CHILD
        simulation into regularly sampled columns.
        """
        regular_cell_arrays = np.full((5, z_cells.shape[0]) + channel_map.shape[1:], np.nan)
        z_step = z_points[1] - z_points[0]

        for j in range(channel_map.shape[1]):
            for i in range(channel_map.shape[2]):

                k = z_points.shape[0] - 2
                z_interface = channel_map[2, j, i]
                n = j*channel_map.shape[2] + i
                l = 0
                layer_nb = len(lithology[n]) - basement_layers
                while k >= 0:
                    if l == 0:
                        if z_points[k] > z_interface:
                            # Cover
                            regular_cell_arrays[0, k, j, i] = 0
                        elif z_points[k + 1] >= z_interface > z_points[k]:
                            # Half-cover
                            ratio = (z_interface - z_points[k])/z_step
                            regular_cell_arrays[0, k, j, i] = ratio
                            regular_cell_arrays[1:, k, j, i] = lithology[n][l]
                            z_interface -= lithology[n][l][0]
                            l += 1
                        else:
                            z_interface -= lithology[n][l][0]
                            l += 1
                            k += 1
                    elif l < layer_nb:
                        if z_points[k] > z_interface:
                            # Sediments
                            regular_cell_arrays[0, k, j, i] = 1
                            regular_cell_arrays[1:, k, j, i] = lithology[n][l - 1]
                        elif z_points[k + 1] >= z_interface > z_points[k]:
                            # Half-sediments
                            ratio = (z_interface - z_points[k])/z_step
                            regular_cell_arrays[0, k, j, i] = 1
                            regular_cell_arrays[1:, k, j, i] = ratio*lithology[n][l] + (1 - ratio)*lithology[n][l - 1]
                            z_interface -= lithology[n][l][0]
                            l += 1
                        else:
                            z_interface -= lithology[n][l][0]
                            l += 1
                            k += 1
                    else:
                        if z_points[k] > z_interface:
                            # Sediments
                            regular_cell_arrays[0, k, j, i] = 1
                            regular_cell_arrays[1:, k, j, i] = lithology[n][l - 1]
                        elif z_points[k + 1] >= z_interface > z_points[k]:
                            # Half-basement
                            regular_cell_arrays[0, k, j, i] = ratio*2 + (1 - ratio)
                            regular_cell_arrays[1:, k, j, i] = lithology[n][l - 1]
                        else:
                            # Basement
                            regular_cell_arrays[0, k, j, i] = 2
                            regular_cell_arrays[1:, k, j, i] = lithology[n][l - 1]
                    k -= 1
                    
        return regular_cell_arrays

    def build_regular_lithology_grid(self,
                                     z_min,
                                     z_max,
                                     z_step,
                                     basement_layers=2,
                                     file_nb=None,
                                     realization=None):
        """
        Interpolates a .litho* file created by a CHILD simulation into a
        regular grid.
        """
        channel_map = self.read_channel_map(file_nb=file_nb,
                                            realization=realization)
        lithology = self.read_lithology(file_nb=file_nb,
                                        realization=realization)
        z_cells = np.arange(z_min + z_step/2, z_max, z_step)
        z_points = np.arange(z_min, z_max + z_step, z_step)

        regular_cell_arrays = DataManager._interpolate_lithology_columns(z_cells,
                                                                         z_points,
                                                                         channel_map,
                                                                         lithology,
                                                                         basement_layers)

        spacing = (channel_map[0, 0, 1] - channel_map[0, 0, 0],
                   channel_map[1, 1, 0] - channel_map[1, 0, 0],
                   z_step)
        extent = ((channel_map[0, 0, 0] - spacing[0]/2,
                   channel_map[0, 0, -1] + spacing[0]/2),
                  (channel_map[1, 0, 0] - spacing[1]/2,
                   channel_map[1, -1, 0] + spacing[1]/2),
                  (z_min,
                   z_max))

        return {'extent': np.array(extent),
                'spacing': np.array(spacing),
                'cell_arrays': regular_cell_arrays}

    def interpolate_channel_map_points(self, channel_map, kind='cubic'):
        """
        Interpolates a .channelmap* file created by a CHILD simulation from
        cell-centered to node-centered properties.
        """
        spacing = (channel_map[0, 0, 1] - channel_map[0, 0, 0],
                   channel_map[1, 1, 0] - channel_map[1, 0, 0])
        extent = (channel_map[0, 0, 0] - spacing[0]/2,
                  channel_map[0, 0, -1] + spacing[0]/2,
                  channel_map[1, 0, 0] - spacing[1]/2,
                  channel_map[1, -1, 0] + spacing[1]/2)
        
        channel_map_points = np.zeros((channel_map.shape[0],
                                       channel_map.shape[1] + 1,
                                       channel_map.shape[2] + 1))
        x = np.linspace(extent[0], extent[1], channel_map.shape[2] + 1)
        y = np.linspace(extent[2], extent[3], channel_map.shape[1] + 1)
        channel_map_points[:2] = np.meshgrid(x, y)

        f = interpolate.interp2d(channel_map[0, 0],
                                 channel_map[1, :, 0],
                                 channel_map[2],
                                 kind=kind)
        channel_map_points[2] = f(channel_map_points[0, 0],
                                  channel_map_points[1, :, 0])
        if channel_map.shape[0] == 4:
            f = interpolate.interp2d(channel_map[0, 0],
                                     channel_map[1, :, 0],
                                     channel_map[3],
                                     kind=kind)
            channel_map_points[3] = f(channel_map_points[0, 0],
                                      channel_map_points[1, :, 0])
        
        return channel_map_points

    def build_lithology_grid(self,
                             basement_layers=2,
                             return_points=True,
                             return_cells=False,
                             interpolation='linear',
                             file_nb=None,
                             realization=None):
        """
        Interpolates a .litho* file created by a CHILD simulation into a
        curvilinear grid.
        """
        channel_map = self.read_channel_map(file_nb=file_nb,
                                            realization=realization)
        lithology = self.read_lithology(file_nb=file_nb,
                                        realization=realization)
        nb_layers_max = max([len(i) for i in lithology]) - basement_layers

        channel_map_points = None
        nb_layers_max_points = None
        lithology_points = None
        if return_points == True:
            channel_map_points = self.interpolate_channel_map_points(channel_map,
                                                                     kind=interpolation)
            nb_layers_max_points = nb_layers_max + 1
            lithology_points = np.empty((3, nb_layers_max_points) + channel_map_points.shape[1:])
            lithology_points[:2] = channel_map_points[:2, np.newaxis]
            lithology_points[2] = 0
        channel_map_cells = None
        nb_layers_max_cells = None
        lithology_cells = None
        if return_cells == True:
            channel_map_cells = channel_map
            nb_layers_max_cells = nb_layers_max
            lithology_cells = np.empty((3, nb_layers_max_cells) + channel_map_cells.shape[1:])
            lithology_cells[:2] = channel_map_cells[:2, np.newaxis]
            lithology_cells[2] = 0
        lithology_cell_arrays = np.full((4, nb_layers_max) + channel_map.shape[1:],
                                        np.nan)
        
        for i in range(nb_layers_max):
            thickness_map = np.zeros(channel_map.shape[1:])
            for v in range(thickness_map.shape[0]):
                for u in range(thickness_map.shape[1]):
                    n = v*thickness_map.shape[1] + u
                    if -(i + 1) - basement_layers >= -lithology[n].shape[0]:
                        thickness_map[v, u] = lithology[n][-(i + 1) - basement_layers, 0]
                        lithology_cell_arrays[:, i, v, u] = lithology[n][-(i + 1) - basement_layers]
            if return_points == True:
                f = interpolate.interp2d(channel_map[0, 0],
                                         channel_map[1, :, 0],
                                         thickness_map,
                                         kind=interpolation)
                lithology_points[2, i] = f(channel_map_points[0, 0],
                                           channel_map_points[1, :, 0])
            if return_cells == True:
                lithology_cells[2, i] = thickness_map

        grid = dict()
        if return_points == True:
            for i in range(nb_layers_max_points - 2, -1, -1):
                lithology_points[2, i] += lithology_points[2, i + 1]
            lithology_points[2, :-1] = (lithology_points[2, 1:] + lithology_points[2, :-1])/2
            lithology_points[2, -1] /= 2
            lithology_points[2] = channel_map_points[2, np.newaxis] - lithology_points[2]
            grid['points'] = lithology_points
        if return_cells == True:
            for i in range(nb_layers_max_cells - 2, -1, -1):
                lithology_cells[2, i] += lithology_cells[2, i + 1]
            lithology_cells[2, :-1] = (lithology_cells[2, 1:] + lithology_cells[2, :-1])/2
            lithology_cells[2, -1] /= 2
            lithology_cells[2] = channel_map_cells[2, np.newaxis] - lithology_cells[2]
            grid['cells'] = lithology_cells
        grid['cell_arrays'] = lithology_cell_arrays

        return grid

    @staticmethod
    def _read_preservation_potential(path):
        """
        Reads a .presSurface or .presSubsurface file created by a CHILD
        simulation.
        """
        with open(path) as file:
            lines = []
            for line in file:
                line = [float(i) for i in line.rstrip('\n').split(' ')]
                lines.append(line)
            if len(lines) > 0:
                preservation_potential = np.full((len(lines), len(lines[-1])), np.nan)
                for i in range(len(lines)):
                    preservation_potential[i, :len(lines[i])] = lines[i]
                return preservation_potential
            else:
                return None

    def read_preservation_potential(self, realization=None):
        """
        Reads a .presSurface, .presSubsurface, and .presSubsurface2 file
        created by a CHILD simulation.
        """
        preservation_potential = dict()

        file_suffix = ''
        if realization is not None:
            file_suffix = '_' + str(realization)
            
        path = self.base_name + file_suffix + '.presSurface'
        preservation_potential['surface'] = DataManager._read_preservation_potential(path)
        path = self.base_name + file_suffix + '.presSubsurface'
        preservation_potential['subsurface'] = DataManager._read_preservation_potential(path)
        path = self.base_name + file_suffix + '.presSubsurface2'
        try:
            preservation_potential['subsurface2'] = np.loadtxt(path, skiprows=2)
        except:
            preservation_potential['subsurface2'] = None

        return preservation_potential

    def read_meander(self, realization=None):
        """
        Reads a .meander file created by a CHILD simulation.
        """
        file_suffix = ''
        if realization is not None:
            file_suffix = '_' + str(realization)
            
        path = self.base_name + file_suffix + '.meander'

        return np.loadtxt(path)
        
    def write_file(self, array, filename, add_size=False, time=None):
        """
        Writes any array to a file following CHILD's format.
        """
        with open(os.path.join(self.base_directory, filename), 'w') as file:
            
            if time is not None:
                file.write(' ' + str(time) + '\n')
            if add_size == True:
                file.write(str(array.shape[0]) + '\n')
            if len(array.shape) == 1:
                for i in range(array.shape[0]):
                    file.write(str(array[i]) + '\n')
            else:
                for i in range(array.shape[0]):
                    file.write(str(array[i, 0]))
                    for j in range(1, array.shape[1]):
                        file.write(' ' + str(array[i, j]))
                    file.write('\n')

    def write_points(self, array, filename, add_size=True):
        """
        Writes points following the format (x, y, x, boundary), for instance to
        define CHILD's simulation grid outside of CHILD.
        """
        with open(os.path.join(self.base_directory, filename + '.points'), 'w') as file:
            
            if add_size == True:
                file.write(str(array.shape[0]) + '\n')
            for i in range(array.shape[0]):
                for j in range(array.shape[1] - 1):
                    file.write(str(array[i, j]) + ' ')
                file.write(str(int(array[i, -1])) + '\n')
                    
    def write_uplift_map(self, uplift_map, file_name='upliftmap', file_nb=1):
        """
        Writes an uplift map into a file following CHILD format.
        """
        padding = ''
        if file_nb < 10:
            padding = '00'
        elif file_nb < 100:
            padding = '0'

        self.write_file(uplift_map, file_name + padding + str(file_nb))
        
    def write_uplift_maps(self, uplift_maps, uplift_times, file_name='upliftmap'):
        """
        Writes several uplifts map into a file following CHILD format.
        """
        for i, uplift_map in enumerate(uplift_maps):
            self.write_uplift_map(uplift_map,
                                  file_name=file_name,
                                  file_nb=i + 1)
            
        with open(os.path.join(self.base_directory,
                               file_name + 'times'), 'w') as file:
            for time in uplift_times:
                file.write(str(time) + '\n')
