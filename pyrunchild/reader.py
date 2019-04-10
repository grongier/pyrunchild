################################################################################
# Imports
################################################################################

import os

import numpy as np

################################################################################
# ChildReader
################################################################################

class ChildReader(object):

    def __init__(self, base_directory, base_name):

        self.base_directory = base_directory
        self.base_name = os.path.join(self.base_directory, base_name)

    def read_nodes(self,
                   read_boundary_flag=True,
                   read_edge_id=True,
                   return_array=True):

        time_slices = []
        nodes = []

        with open(self.base_name + '.nodes') as node_file, \
             open(self.base_name + '.z') as z_file:

            node_line = node_file.readline()
            z_line = z_file.readline()

            while node_line and z_line:

                time_slice = float(node_line.rstrip('\n'))
                node_line = node_file.readline()
                z_line = z_file.readline()
                number_nodes = int(node_line.rstrip('\n'))

                temp_nodes = []
                for i in range(number_nodes):
                    node_line = node_file.readline()
                    node_line = node_line.rstrip('\n').split(' ')
                    z_line = z_file.readline()
                    node = [float(i) for i in node_line[:2]]
                    node.append(float(z_line.rstrip('\n')))
                    if read_boundary_flag==True:
                        node.append(int(node_line[3]))
                    if read_edge_id==True:
                        node.append(int(node_line[2]))
                    temp_nodes.append(node)

                time_slices.append(time_slice)
                nodes.append(temp_nodes)

                node_line = node_file.readline()
                z_line = z_file.readline()

        if return_array == True:
            return np.array(nodes), time_slices
        else:
            return nodes, time_slices
        
    def read_triangles(self,
                       return_array=True):

        time_slices = []
        triangles = []

        with open(self.base_name + '.tri') as file:

            line = file.readline()

            while line:

                time_slice = float(line.rstrip('\n'))
                line = file.readline()
                number_triangles = int(line.rstrip('\n'))

                temp_triangles = []
                for i in range(number_triangles):
                    line = file.readline()
                    line = line.rstrip('\n').split(' ')
                    triangle = [int(i) for i in line[:3]]
                    temp_triangles.append(triangle)

                time_slices.append(time_slice)
                triangles.append(temp_triangles)

                line = file.readline()

        if return_array == True:
            return np.array(triangles), time_slices
        else:
            return triangles, time_slices
        
    def read_output_file(self,
                         file_suffix,
                         return_array=True):
        
        if file_suffix[0] != '.':
            file_suffix = '.' + file_suffix

        time_slices = []
        output = []

        with open(self.base_name + file_suffix) as file:

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
