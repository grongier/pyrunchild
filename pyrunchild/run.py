################################################################################
# Imports
################################################################################

import os
import subprocess
import sys

from pyrunchild.writer import ChildWriter
from pyrunchild.reader import ChildReader

################################################################################
# Miscellaneous
################################################################################

def query_yes_no(question, default = "yes"):
    '''
    Ask a yes/no question via raw_input() and return the answer
    Written by Trent Mick under the MIT license, see:
    https://code.activestate.com/recipes/577058-query-yesno/
    
    @param question: A string that is presented to the user
    @param default: The presumed answer if the user just hits <Enter>.
                    It must be "yes" (the default), "no" or None (meaning
                    an answer is required of the user)
    @return The "answer", i.e., either "yes" or "no"
    '''
    
    valid = {"yes":"yes",   "y":"yes",  "ye":"yes",
             "no":"no",     "n":"no"}
    if default == None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while 1:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return default
        elif choice in valid.keys():
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "\
                             "(or 'y' or 'n').\n")

################################################################################
# Child
################################################################################

class Child(object):
    
    def __init__(self, base_directory, base_name, preset_parameters=True):

        self.base_directory = os.path.abspath(base_directory)
        os.makedirs(self.base_directory, exist_ok=True)
        self.base_name = os.path.join(self.base_directory, base_name)
        
        self.writer = ChildWriter(self.base_directory,
                                  preset_parameters=preset_parameters)
        self.writer.set_run_control(OUTFILENAME=base_name)
        self.reader = ChildReader(self.base_directory, base_name)
        
    def run(self, input_file=None):
        
        if input_file is None:
            input_file = self.writer.locate_input_file()
        
        if os.path.isfile(input_file) == False:
            write_input = query_yes_no('No input file defined, do you want the writer to create one based on the parameters defined so far?')
            if write_input == 'yes':
                self.writer.write_input_parameters()
            else:
                return False
            
        process = subprocess.Popen('child ' + input_file,
                                   cwd=self.writer.base_directory,
                                   shell=True,
                                   stdout = subprocess.PIPE, 
                                   stderr = subprocess.STDOUT)
        for line in iter(process.stdout.readline, b""):
            print(line.strip().decode('ascii'))
            
        return True
    
    def generate_nodes(self,
                       OUTFILENAME='nodes',
                       OPTINITMESHDENS=0,
                       X_GRID_SIZE=10000,
                       Y_GRID_SIZE=10000,
                       OPT_PT_PLACE=1,
                       GRID_SPACING=200,
                       NUM_PTS='n/a',
                       TYP_BOUND=1,
                       NUMBER_OUTLETS=0,
                       OUTLET_X_COORD='n/a',
                       OUTLET_Y_COORD='n/a',
                       MEAN_ELEV=0,
                       RAND_ELEV=1,
                       OPTINLET=0,
                       INLET_X='n/a',
                       INLET_Y='n/a',
                       SEED=100):
        
        node_writer = ChildWriter(self.base_directory)
        node_writer.set_run_control(OUTFILENAME=OUTFILENAME,
                                    RUNTIME=0,
                                    OPINTRVL=1,
                                    SEED=SEED)
        node_writer.set_mesh(OPTREADINPUT=10,
                             OPTINITMESHDENS=OPTINITMESHDENS,
                             X_GRID_SIZE=X_GRID_SIZE,
                             Y_GRID_SIZE=Y_GRID_SIZE,
                             OPT_PT_PLACE=OPT_PT_PLACE,
                             GRID_SPACING=GRID_SPACING,
                             NUM_PTS=NUM_PTS)
        node_writer.set_boundaries(TYP_BOUND=TYP_BOUND,
                                   NUMBER_OUTLETS=NUMBER_OUTLETS,
                                   OUTLET_X_COORD=OUTLET_X_COORD,
                                   OUTLET_Y_COORD=OUTLET_Y_COORD,
                                   MEAN_ELEV=MEAN_ELEV,
                                   RAND_ELEV=RAND_ELEV,
                                   OPTINLET=OPTINLET,
                                   INLET_X=INLET_X,
                                   INLET_Y=INLET_Y)
        node_writer.set_tectonics(OPTNOUPLIFT=1)
        node_writer.set_fluvial_transport(OPTNOFLUVIAL=1)
        node_writer.set_hillslope_transport(OPTNODIFFUSION=1)
        node_writer.set_various(OPTTSOUTPUT=0)
        node_writer.write_input_parameters()
        
        self.run(input_file=os.path.join(self.base_directory,
                                         OUTFILENAME + '.in'))
        
        node_reader = ChildReader(self.base_directory, OUTFILENAME)
        nodes, _ = node_reader.read_nodes()
        
        return nodes[0]
    
    def clean_directory(self):
        
        subprocess.call('rm ' + self.base_directory + 'run.time', shell=True)
        subprocess.call('rm ' + self.base_name + '*', shell=True)
