################################################################################
# Imports
################################################################################

import os
import subprocess
import sys
from functools import partial
from multiprocessing import Pool
import numpy as np

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
    
    def __init__(self,
                 base_directory,
                 base_name,
                 child_executable='child',
                 preset_parameters=True,
                 seed=100):

        np.random.seed(seed)
        self.seed = seed

        self.base_directory = os.path.abspath(base_directory)
        os.makedirs(self.base_directory, exist_ok=True)
        self.base_name = os.path.join(self.base_directory, base_name)
        
        self.writer = ChildWriter(self.base_directory,
                                  preset_parameters=preset_parameters)
        self.writer.set_run_control(OUTFILENAME=base_name)
        self.reader = ChildReader(self.base_directory, base_name)

        self.child_executable = os.path.expanduser(child_executable)
        
    def run(self, input_name=None, silent_mode=False, total_silent_mode=False, write_log=False):
        
        if input_name is None:
            input_name = self.base_name
        
        # if os.path.isfile(input_file) == False:
        #     write_input = query_yes_no('No input file defined, do you want the writer to create one based on the parameters defined so far?')
        #     if write_input == 'yes':
        #         self.writer.write_input_parameters()
        #     else:
        #         return False

        options = ''
        if silent_mode == True:
            options = '--silent-mode'

        with subprocess.Popen([self.child_executable, options, input_name + '.in'],
                              cwd=self.base_directory,
                              stdout = subprocess.PIPE, 
                              stderr = subprocess.STDOUT) as process:
            if total_silent_mode == False or write_log == True:
                log_file = None
                if write_log == True:
                    log_file = open(os.path.join(self.base_directory,
                                                 input_name + '.log'),
                                    'w')
                for line in iter(process.stdout.readline, b""):
                    if total_silent_mode == False:
                        print(line.strip().decode('ascii'))
                    if log_file is not None:
                        log_file.write(line.strip().decode('ascii') + '\n')
                if log_file is not None:
                    log_file.close()
            else:
                process.communicate()
            
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode,
                                                process.args)

    def multi_run(self,
                  nb_realizations,
                  n_jobs=-1,
                  total_silent_mode=True,
                  write_log=False):

        if n_jobs == -1:
            try:
                n_jobs = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
            except KeyError:
                n_jobs = multiprocessing.cpu_count()

        with Pool(processes=n_jobs) as pool:

            base_name = self.writer.parameter_values['OUTFILENAME']
            input_file_names = []
            for i in range(nb_realizations):
                if self.seed is not None:
                    np.random.seed(self.seed + i)
                    self.writer.parameter_values['SEED'] = self.seed + i
                self.writer.parameter_values['OUTFILENAME'] = base_name + '_' + str(i + 1)
                input_file_names.append(base_name + '_' + str(i + 1))
                self.writer.write_input_parameters()

            pool.map(partial(self.run,
                             total_silent_mode=total_silent_mode,
                             write_log=write_log),
                     input_file_names)

    def sub_run(self,
                nb_realizations):

        base_name = self.writer.parameter_values['OUTFILENAME']
        processes = []
        for i in range(nb_realizations):
            if self.seed is not None:
                np.random.seed(self.seed + i)
                self.writer.parameter_values['SEED'] = self.seed + i
            self.writer.parameter_values['OUTFILENAME'] = base_name + '_' + str(i + 1)
            self.writer.write_input_parameters()

            input_name = base_name + '_' + str(i + 1) + '.in'
            processes.append(subprocess.Popen([self.child_executable, input_name],
                                              cwd=self.base_directory))
                                              # stdout=subprocess.DEVNULL,
                                              # stderr=subprocess.STDOUT)
        for process in processes:
            process.wait()
    
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
                       SLOPED_SURF=0,
                       UPPER_BOUND_Z=0,
                       OPTINLET=0,
                       INLET_X='n/a',
                       INLET_Y='n/a',
                       SEED=100,
                       delete_files=True):
        
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
                                   SLOPED_SURF=SLOPED_SURF,
                                   UPPER_BOUND_Z=UPPER_BOUND_Z,
                                   OPTINLET=OPTINLET,
                                   INLET_X=INLET_X,
                                   INLET_Y=INLET_Y)
        node_writer.set_tectonics(OPTNOUPLIFT=1)
        node_writer.set_fluvial_transport(OPTNOFLUVIAL=1)
        node_writer.set_hillslope_transport(OPTNODIFFUSION=1)
        node_writer.set_various(OPTTSOUTPUT=0)
        node_writer.write_input_parameters()
        
        input_name = os.path.join(self.base_directory, OUTFILENAME)
        self.run(input_name=input_name)
        
        node_reader = ChildReader(self.base_directory, OUTFILENAME)
        nodes, _ = node_reader.read_nodes()

        if delete_files == True:
            subprocess.call('rm ' + self.base_directory + 'run.time', shell=True)
            subprocess.call('rm ' + input_name + '*', shell=True)
        
        return nodes[0]

    def curate_floodplain_log(self,
                              input_name=None,
                              delete_original_log=False,
                              realization_nb=None):

        if input_name is None:
            input_name = self.base_name

        file_suffix = ''
        if realization_nb is not None:
            file_suffix = '_' + str(realization_nb)
        with open(input_name + file_suffix + '.log', 'r') as file,\
             open(input_name + file_suffix + '_floodplain.log', 'w') as curated_file:
            line = file.readline()
            while line:
                line = file.readline()
                if 'Channel goes up' in line:
                    curated_file.write(line)
                if 'Channel Belt Geometry:' in line:
                    curated_file.write(line)
                    for i in range(5):
                        line = file.readline()
                        curated_file.write(line)
                        
        if delete_original_log == True:
            subprocess.call('rm ' + input_name + file_suffix + '.log', shell=True)
    
    def clean_directory(self):
        
        subprocess.call('rm ' + self.base_directory + 'run.time', shell=True)
        subprocess.call('rm ' + self.base_name + '*', shell=True)
