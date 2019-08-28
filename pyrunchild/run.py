################################################################################
# Imports
################################################################################

import os
import time
import subprocess
import sys
from functools import partial
from contextlib import ExitStack
import multiprocessing as mp
from threading import Timer
import numpy as np

from pyrunchild.writer import InputWriter
from pyrunchild.utils import ReturnTimer

################################################################################
# Child
################################################################################

class Child(InputWriter):
    
    def __init__(self,
                 base_directory,
                 base_name,
                 child_executable='child',
                 nb_realizations=1,
                 init_realization_nb=0,
                 preset_parameters=True,
                 seed=100):

        InputWriter.__init__(self,
                             base_directory,
                             base_name,
                             nb_realizations=nb_realizations,
                             init_realization_nb=init_realization_nb,
                             preset_parameters=preset_parameters,
                             seed=seed)

        self.child_executable = os.path.expanduser(child_executable)
        
    def talkative_run(self,
                      realization=0,
                      input_name=None,
                      silent_mode=False,
                      write_log=False):
        
        if input_name is None:
            if self.base_names[realization] is None:
                self.write_input_file(realization)
            input_name = self.base_names[realization]

        options = ''
        if silent_mode == True:
            options = '--silent-mode'

        with ExitStack() as stack:

            log_file = None
            if write_log == True:
                log_file = stack.enter_context(open(os.path.join(self.base_directory,
                                                                 input_name + '.log'),
                                                    'w'))
            process = stack.enter_context(subprocess.Popen([self.child_executable,
                                                            options,
                                                            input_name + '.in'],
                                                           cwd=self.base_directory,
                                                           stdout = subprocess.PIPE, 
                                                           stderr = subprocess.DEVNULL))

            for line in iter(process.stdout.readline, b""):
                print(line.strip().decode('ascii'))
                if log_file is not None:
                    log_file.write(line.strip().decode('ascii') + '\n')

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode,
                                                process.args)

    def terminate_unchanged_file(self,
                                 process,
                                 file_name,
                                 runtime,
                                 max_time):

        if file_name is not None:

            modif_time = os.path.getmtime(os.path.join(self.base_directory,
                                                       file_name))
            current_time = time.time()

            if current_time - modif_time > max_time:
                process.terminate()
                return 'Reason: File ' + file_name + ' has not been modified for the last ' + str(max_time) + 's at runtime ' + str(runtime) + 's\n'

    def run(self,
            realization=0,
            input_name=None,
            silent_mode=False,
            print_log=False,
            write_log=False,
            timeout=None,
            file_timeout=None,
            max_ellapsed_time=1e8,
            file_type=None,
            update_seed=True,
            resolve_parameters=False,
            max_attempts=1,
            save_previous_input_file=True,
            return_parameter_values=False):

        random_state = np.random.RandomState(self.seed + realization)
        if input_name is None:
            if self.base_names[realization] is None:
                self.write_input_file(realization,
                                      save_previous_file=save_previous_input_file,  
                                      random_state=random_state)
            input_name = self.base_names[realization]

        options = ''
        if silent_mode == True:
            options = '--silent-mode'

        return_code = None
        attempt = 0
        while return_code != 0 and attempt < max_attempts:

            with ExitStack() as stack:

                init_realization_nb = 0
                if self.init_realization_nb is not None:
                    init_realization_nb = self.init_realization_nb
                message = 'Realization ' + str(init_realization_nb + realization) + ': Failed attempt ' + str(attempt + 1) + '/' + str(max_attempts) + '\n'

                if attempt > 0:
                    if update_seed == True:
                        self.parameter_values[realization]['SEED'] += self.nb_realizations
                    if resolve_parameters == False:
                        random_state.seed(self.seed + realization)
                    self.write_input_file(realization,
                                          resolve_parameters=resolve_parameters,
                                          save_previous_file=save_previous_input_file,
                                          random_state=random_state)

                stdout = subprocess.DEVNULL
                if print_log == True:
                    stdout = subprocess.PIPE
                elif write_log == True:
                    stdout = stack.enter_context(open(os.path.join(self.base_directory,
                                                                   input_name + '.log'),
                                                      'w'))
                process = stack.enter_context(subprocess.Popen([self.child_executable,
                                                                options,
                                                                input_name + '.in'],
                                                               cwd=self.base_directory,
                                                               stdout = stdout, 
                                                               stderr = subprocess.DEVNULL,
                                                               universal_newlines=True))

                file_name = None
                if file_type is not None:
                    file_name = input_name + '.' + file_type
                file_timers = []
                runtime = file_timeout
                while runtime is not None and runtime + file_timeout < timeout:
                    file_timer = ReturnTimer(runtime,
                                             self.terminate_unchanged_file,
                                             args=[process,
                                                   file_name,
                                                   runtime,
                                                   max_ellapsed_time])
                    file_timers.append(file_timer)
                    runtime += file_timeout
                try:
                    for file_timer in file_timers:
                        file_timer.start()
                    stdout, _ = process.communicate(timeout=timeout)
                    error = None
                    i = 0
                    while error is None and i < len(file_timers):
                        error = file_timers[i].join()
                        if error is not None:
                            message += error
                        i += 1
                except subprocess.TimeoutExpired:
                    process.terminate()
                    message += 'Reason: Ran for too long\n'
                finally:
                    for file_timer in file_timers:
                        file_timer.cancel()

                return_code = process.returncode
                if return_code != 0:
                    if 'Reason' not in message:
                        if return_code is not None:
                            exception = subprocess.CalledProcessError(return_code, process.args)
                            message += 'Reason: ' + str(exception) + '\n'
                        else:
                            message += 'Reason: Unknown\n'

                    print(message, end='')
                    error_file_path = os.path.join(self.base_directory,
                                                   self.parameter_values[realization]['OUTFILENAME'] + '.err')
                    with open(error_file_path, 'a') as file:
                        file.write(message)

                    attempt += 1

                if print_log == True:
                    for line in stdout.split('\n'):
                        print(line)

        if return_parameter_values == True:
            return self.base_names[realization], self.parameter_values[realization]

    def multi_run(self,
                  n_jobs=-1,
                  chunksize=1,
                  print_log=False,
                  write_log=False,
                  timeout=None,
                  file_timeout=None,
                  max_ellapsed_time=1e8,
                  file_type='storm',
                  update_seed=True,
                  resolve_parameters=False,
                  save_previous_input_file=True,
                  max_attempts=1):

        if n_jobs == -1:
            try:
                n_jobs = int(os.environ["SLURM_NTASKS"])
            except KeyError:
                n_jobs = mp.cpu_count()

        with mp.Pool(processes=n_jobs) as pool:

            results = pool.map(partial(self.run,
                                       print_log=print_log,
                                       write_log=write_log,
                                       timeout=timeout,
                                       file_timeout=file_timeout,
                                       max_ellapsed_time=max_ellapsed_time,
                                       file_type=file_type,
                                       update_seed=update_seed,
                                       resolve_parameters=resolve_parameters,
                                       max_attempts=max_attempts,
                                       save_previous_input_file=save_previous_input_file,
                                       return_parameter_values=True),
                               range(self.nb_realizations),
                               chunksize=chunksize)

            self.base_names, self.parameter_values = zip(*results)
            self.base_names = list(self.base_names)
            self.parameter_values = list(self.parameter_values)

    def sub_run(self, timeout=1e8):

        if None in self.base_names:
            self.write_input_files()

        processes = []
        for r in range(self.nb_realizations):
            input_name = self.base_names[r] + '.in'
            processes.append(subprocess.Popen([self.child_executable, input_name],
                                              cwd=self.base_directory,
                                              stdout=subprocess.DEVNULL,
                                              stderr=subprocess.DEVNULL))
        for process in processes:
            timer = Timer(timeout, process.terminate)
            try:
                timer.start()
                process.wait()
            finally:
                timer.cancel()
    
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
        
        node_writer = InputWriter(self.base_directory, OUTFILENAME)
        node_writer.set_run_control(RUNTIME=0,
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
        node_writer.write_input_files()
        
        self.run(input_name=OUTFILENAME)
        
        nodes, _ = node_writer.read_nodes()

        if delete_files == True:
            subprocess.call('rm ' + os.path.join(self.base_directory, 'run.time'),
                            shell=True)
            subprocess.call('rm ' + os.path.join(self.base_directory, OUTFILENAME) + '*',
                            shell=True)
        
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
