"""CHILD runner"""

# LICENCE GOES HERE


import os
from glob import glob
from functools import partial
from contextlib import ExitStack
import subprocess
import multiprocessing as mp
import numpy as np

from pyrunchild.writer import InputWriter


################################################################################
# Child

class Child(InputWriter):
    """
    Class to run one or several CHILD simulations.
    
    Parameters
    ----------
    base_directory : str
        Directory to read and write files.
    base_name : str
        Base name of CHILD's files.
    child_executable : str
        Path to CHILD's child_executable.
    nb_realizations : int (default 1)
        Number of simulations.
    init_realization_nb : int (default 0)
        Number of the first simulation.
    preset_parameters : bool (default True)
        If true, presets CHILD's parameters with default values. It is strongly
        advised to keep this true, because CHILD always requires some
        parameters in its input values, even when those parameters aren't
        actually required in a simulation.
    seed : int (default 42)
        Seed used for the random draw of parameter values, if required.
        
    """
    def __init__(self,
                 base_directory,
                 base_name,
                 child_executable='child',
                 nb_realizations=1,
                 init_realization_nb=0,
                 preset_parameters=True,
                 seed=42):

        InputWriter.__init__(self,
                             base_directory,
                             base_name,
                             nb_realizations=nb_realizations,
                             init_realization_nb=init_realization_nb,
                             preset_parameters=preset_parameters,
                             seed=seed)

        self.child_executable = os.path.expanduser(child_executable)

    def run(self,
            realization=0,
            input_name=None,
            silent_mode=False,
            print_log=False,
            write_log=False,
            timeout=None,
            update_seed=True,
            resolve_parameters=False,
            max_attempts=1,
            save_previous_input_file=True,
            extensions_to_remove=None,
            return_parameter_values=False):
        """
        Runs CHILD.
        
        Parameters
        ----------
        realization : int (default 0)
            Realization to run.
        input_name : str, optional (default None)
            Name of a specific input file to run (must be in the base directory).
        silent_mode : bool (default False)
            If true, runs CHILD in silent mode.
        print_log : bool (default False)
            If true, prints the log at the end of the simulation.
        write_log : bool (default False)
            If true, writes the log into a file.
        timeout : float, optional (default None)
            Maximum run time for the simulation.
        update_seed : bool (default True)
            If true and the simulation fails or times out, try again with a
            different seed in CHILD.
        resolve_parameters : bool (default True)
            If true and the simulation fails or times out, try again with a
            different parameter values. Only works if some of the parameters
            are randomly drawn.
        max_attempts : int (default 1)
            Maximum number of other attempts if the simulation fails or times
            out.
        save_previous_input_file : bool (default True)
            If true and another simulation is required, save the input files of
            the failed simulation.
        extensions_to_remove : list of str, optional (default None)
            Files to suppress at the end of the simulation.
        return_parameter_values : bool (default False)
            Return the parameter values of the simulation.

        Returns
        -------
        dict
            The parameter values, only if return_parameter_values is true.
            
        """
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

        success = False
        attempt = 0
        while success == False and attempt < max_attempts:

            with ExitStack() as stack:

                init_realization_nb = 0
                if self.init_realization_nb is not None:
                    init_realization_nb = self.init_realization_nb

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
                try:
                    _ = subprocess.run([self.child_executable, options, input_name + '.in'],
                                       stdout=stdout, 
                                       stderr=subprocess.PIPE, # subprocess.DEVNULL
                                       cwd=self.base_directory,
                                       timeout=timeout,
                                       check=True,
                                       text=True)
                except (subprocess.SubprocessError, subprocess.TimeoutExpired) as error:
                    message = 'Realization ' + str(init_realization_nb + realization) + ': Failed attempt ' + str(attempt + 1) + '/' + str(max_attempts) + '\n'
                    message += 'Reason: ' + str(error) + '\n'
                    print(message, end='')

                    error_file_path = os.path.join(self.base_directory,
                                                   self.parameter_values[realization]['OUTFILENAME'] + '.err')
                    with open(error_file_path, 'a') as file:
                        file.write(message)

                    if print_log == True:
                        print('\nLOG:')
                        output = error.stdout
                        if isinstance(output, (bytes, bytearray)):
                            output = output.decode()
                        for line in output.split('\n'):
                            print(line)
                    if error.stderr is not None:
                        print('ERROR:')
                        for line in error.stderr.split('\n'):
                            print(line)

                    attempt += 1
                else:
                    success = True

        if extensions_to_remove is not None:
            for extension in extensions_to_remove:
                file_name = self.parameter_values[realization]['OUTFILENAME'] + extension
                file_base_path = self.base_directory + '/' + file_name
                file_paths = glob(file_base_path)
                for path in file_paths:
                    if os.path.isfile(path):
                        os.remove(path)

        if return_parameter_values == True:
            return self.base_names[realization], self.parameter_values[realization]

    def multi_run(self,
                  n_jobs=-1,
                  chunksize=1,
                  print_log=False,
                  write_log=False,
                  timeout=None,
                  update_seed=True,
                  resolve_parameters=False,
                  save_previous_input_file=True,
                  extensions_to_remove=None,
                  max_attempts=1):
        """
        Runs multiple CHILD simulations in parallel.
        
        Parameters
        ----------
        n_jobs : int (default -1)
            Number of jobs to use for the computation. -1 means using all
            available processors.
        chunksize : int (default 1)
            Size of each chunk sent to the jobs.
        print_log : bool (default False)
            If true, prints the log at the end of the simulation.
        write_log : bool (default False)
            If true, writes the log into a file.
        timeout : float, optional (default None)
            Maximum run time for the simulation.
        update_seed : bool (default True)
            If true and the simulation fails or times out, try again with a
            different seed in CHILD.
        resolve_parameters : bool (default True)
            If true and the simulation fails or times out, try again with a
            different parameter values. Only works if some of the parameters
            are randomly drawn.
        save_previous_input_file : bool (default True)
            If true and another simulation is required, save the input files of
            the failed simulation.
        extensions_to_remove : list of str, optional (default None)
            Files to suppress at the end of the simulation.
        max_attempts : int (default 1)
            Maximum number of other attempts if the simulation fails or times
            out.
            
        """
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
                                       update_seed=update_seed,
                                       resolve_parameters=resolve_parameters,
                                       max_attempts=max_attempts,
                                       save_previous_input_file=save_previous_input_file,
                                       extensions_to_remove=extensions_to_remove,
                                       return_parameter_values=True),
                               range(self.nb_realizations),
                               chunksize=chunksize)

            self.base_names, self.parameter_values = zip(*results)
            self.base_names = list(self.base_names)
            self.parameter_values = list(self.parameter_values)

    def talkative_run(self,
                      realization=0,
                      input_name=None,
                      silent_mode=False,
                      write_log=False):
        """
        Runs CHILD while displaying the log information as it comes.
        
        Parameters
        ----------
        realization : int (default 0)
            Realization to run.
        input_name : str, optional (default None)
            Name of a specific input file to run (must be in the base directory).
        silent_mode : bool (default False)
            If true, runs CHILD in silent mode.
        write_log : bool (default False)
            If true, writes the log into a file.
            
        """
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
    
    def generate_nodes(self,
                       OUTFILENAME='nodes',
                       OPTINITMESHDENS=0,
                       X_GRID_SIZE=10000.,
                       Y_GRID_SIZE=10000.,
                       OPT_PT_PLACE=1,
                       GRID_SPACING=200.,
                       NUM_PTS='n/a',
                       TYP_BOUND=1,
                       NUMBER_OUTLETS=0,
                       OUTLET_X_COORD='n/a',
                       OUTLET_Y_COORD='n/a',
                       MEAN_ELEV=0.,
                       RAND_ELEV=1.,
                       SLOPED_SURF=0,
                       UPPER_BOUND_Z=0.,
                       SEED=42,
                       delete_files=True):
        """
        Runs CHILD over a null period of time just to get the nodes. It lets us
        modify those nodes before using them in another simulation.
        
        Parameters
        ----------
        OUTFILENAME : str (default 'nodes')
            Base name for output files.
        OPTINITMESHDENS : int (default 0)
            Option for densifying the initial mesh by inserting a new node at
            the circumcenter of each triangle. The value of this parameter is
            the number of successive densification passes (for example, if 2,
            then the mesh is densified twice).
        X_GRID_SIZE : float (default 10000.)
            Total length of model domain in x direction (m).
        Y_GRID_SIZE : float (default 10000.)
            Total length of model domain in y direction (m).
        OPT_PT_PLACE : int (default 1)
            Method of placing points when generating a new mesh:
            0 = uniform hexagonal mesh;
            1 = regular staggered (hexagonal) mesh with small random offsets in
                (x, y) positions;
            2 = random placement.
        GRID_SPACING : float (default 200.)
            Mean distance between grid nodes (m).
        NUM_PTS : int (default 'n/a')
            Number of points in grid interior, if random point positions are used.
        TYP_BOUND : int (default 1)
            Configuration of boundaries with a rectangular mesh:
            0 = open boundary in one corner;
            1 = open boundary along x = 0;
            2 = open boundaries along x = 0 and x = xmax;
            3 = open boundaries along all four sides;
            4 = single open boundary node at specified coordinates.
        NUMBER_OUTLETS : int (default 0)
            Number of outlets.
        OUTLET_X_COORD : float (default 'n/a')
            x coordinate of single-node outlet (open boundary) (m).
        OUTLET_Y_COORD : float (default 'n/a')
            y coordinate of single-node outlet (open boundary) (m).
        MEAN_ELEV : float (default 0.)
            Mean elevation of initial surface (m).
        RAND_ELEV : float (default 1.)
            Maximum amplitude of random variations in initial node elevations (m).
        SLOPED_SURF : int (default 0)
            Option for initial sloping surface (downward toward y = 0).
        UPPER_BOUND_Z : float (default 0.)
            If sloping initial surface is applied, this sets the slope by setting
            the altitude of the model edge at y = ymax (m).
        SEED : int (default 42)
            Seed for random number generation. Must be an integer.
        delete_files : bool (default True)
            If true, deletes the files generated by CHILD to create the nodes.

        Returns
        -------
        nodes : ndarray, shape (n, 5)
            The nodes.
            
        """
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
                                   UPPER_BOUND_Z=UPPER_BOUND_Z)
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
    
    def clean_directory(self):
        """
        Cleans the base directory of any file generated by CHILD.
        """
        subprocess.call('rm ' + self.base_directory + 'run.time', shell=True)
        subprocess.call('rm ' + self.base_name + '*', shell=True)
