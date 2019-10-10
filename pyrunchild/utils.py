################################################################################
# Imports
################################################################################

import os
import re
from glob import glob
import textwrap
from threading import Timer
import numpy as np
from scipy import stats, linalg
from scipy.spatial import cKDTree
import colorsys
from matplotlib import colors

################################################################################
# Miscellaneous
################################################################################

def divide_line(string,
                line_size,
                start_str='#   ',
                join_str='\n#   ',
                end_str='\n'):
        
    return start_str + join_str.join(textwrap.wrap(string, line_size)) + end_str


def rename_old_file(file_path):

    if os.path.isfile(file_path) == True:
        new_file_path = file_path + '_old'
        old_file_paths = glob(new_file_path + '*')
        new_file_path += str(len(old_file_paths))

        os.rename(file_path, new_file_path)


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


class ReturnTimer(Timer):

    def __init__(self, interval, function, args=[], kwargs={}):

        self._original_function = function
        super(ReturnTimer, self).__init__(interval,
                                          self._do_execute,
                                          args,
                                          kwargs)
        self.result = None

    def _do_execute(self, *a, **kw):

        self.result = self._original_function(*a, **kw)

    def join(self):
        
        super(ReturnTimer, self).join()
        return self.result

################################################################################
# Interpolation
################################################################################

def griddata_idw(points, values, xi, nb_neighbors=8, p=2):
    """ Interpolate unstructured D-dimensional data using the inverse distance
        weighted interpolation

    @param points: ndarray of n data point coordinates, shape (n, D)
    @param values: ndarray of n data point values from V variables, shape (n, V)
    @param xi: ndarray of m data point coordinates, shape (m, D), at which to
               interpolate
    @param nb_neighbors: positive interger representing the number of neighbors
                         to consider when interpolating each point
    @param p: positive float representing the power parameter of the 
              interpolation

    @return a ndarray, shape (m, V), of interpolated values
    """
    tree = cKDTree(points)

    distances, neighbors = tree.query(xi, k=nb_neighbors)
    weights = np.zeros((1,) + distances.shape)
    weights[0, :, 0] = 1
    weights[0, distances[:, 0] != 0] = 1/distances[distances[:, 0] != 0]**p

    return np.sum(weights*values[:, neighbors], axis=-1)/np.sum(weights, axis=-1)

################################################################################
# RangeModel
################################################################################

class RangeModel(object):
    
    def __init__(self, start=0, step=1):

        self.start = start
        self.step = step

    def rvs(self, size=1):
        
        samples = np.arange(self.start, self.start + size, self.step)
        self.start += size

        if samples.shape[0] == 1:
            return samples[0]
        return samples

################################################################################
# BinaryModel
################################################################################

class BinaryModel:
    
    def __init__(self, model):
        
        self.model = model

    def rvs(self, size=1, random_state=None):
                  
        sample = self.model.rvs(size=size, random_state=random_state)

        if size == 1:
            return np.array((sample[0], 1 - sample[0]))
        return np.array((sample, 1 - sample)).T

################################################################################
# MixtureModel
################################################################################

class MixtureModel(object):
    
    def __init__(self, submodels, weights=None):
        
        self.submodels = submodels
        self.weights = weights

    def rvs(self, size=1, random_state=None):
        
        submodels_samples = np.zeros((size, len(self.submodels)))
        for i, submodel in enumerate(self.submodels):
            submodels_samples[:, i] = submodel.rvs(size=size,
                                                   random_state=random_state)
        choice = np.random.choice
        if random_state is not None:
            choice = random_state.choice
        rand_indices = choice(np.arange(len(self.submodels)),
                              size=size,
                              p=self.weights)
        samples = submodels_samples[np.arange(size), rand_indices]
        
        if samples.shape[0] == 1:
            return samples[0]
        return samples

################################################################################
# MemoryModel
################################################################################

class MemoryModel(object):
    
    def __init__(self, model, call_memory=1):
        
        self.model = model
        self.call_memory = call_memory
        self.call_count = 0
        self.last_sample = None

    def rvs(self, size=1, random_state=None):
        
        if self.call_count == 0:           
            self.last_sample = self.model.rvs(size=size,
                                              random_state=random_state)
            self.call_count += 1
        elif self.call_count < self.call_memory:
            self.call_count += 1
        elif self.call_count == self.call_memory:
            self.last_sample = self.model.rvs(size=size,
                                              random_state=random_state)
            self.call_count = 1
            
        if self.last_sample.shape[0] == 1:
            return self.last_sample[0]
        return self.last_sample

################################################################################
# Time series
################################################################################

class TimeSeriesConstraint:
    
    def __init__(self,
                 parameter,
                 initial_time,
                 time_steps,
                 final_time=None,
                 initial_value=None,
                 rate=None,
                 final_value=None,
                 value=None,
                 vmin=None,
                 vmax=None,
                 autocorr=None,
                 mode='interpolate'):
        
        valid_parameters = ['ST_PMEAN', 'ST_STDUR', 'ST_ISTDUR',
                            'UPRATE', 'FP_INLET_ELEVATION', 'FP_MU']
        if parameter not in valid_parameters:
            raise ValueError("Invalid time series parameter")
        if (value is None
            and (rate is None or initial_value is None or final_value is None)):
            raise ValueError("Invalid parameters: either use value, or rate with an initial and final value")
        if mode != 'interpolate' and mode != 'forward' and mode != '':
            raise ValueError("Invalid mode, should be interpolate or forward")

        self.parameter = parameter
        self.initial_time = initial_time
        self.time_steps = time_steps
        self.final_time = final_time
        self.rate = rate
        self.initial_value = initial_value
        self.final_value = final_value
        self.value = value
        self.autocorr = autocorr
        self.mode = mode

class ConstrainedTimeSeries:

    def __init__(self,
                 main_constraint,
                 other_constraints=None,
                 max_run_time=np.inf,
                 output_path='.',
                 inline=False,
                 set_out_intrvl=False):

        self.main_constraint = main_constraint
        self.other_constraints = other_constraints
        self.max_run_time = max_run_time
        self.output_path = os.path.abspath(output_path)
        self.inline = inline
        self.set_out_intrvl = set_out_intrvl

        self.times = dict()
        self.values = dict()
        self.is_called = dict()
        self.is_called['RUNTIME'] = None
        if self.set_out_intrvl == True:
            self.is_called['OPINTRVL'] = None
        self.is_called[main_constraint.parameter] = None
        if other_constraints is not None:
            for constraint in other_constraints:
                self.is_called[constraint.parameter] = None
        self.mode = dict()
        self.mode[main_constraint.parameter] = main_constraint.mode
        if other_constraints is not None:
            for constraint in other_constraints:
                self.mode[constraint.parameter] = constraint.mode
        self.parameter_values = dict()

    def get_value(self, value, random_state=None):

        if isinstance(value, (stats._distn_infrastructure.rv_frozen, MixtureModel)) == True:
            return value.rvs(random_state=random_state)
        return value

    def build_main_time_series(self, max_iter=1e8, random_state=None):

        key = self.main_constraint.parameter
        self.times[key] = [np.inf]
        self.values[key] = [np.nan]
        C = None
        if self.main_constraint.autocorr is not None:
            C = linalg.cholesky([[1., 0.], [self.main_constraint.autocorr, 1.]],
                                lower=True)
        
        iter_count = 0
        while (((self.main_constraint.rate is not None and 
                 self.values[key][-1] != self.main_constraint.final_value)
                or self.times[key][-1] > self.max_run_time)
               and iter_count < max_iter):

            self.times[key] = [self.main_constraint.initial_time]
            initial_value = self.main_constraint.initial_value
            previous_value = None
            if initial_value is None:
                initial_value = self.get_value(self.main_constraint.value,
                                               random_state=random_state)
                previous_value = self.main_constraint.value.cdf(initial_value)
                previous_value = stats.distributions.norm().ppf(previous_value)
            self.values[key] = [initial_value]
            while (self.values[key][-1] != self.main_constraint.final_value
                   and self.times[key][-1] < self.max_run_time):

                time_step = self.get_value(self.main_constraint.time_steps,
                                           random_state=random_state)
                new_time = self.times[key][-1] + time_step
                
                new_value = None
                if self.main_constraint.rate is not None:
                    rate = self.get_value(self.main_constraint.rate,
                                          random_state=random_state)
                    new_value = self.values[key][-1] + rate*time_step
                    if new_value > self.main_constraint.final_value:
                        rate = (self.main_constraint.final_value
                                - self.times[key][-1])/(new_value
                                                        - self.times[key][-1])
                        new_time = np.round(self.times[key][-1] + rate*time_step)
                        new_value = self.main_constraint.final_value
                else:
                    if new_time > final_time:
                        new_time = final_time
                        new_value = self.values[key][-1]
                    else:
                        new_value = self.get_value(stats.distributions.norm(),
                                                   random_state=random_state)
                        if C is not None:
                            new_value = np.dot(C, (previous_value, new_value))[1]
                            previous_value = new_value
                        new_value = stats.distributions.norm().cdf(new_value)
                        new_value = self.main_constraint.value.ppf(new_value)

                self.times[key].append(new_time)
                self.values[key].append(new_value)

            iter_count += 1
            
    def build_other_time_series(self, max_iter=1e8, random_state=None):
        
        final_time = self.times[self.main_constraint.parameter][-1]

        for constraint in self.other_constraints:
            key = constraint.parameter
            self.times[key] = [np.inf]
            self.values[key] = [np.nan]
            C = None
            if constraint.autocorr is not None:
                C = linalg.cholesky([[1., 0.], [constraint.autocorr, 1.]],
                                    lower=True)
            
            iter_count = 0
            while (((constraint.rate is not None
                     and self.values[key][-1] != constraint.final_value)
                    or self.times[key][-1] > final_time)
                   and iter_count < max_iter):

                self.times[key] = [constraint.initial_time]
                initial_value = constraint.initial_value
                previous_value = None
                if initial_value is None:
                    initial_value = self.get_value(constraint.value,
                                                   random_state=random_state)
                    previous_value = constraint.value.cdf(initial_value)
                    previous_value = stats.distributions.norm().ppf(previous_value)
                self.values[key] = [initial_value]
                while (self.values[key][-1] != constraint.final_value
                       and self.times[key][-1] < final_time):

                    time_step = self.get_value(constraint.time_steps,
                                               random_state=random_state)
                    new_time = self.times[key][-1] + time_step

                    new_value = None
                    if constraint.rate is not None:
                        rate = self.get_value(constraint.rate,
                                              random_state=random_state)
                        new_value = self.values[key][-1] + rate*time_step
                        if new_value > constraint.final_value:
                            rate = (constraint.final_value
                                    - self.times[key][-1])/(new_value
                                                            - self.times[key][-1])
                            new_time = np.round(self.times[key][-1] + rate*time_step)
                            new_value = constraint.final_value
                    else:
                        if new_time > final_time:
                            new_time = final_time
                            new_value = self.values[key][-1]
                        else:
                            new_value = self.get_value(stats.distributions.norm(),
                                                       random_state=random_state)
                            if C is not None:
                                new_value = np.dot(C, (previous_value, new_value))[1]
                                previous_value = new_value
                            new_value = stats.distributions.norm().cdf(new_value)
                            new_value = constraint.value.ppf(new_value)

                    self.times[key].append(new_time)
                    self.values[key].append(new_value)

                iter_count += 1

    def write_time_series(self, base_name=None, save_previous_file=True):

        for key in self.values:
            time_series = ''
            if self.inline == True:
                time_series = '@inline '
                for time, value in zip(self.times[key], self.values[key]):
                    time_series += str(time) + ':' + str(value) + ' '
                time_series += self.mode[key]
            else:
                file_name = key + '.dat'
                if base_name is not None:
                    file_name = base_name + '_' + file_name
                file_path = os.path.join(self.output_path, file_name)
                if save_previous_file == True:
                    rename_old_file(file_path)
                with open(file_path, 'w') as file:
                    for time, value in zip(self.times[key], self.values[key]):
                        file.write(str(time) + ' ' + str(value) + '\n')
                time_series = '@file ' + file_name + ' 1 2 '
                time_series += self.mode[key]

            self.parameter_values['RUNTIME'] = str(self.times[key][-1])
            self.is_called['RUNTIME'] = False
            if self.set_out_intrvl == True:
                self.parameter_values['OPINTRVL'] = self.parameter_values['RUNTIME']
                self.is_called['OPINTRVL'] = False
            self.parameter_values[key] = time_series
            self.is_called[key] = False

    def write(self,
              parameter_name,
              base_name=None,
              save_previous_file=True,
              max_iter=1e8,
              random_state=None):

        if self.is_called['RUNTIME'] is None:
            self.build_main_time_series(max_iter=max_iter,
                                        random_state=random_state)
            self.build_other_time_series(max_iter=max_iter,
                                         random_state=random_state)
            self.write_time_series(base_name=base_name,
                                   save_previous_file=save_previous_file)

        self.is_called[parameter_name] = True
        if all(i == True for i in self.is_called.values()):
            for key in self.is_called:
                self.is_called[key] = None

        return self.parameter_values[parameter_name]

################################################################################
# Colormaps
################################################################################

def _build_sand_cmap(light_fraction_1,
                     light_fraction_2,
                     light_fraction_3,
                     light_fraction_4,
                     use_gold_sand=False,
                     name='sand'):
    
    mississippi_mud = (15/360, 0.29 + light_fraction_1*0.29, 0.14)
    rio_grande_mud = (22/360, 0.48 + light_fraction_2*0.48, 0.33)
    yuma_sand = (50/360, 0.89 + light_fraction_3*0.89, 0.78)
    drifted_sand = (55/360, 0.94 + light_fraction_4*0.94, 0.40)
    color_list = [colorsys.hls_to_rgb(*mississippi_mud) + (1,),
                  colorsys.hls_to_rgb(*rio_grande_mud) + (1,),
                  colorsys.hls_to_rgb(*yuma_sand) + (1,),
                  colorsys.hls_to_rgb(*drifted_sand) + (1,)]
    if use_gold_sand:
        gold_sand = (46/360, 0.82 + light_fraction_3*0.82, 0.83)
        color_list[2] = colorsys.hls_to_rgb(*gold_sand) + (1,)
    
    return colors.LinearSegmentedColormap.from_list(name, color_list)

sand = _build_sand_cmap(-0.4, -0.18580533, -0.47128079, 0.,
                        use_gold_sand=True,
                        name='sand')

sand_light = _build_sand_cmap(0.25, 0.0854333, -0.4679755, 0.,
                              use_gold_sand=False,
                              name='sand_light')
