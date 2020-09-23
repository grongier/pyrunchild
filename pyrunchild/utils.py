"""CHILD utils"""

# The MIT License (MIT)
# Copyright (c) 2020 CSIRO
#
# Author: Guillaume Rongier
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


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

def divide_line(string,
                line_size,
                start_str='#   ',
                join_str='\n#   ',
                end_str='\n'):
    """
    Divides a single-line string into several lines.

    Parameters
    ----------
    string : str
        The single-line string.
    line_size : int
        The line size.
    start_str : str
        A string to add at the beginning of new string.
    join_str : str
        A string to add between lines.
    end_str : str
        A string to add at the end of new string.

    Returns
    -------
    str
        The new string.

    """
    return start_str + join_str.join(textwrap.wrap(string, line_size)) + end_str


def rename_old_file(file_path):
    """
    Renames a file by adding '_old*' after its extension, where '*' is a number
    that gets incremented as the number of old files grows.

    Parameters
    ----------
    file_path : str
        The path to the file.

    """
    if os.path.isfile(file_path) == True:
        new_file_path = file_path + '_old'
        old_file_paths = glob(new_file_path + '*')
        new_file_path += str(len(old_file_paths))

        os.rename(file_path, new_file_path)


def sorted_alphanumeric(l):
    """
    Sorts a list of strings with numbers.

    Parameters
    ----------
    l : list
        The list to sort.

    Returns
    -------
    list
        The sorted list.

    """
    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(l, key=alphanum_key)


################################################################################
# Interpolation

def griddata_idw(points, values, xi, nb_neighbors=8, p=2.):
    """
    Interpolates unstructured D-dimensional data using the inverse distance
    weighted interpolation.

    Parameters
    ----------
    points : ndarray, shape (n, D)
        Data point coordinates.
    values : ndarray, shape (n, V)
        Data values.
    xi : ndarray, shape (m, D)
        Point coordinates at which to interpolate the data.
    n_neighbors : int (default 8)
        Number of neighbors to consider when interpolating each point.
    p : float (default 2.)
        Power parameter of the interpolation.

    Returns
    -------
    ndarray, shape (m, V)
        Interpolated values.

    """
    tree = cKDTree(points)

    distances, neighbors = tree.query(xi, k=n_neighbors)
    weights = np.zeros((1,) + distances.shape)
    weights[0, :, 0] = 1
    weights[0, distances[:, 0] != 0] = 1/distances[distances[:, 0] != 0]**p

    return np.sum(weights*values[:, neighbors], axis=-1)/np.sum(weights, axis=-1)


################################################################################
# Draw models

class RangeModel:
    """
    Model in which the values aren't random, but are consecutive.
    
    Parameters
    ----------
    start : int (default 0)
        Initial value of the range.
    step : int (default 1)
        Step between two values of the range.
        
    """
    def __init__(self, start=0, step=1):

        self.start = start
        self.step = step

    def rvs(self, size=1):
        
        samples = np.arange(self.start, self.start + size, self.step)
        self.start += size

        if samples.shape[0] == 1:
            return samples[0]
        return samples


class BinaryModel:
    """
    Model in which a first value v1 is randomly drawn between 0 and 1, and a
    second value v2 is equal to 1 - v1. The first value is returned with the
    first call to rvs, the second value with the second call.
    
    Parameters
    ----------
    model : scipy.stats' rv_continuous
        Model from which to draw the first value.
        
    """
    def __init__(self, model):
        
        self.model = model

        self.call_count = 0
        self.last_sample = None

    def rvs(self, size=1, random_state=None):
        
        if self.call_count == 0:           
            self.last_sample = self.model.rvs(size=size,
                                              random_state=random_state)
            self.call_count += 1
        else:
            self.last_sample = 1 - self.last_sample
            self.call_count -= 1
            
        if self.last_sample.shape[0] == 1:
            return self.last_sample[0]
        return self.last_sample


class MixtureModel:
    """
    Model in which the values are randomly drawn from several random models.
    
    Parameters
    ----------
    submodels : list of scipy.stats' rv_continuous
        Models from which to draw first values.
    weights : array-like, optional (default None)
        Weights of each sub-models to influence their selection.
        
    """
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


class MemoryModel:
    """
    Model in which the same random value can be returned several times.
    
    Parameters
    ----------
    model : scipy.stats' rv_continuous
        Model from which to draw the values.
    call_memory : int (default 1)
        Number of times the same value is returned.
        
    """
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


class DependencyModel:
    """
    Model in which the previous random values influence the new ones.
    
    Parameters
    ----------
    model : scipy.stats' rv_continuous
        Model from which to draw the values.
    call_memory : int (default 1)
        Number of times the same value is returned.
        
    """
    def __init__(self, model, call_memory=1):
        
        self.model = model
        self.args = model.args

        self.call_count = 0
        self.call_memory = call_memory
        self.last_sample = None

    def rvs(self, size=1, random_state=None):
        
        if self.call_count == 0:           
            self.last_sample = self.model.rvs(size=size,
                                              random_state=random_state)
            self.call_count += 1
        elif self.call_count < self.call_memory:
            self.call_count += 1
        else:
            for i in range(self.last_sample.shape[0]):
                self.model.args = (self.args[0],
                                   self.last_sample[i] - self.args[0])
                self.last_sample[i] = self.model.rvs(size=1,
                                                     random_state=random_state)
            self.call_count = 0
            self.model.args = self.args
            
        if self.last_sample.shape[0] == 1:
            return self.last_sample[0]
        return self.last_sample


class TwoGrainsModel:
    """
    Model to randomly draw grain sizes and their proportions in a two fractions
    mixture.
    
    Parameters
    ----------
    coarse_diameter : float or scipy.stats' rv_continuous
        Diameter of the coarse fraction.
    fine_diameter : float or scipy.stats' rv_continuous
        Diameter of the fine fraction.
    coarse_proportion : float or scipy.stats' rv_continuous
        Proportion of the coarse fraction. Must be between 0 and 1.
    is_inlet : bool (default True)
        True if an inlet is used in CHILD, false otherwise.
    is_meandering : bool (default True)
        True if the meandering mode is used in CHILD, false otherwise.
        
    """
    def __init__(self,
                 coarse_diameter,
                 fine_diameter,
                 coarse_proportion,
                 is_inlet=True,
                 is_meandering=True):

        self.coarse_diameter = coarse_diameter
        self.fine_diameter = fine_diameter
        self.coarse_proportion = coarse_proportion
        self.is_inlet = is_inlet
        self.is_meandering = is_meandering

        self.i_calls = 0
        self.nb_calls = 6
        if self.is_inlet == True:
            self.nb_calls += 2
        if self.is_meandering == True:
            self.nb_calls += 1

        self.parameter_values = dict()

    def get_value(self, value, random_state=None):

        if isinstance(value, (stats._distn_infrastructure.rv_frozen, MixtureModel, MemoryModel, DependencyModel)) == True:
            return value.rvs(random_state=random_state)
        return value

    def draw_parameters(self, random_state=None):

        self.parameter_values['GRAINDIAM1'] = self.get_value(self.coarse_diameter,
                                                             random_state=random_state)
        self.parameter_values['GRAINDIAM2'] = self.get_value(self.fine_diameter,
                                                             random_state=random_state)
        self.parameter_values['REGPROPORTION1'] = self.get_value(self.coarse_proportion,
                                                                 random_state=random_state)
        self.parameter_values['REGPROPORTION2'] = 1 - self.parameter_values['REGPROPORTION1']
        self.parameter_values['BRPROPORTION1'] = self.parameter_values['REGPROPORTION1']
        self.parameter_values['BRPROPORTION2'] = self.parameter_values['REGPROPORTION2']
        if self.is_inlet == True:
            self.parameter_values['INSEDLOAD1'] = self.parameter_values['REGPROPORTION1']
            self.parameter_values['INSEDLOAD2'] = self.parameter_values['REGPROPORTION2']
        if self.is_meandering == True:
            self.parameter_values['MEDIAN_DIAMETER'] = self.parameter_values['GRAINDIAM1']*self.parameter_values['REGPROPORTION1'] \
                                                       + self.parameter_values['GRAINDIAM2']*self.parameter_values['REGPROPORTION2']

    def rvs(self, parameter_name, random_state=None):

        if self.i_calls == 0:
            self.draw_parameters(random_state=random_state)

        self.i_calls += 1
        if self.i_calls == self.nb_calls:
            self.i_calls = 0

        return self.parameter_values[parameter_name]


class TimeSeriesConstraint:
    """
    Defines the constraints on one of CHILD's time series parameters. Not all
    constraints are meant to work together, e.g., value and rate are mutually
    exclusive. Not all configurations were thoroughly tested, so make sure to
    test that it does what you want before using this.
    
    Parameters
    ----------
    parameter : str
        Name of the parameter to constrain. Valid values are:
        'ST_PMEAN', 'ST_STDUR', 'ST_ISTDUR', 'UPRATE', 'FP_INLET_ELEVATION', 'FP_MU'
    initial_time : float
        Initial time of the time series.
    time_steps : float or scipy.stats' rv_continuous
        Time steps for the variations of the time series.
    final_time : float, optional (default None)
        Final time of the time series.
    initial_value : float, optional (default None)
        Initial value of the time series.
    rate : float or scipy.stats' rv_continuous, optional (default None)
        Rate of variation of the time series.
    final_value : float, optional (default None)
        Final value of the time series.
    minimal_value : float, optional (default None)
        Minimal value of the time series.
    value : float or scipy.stats' rv_continuous, optional (default None)
        Value of the time series.
    autocorr : float, optional (default None)
        Autocorrelation of the time series.
    mode : str (default 'interpolate')
        Interpolation mode of the time series in CHILD. 'interpolate' means
        linear variations between time steps, 'forward' means constant values
        between time steps.
        
    """
    def __init__(self,
                 parameter,
                 initial_time,
                 time_steps,
                 final_time=None,
                 initial_value=None,
                 rate=None,
                 final_value=None,
                 minimal_value=None,
                 value=None,
                 autocorr=None,
                 mode='interpolate'):
        
        valid_parameters = ['ST_PMEAN', 'ST_STDUR', 'ST_ISTDUR',
                            'UPRATE', 'FP_INLET_ELEVATION', 'FP_MU']
        if parameter not in valid_parameters:
            raise ValueError("Invalid time series parameter")
        if (value is None and (rate is None or initial_value is None)):
            raise ValueError("Invalid parameters: either use value, or rate with an initial value")
        if mode != 'interpolate' and mode != 'forward' and mode != '':
            raise ValueError("Invalid mode, should be interpolate or forward")

        self.parameter = parameter
        self.initial_time = initial_time
        self.time_steps = time_steps
        self.final_time = final_time
        self.rate = rate
        self.initial_value = initial_value
        self.final_value = final_value
        self.minimal_value = minimal_value
        self.value = value
        self.autocorr = autocorr
        self.mode = mode

class ConstrainedTimeSeries:
    """
    Model in which CHILD's time-series parameters are randomly drawn for one
    or several parameters depending on predefined constraints.
    
    Parameters
    ----------
    main_constraint : TimeSeriesConstraint
        Constraint on the main time-series parameter, which defines the final
        time for all the time series.
    other_constraints : list of TimeSeriesConstraint
        Constraints on the other time-series parameters.
    max_run_time : float (default np.inf)
        Extra constraint on the final time of the time series to make sure that
        simulation time cannot become an issue. Only useful when the main
        constraint doesn't define a final time.
    output_path : str (default '.')
        Path to the directory in which the time-series files are created.
        Default is the current directory.
    inline : bool (default False)
        If true, the time series are directly written in CHILD's input file.
        CHILD has line length limit, so it only works for short time series.
        If false, an extra file containing the time series is created, and
        only its path is written in CHILD's input file.
    set_out_intrvl : bool (default False)
        If true, set CHILD's OPINTRVL parameter to the final time of the time
        series, which means that CHILD is creating output files only once, at
        the very end of the simulation.
        
    """
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

        if isinstance(value, (stats._distn_infrastructure.rv_frozen, MixtureModel, MemoryModel, DependencyModel)) == True:
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
        while (((self.values[key][-1] != self.main_constraint.final_value
                 and self.times[key][-1] != self.main_constraint.final_time)
                or self.times[key][-1] > self.max_run_time
                or (self.main_constraint.minimal_value is not None
                    and any(x < self.main_constraint.minimal_value for x in self.values[key]) == True))
               and iter_count < max_iter):

            self.times[key] = [self.main_constraint.initial_time]
            initial_value = None
            previous_value = None
            if self.main_constraint.initial_value is None:
                initial_value = self.get_value(self.main_constraint.value,
                                               random_state=random_state)
                previous_value = self.main_constraint.value.cdf(initial_value)
                previous_value = stats.distributions.norm().ppf(previous_value)
            else:
                initial_value = self.get_value(self.main_constraint.initial_value,
                                               random_state=random_state)
            self.values[key] = [initial_value]
            while (self.values[key][-1] != self.main_constraint.final_value
                   and self.times[key][-1] != self.main_constraint.final_time
                   and self.times[key][-1] < self.max_run_time):

                time_step = self.get_value(self.main_constraint.time_steps,
                                           random_state=random_state)
                new_time = self.times[key][-1] + time_step
                
                new_value = None
                if self.main_constraint.rate is not None:
                    rate = self.get_value(self.main_constraint.rate,
                                          random_state=random_state)
                    if self.main_constraint.final_time is not None:
                        if new_time > self.main_constraint.final_time:
                            new_time = self.main_constraint.final_time
                            time_step = new_time - self.times[key][-1]
                        new_value = self.values[key][-1] + rate*time_step
                    elif self.main_constraint.final_value is not None:
                        new_value = self.values[key][-1] + rate*time_step
                        if new_value > self.main_constraint.final_value:
                            rate = (self.main_constraint.final_value
                                    - self.times[key][-1])/(new_value
                                                            - self.times[key][-1])
                            new_time = np.round(self.times[key][-1] + rate*time_step)
                            new_value = self.main_constraint.final_value
                else:
                    new_value = self.get_value(stats.distributions.norm(),
                                               random_state=random_state)
                    if C is not None:
                        new_value = np.dot(C, (previous_value, new_value))[1]
                        previous_value = new_value
                    new_value = stats.distributions.norm().cdf(new_value)
                    new_value = self.main_constraint.value.ppf(new_value)
                    if new_time > self.main_constraint.final_time:
                        new_value = self.values[key][-1] + (new_value - self.values[key][-1])*(self.main_constraint.final_time - self.times[key][-1])/(new_time - self.times[key][-1])
                        new_time = self.main_constraint.final_time

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
            while (((self.times[key][-1] != final_time))
                   and iter_count < max_iter):

                self.times[key] = [constraint.initial_time]
                initial_value = None
                previous_value = None
                if constraint.initial_value is None:
                    initial_value = self.get_value(constraint.value,
                                                   random_state=random_state)
                    previous_value = constraint.value.cdf(initial_value)
                    previous_value = stats.distributions.norm().ppf(previous_value)
                else:
                    initial_value = self.get_value(constraint.initial_value,
                                                   random_state=random_state)
                self.values[key] = [initial_value]
                while self.times[key][-1] < final_time:

                    time_step = self.get_value(constraint.time_steps,
                                               random_state=random_state)
                    new_time = self.times[key][-1] + time_step

                    new_value = None
                    if constraint.rate is not None:
                        rate = self.get_value(constraint.rate,
                                              random_state=random_state)
                        if new_time > final_time:
                            new_time = final_time
                            time_step = new_time - self.times[key][-1]
                        new_value = self.values[key][-1] + rate*time_step
                    else:
                        new_value = self.get_value(stats.distributions.norm(),
                                                   random_state=random_state)
                        if C is not None:
                            new_value = np.dot(C, (previous_value, new_value))[1]
                            previous_value = new_value
                        new_value = stats.distributions.norm().cdf(new_value)
                        new_value = constraint.value.ppf(new_value)
                        if new_time > final_time:
                            new_value = self.values[key][-1] + (new_value - self.values[key][-1])*(final_time - self.times[key][-1])/(new_time - self.times[key][-1])
                            new_time = final_time

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

def _build_sand_cmap(light_fraction_1,
                     light_fraction_2,
                     light_fraction_3,
                     light_fraction_4,
                     use_gold_sand=False,
                     reverse=False,
                     name='sand'):
    """
    Builds a colormap following a sandy color scheme.

    Parameters
    ----------
    light_fraction_1 : float
        Light fraction of the first color.
    light_fraction_2 : float
        Light fraction of the second color.
    light_fraction_3 : float
        Light fraction of the third color.
    light_fraction_4 : float
        Light fraction of the fourth color.
    use_gold_sand : bool (default False)
        If true, uses Gold Sand as third color, otherwise uses Yuma Sand.
    reverse : bool (default False)
        If true, reverses the color list.
    name : str (default 'sand')
        Name of the colormap.

    Returns
    -------
    cmap
        The colormap.

    """
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
    if reverse:
        color_list = color_list[::-1]
    
    return colors.LinearSegmentedColormap.from_list(name, color_list)


sand = _build_sand_cmap(-0.4, -0.18580532707557418, -0.47128079235588854, 0.0,
                        use_gold_sand=True,
                        name='sand')
sand_r = _build_sand_cmap(-0.4, -0.18580532707557418, -0.47128079235588854, 0.0,
                          use_gold_sand=True,
                          reverse=True,
                          name='sand_r')

sand_light = _build_sand_cmap(0.25, 0.08543330133998818, -0.4679754966612923, 0.0,
                              use_gold_sand=False,
                              name='sand_light')
sand_light_r = _build_sand_cmap(0.25, 0.08543330133998818, -0.4679754966612923, 0.0,
                                use_gold_sand=False,
                                reverse=True,
                                name='sand_light_r')

sand_extra_light = _build_sand_cmap(0.6, 0.24604133456286564, -0.4370096077612467, 0.0,
                                    use_gold_sand=False,
                                    name='sand_extra_light')
sand_extra_light_r = _build_sand_cmap(0.6, 0.24604133456286564, -0.4370096077612467, 0.0,
                                      use_gold_sand=False,
                                      reverse=True,
                                      name='sand_extra_light_r')
