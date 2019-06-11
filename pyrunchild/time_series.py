################################################################################
# Imports
################################################################################

import os
import numpy as np
from scipy import stats

################################################################################
# MixtureModel
################################################################################

class MixtureModel(object):
    
    def __init__(self, submodels, weights=None, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        self.weights = weights

    def rvs(self, size=1, random_state=None):
        
        submodels_samples = np.zeros((size, len(self.submodels)))
        for i, submodel in enumerate(self.submodels):
            submodels_samples[:, i] = submodel.rvs(size=size,
                                                   random_state=random_state)
        rand_indices = np.random.choice(np.arange(len(self.submodels)),
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
    
    def __init__(self, model, call_memory=1, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
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
# Linear time series
################################################################################

class LinearTimeSeries(object):

    def __init__(self,
                 nb_steps,
                 rates,
                 initially_positive,
                 initial_value,
                 initial_time,
                 final_time,
                 seed=None):

        if seed is not None:
            np.random.seed(seed)

        self.nb_steps = nb_steps
        self.rates = rates
        self.initially_positive = initially_positive
        self.initial_value = initial_value
        self.initial_time = initial_time
        self.final_time = final_time

    def get_value(self, value):

        if isinstance(value, stats._distn_infrastructure.rv_frozen) == True:
            return value.rvs()
        return value

    def build_time_series(self,
                          nb_steps,
                          rates,
                          initial_value,
                          initial_time,
                          final_time):

        times = np.random.rand(nb_steps)
        times /= np.sum(times)
        times *= final_time - initial_time
        times += initial_time
        cum_times = np.cumsum(times)

        new_value = initial_value
        time_series = '@inline ' + str(initial_time) + ':' + str(new_value) + ' '
        for time, cum_time, rate in zip(times, cum_times, rates):
            new_value += rate*time
            time_series += str(cum_time) + ':' + str(new_value) + ' '
        time_series += 'interpolate'

        return time_series

    def write(self):

        nb_steps = self.get_value(self.nb_steps)
        initial_value = self.get_value(self.initial_value)
        initial_time = self.get_value(self.initial_time)
        final_time = self.get_value(self.final_time)

        initially_positive = self.initially_positive
        if isinstance(self.initially_positive, float) == True:
            initially_positive = stats.bernoulli(self.initially_positive).rvs()
        rates = self.rates
        if isinstance(self.rates, stats._distn_infrastructure.rv_frozen) == True:
            rates = [self.get_value(self.rates) for step in range(nb_steps)]
        sign = 1
        if initially_positive == False:
            sign = -1
        for i, rate in enumerate(rates):
            rates[i] = sign*rate
            sign *= -1

        return self.build_time_series(nb_steps,
                                      rates,
                                      initial_value,
                                      initial_time,
                                      final_time)

################################################################################
# FloodplainTimeSeries
################################################################################

class FloodplainTimeSeries(object):

    def __init__(self,
                 initial_time,
                 initial_inlet_elevation,
                 final_inlet_elevation,
                 time_steps,
                 inlet_elevation_rate,
                 max_run_time=np.inf,
                 other_parameters=None,
                 output_path='.',
                 inline=False,
                 set_out_intrvl=False,
                 seed=None):

        self.initial_time = initial_time
        self.time_steps = time_steps
        self.initial_inlet_elevation = initial_inlet_elevation
        self.final_inlet_elevation = final_inlet_elevation
        self.inlet_elevation_rate = inlet_elevation_rate
        self.max_run_time = max_run_time
        self.other_parameters = other_parameters
        self.output_path = os.path.abspath(output_path)
        self.inline = inline
        self.set_out_intrvl = set_out_intrvl
        if seed is not None:
            np.random.seed(seed)

        self.parameter_values = dict()
        self.parameter_values['RUNTIME'] = None
        self.parameter_values['OPINTRVL'] = None
        self.parameter_values['ST_PMEAN'] = None
        self.parameter_values['ST_STDUR'] = None
        self.parameter_values['ST_ISTDUR'] = None
        self.parameter_values['UPRATE'] = None
        self.parameter_values['FP_INLET_ELEVATION'] = None
        self.parameter_values['FP_MU'] = None

        self.is_called = dict()
        self.is_called['RUNTIME'] = None
        self.is_called['OPINTRVL'] = None
        self.is_called['FP_INLET_ELEVATION'] = None
        if other_parameters is not None:
            for key in other_parameters:
                self.is_called[key] = None

    def get_value(self, value):

        if isinstance(value, (stats._distn_infrastructure.rv_frozen, MixtureModel)) == True:
            return value.rvs()
        return value

    def build_varying_elevation(self, max_iter):

        self.times = [np.inf]
        iter_count = 0
        while self.times[-1] >= self.max_run_time and iter_count < max_iter:

            self.times = [self.initial_time]
            self.inlet_elevations = [self.initial_inlet_elevation]
            while self.inlet_elevations[-1] != self.final_inlet_elevation:

                time_step = self.get_value(self.time_steps)
                new_time = self.times[-1] + time_step
                inlet_elevation_rate = self.get_value(self.inlet_elevation_rate)
                new_inlet_elevation = self.inlet_elevations[-1] + inlet_elevation_rate*time_step
                if new_inlet_elevation > self.final_inlet_elevation:
                    rate = (self.final_inlet_elevation
                            - self.times[-1])/(new_inlet_elevation
                                               - self.times[-1])
                    new_time = np.round(self.times[-1] + rate*time_step)
                    new_inlet_elevation = self.final_inlet_elevation

                self.times.append(new_time)
                self.inlet_elevations.append(new_inlet_elevation)
                
            iter_count += 1

    def build_time_series(self, base_name=None):

        time_series = ''
        if self.inline == True:
            time_series = '@inline '
            for time, elevation in zip(self.times, self.inlet_elevations):
                time_series += str(time) + ':' + str(elevation) + ' '
            time_series += 'interpolate'
        else:
            file_name = 'FP_INLET_ELEVATION.dat'
            if base_name is not None:
                file_name = base_name + '_' + file_name
            with open(os.path.join(self.output_path, file_name), 'w') as file:
                for time, elevation in zip(self.times, self.inlet_elevations):
                    file.write(str(time) + ' ' + str(elevation) + '\n')
            time_series = '@file ' + file_name + ' 1 2 interpolate'

        self.parameter_values['RUNTIME'] = str(self.times[-1])
        self.is_called['RUNTIME'] = False
        if self.set_out_intrvl == True:
            self.parameter_values['OPINTRVL'] = self.parameter_values['RUNTIME']
            self.is_called['OPINTRVL'] = False
        self.parameter_values['FP_INLET_ELEVATION'] = time_series
        self.is_called['FP_INLET_ELEVATION'] = False
        
    def build_other_time_series(self):

        if self.other_parameters is not None:
            for key in self.other_parameters:
                if key in self.parameter_values:
                    time_series = ''
                    if self.inline == True:
                        time_series = '@inline '
                        for time in self.times:
                            parameter_value = self.get_value(self.other_parameters[key][0])
                            time_series += str(time) + ':' + str(parameter_value) + ' '
                        time_series += self.other_parameters[key][1]
                    else:
                        file_name = key + '.dat'
                        if base_name is not None:
                            file_name = base_name + '_' + file_name
                        with open(os.path.join(self.output_path, file_name), 'w') as file:
                            for time in self.times:
                                parameter_value = self.get_value(self.other_parameters[key][0])
                                file.write(str(time) + ' ' + str(parameter_value) + '\n')
                        time_series = '@file ' + file_name + ' 1 2 interpolate'

                    self.parameter_values[key] = time_series
                    self.is_called[key] = False
                else:
                    print(key, 'is not a time-varying parameter')

    def write(self, parameter_name, base_name=None, max_iter=1e6):

        if self.is_called['RUNTIME'] is None:
            self.build_varying_elevation(max_iter)
            self.build_time_series(base_name=base_name)
            self.build_other_time_series()

        self.is_called[parameter_name] = True
        if all(i == True for i in self.is_called.values()):
            for key in self.is_called:
                self.is_called[key] = None

        return self.parameter_values[parameter_name]
