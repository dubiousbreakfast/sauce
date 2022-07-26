from scipy import signal
import numpy as np
from scipy import interpolate
from scipy import optimize

"""
Filters for use in analyzing traces.

This will be fleshed out at some point
- Caleb Marshall, 2022 Ohio University

"""

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]
    
def delay_trace(trace, delay, fill_value=0.0):
    new_trace = np.empty_like(trace)
    if delay > 0:
        new_trace[:delay] = fill_value
        new_trace[delay:] = trace[:-delay]
    elif delay < 0:
        new_trace[delay:] = fill_value
        new_trace[:delay] = trace[-delay:]
    else:
        new_trace[:] = trace
    
    return new_trace

def baseline_correction(trace, baseline):
    avg = trace[:baseline].mean()
    return avg

def pole_zero(trace, decay_time):
    M = np.exp(-1.0/decay_time) # defined in terms of clock ticks
    w = np.ones(len(trace))
    Pz = (1-M)*signal.fftconvolve(trace, w, mode='full') 
    new_trace = trace + Pz[:len(trace)]
    return new_trace


def baseline_pole_zero(trace, decay_time, baseline):
    bl = baseline_correction(trace[:baseline])
    trace = trace - bl
    
    if decay_time > 0:
        trace = pole_zero(trace, decay_time)

    return trace

def trapezoid_filter(trace, length, gap):
    
    g = np.zeros(gap)
    l = np.ones(length)/length    
    h = np.concatenate((l, g, -l))
    y = signal.fftconvolve(trace, h, mode='full')  
    return y[:len(trace)]


def differential_filter(trace, shaping_time):
    window = np.ones(shaping_time)/shaping_time
    low_trace = signal.fftconvolve(trace, window, mode='full')[:len(trace)]
    trace = trace - low_trace
    return trace

def integral_filter(trace, shaping_time, num=4):

    window = np.ones(shaping_time)/shaping_time
    for i in range(num):
        trace = signal.fftconvolve(trace, window, mode='full')[:len(trace)] 

    trace = trace*3
    return trace


def semi_gaus_filter(trace, shaping_time):

    trace = differential_filter(trace, shaping_time)
    trace = integral_filter(trace, shaping_time)
    return trace

def constant_fraction_filter(trace, delay, fraction):
    
    attenuated = trace * fraction
    delayed = delay_trace(trace, delay)
    
    trace =  attenuated - delayed
    return trace


def zero_crossing_spline(trace, start, stop):
    # find points around zero crossing
    # Using cubic spline for interpolation

    # start and stop define the range to search for the zero crossing
    
    x = np.arange(len(trace))
    f = interpolate.CubicSpline(x, trace)
    crossings = f.roots()
    try:
        cross = np.where((crossings > start) & (crossings < stop))[0].min()
        return crossings[cross]
    except ValueError:
        return np.nan
    # cross = find_nearest(crossings, max_slope)

def pulse_shape(trace, fast_len, slow_len, rise_len=4):
    trace = trace - baseline(trace)
    height = np.argmax(trace)

    begin = height - rise_len
    # fast section
    total = trace[begin:height+fast_len+slow_len].sum()
    start_of_fast = int(begin + rise_len/2)
    fast = trace[start_of_fast:height+fast_len].sum()
    # slow section
    slow = trace[height+fast_len:height+fast_len+slow_len].sum()
    return total, fast, slow

def zero_crossing_threshold(trace, threshold):
    """
    sort of like the cfd for the pixie-16 cards, but different
    This kinda sucks because the threshold is applied to the attenuated 
    signal.
    """
    x = np.arange(len(trace))
    f = interpolate.CubicSpline(x, trace)

    crossings = f.roots()

    # 1) find first value above threshold
    above_thres = np.where(trace > threshold)[0][0]
    # 2) find first value < 0
    cross = crossings[crossings > above_thres][0]
    
    return cross


class Filter():

    def __init__(self):
        self.filter_lst = []
        self.filter_args = []
        self.filter_kwargs = []
        
    def add_filter(self, filter_func, *args, **kwargs):
        self.filter_lst.append(filter_func)
        self.filter_args.append(args)
        self.filter_kwargs.append(kwargs)

    def remove_filter(self, pos=-1):
        del self.filter_lst[pos]
        del self.filter_args[pos]
        del self.filter_kwargs[pos]

    def apply(self, trace):
        # apply the filters in succession
        result = np.copy(trace)
        for func, args in zip(self.filter_lst, self.filter_args):
            result = func(result, *args)
        return result

    
    def __call__(self, trace):
        return self.apply(trace)
