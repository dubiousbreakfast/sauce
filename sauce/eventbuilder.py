import pandas as pd
import numpy as np
import numba as nb
from . import detectors

"""
The philosophy of SAUCE is to build meaningful correlations between 
detector events. This is done by mimicking the way a traditional ADC gate
would work. In other words:

1) Define the detectors that will "open" the gate
2) Look for all coincidences within a given dT.

What we don't want are arbitrary build windows based on higher rate detectors.
  
"""


def get_closest(array, values):
    """
    I stole this from https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array

    Note, arrays should be sorted!
    """

    
    # make sure array is a numpy array
    array = np.array(array)

    # get insert positions
    idxs = np.searchsorted(array, values, side="left")
    
    # find indexes where previous index is closer
    prev_idx_is_less = ((idxs == len(array))|(np.fabs(values - array[np.maximum(idxs-1, 0)])
                                              < np.fabs(values - array[np.minimum(idxs, len(array)-1)])))
    idxs[prev_idx_is_less] -= 1
    
    return idxs

@nb.njit
def reduce_intervals(low, high):
    """
    bisgetti code
    
    Help create arrays that define 
    closed, disjoint intervals.

    Index i takes us through the high values
    Index j takes us through the low values
        
    """
    reject = []

    i = 0
    j = 1
    
    while i < len(high) and j < len(low):

        h = high[i]
        l = low[j]
        if l <= h:
            reject.append(j)
            j += 1
        elif l > h:
            i = j
            j += 1
        
    return reject    


@nb.njit
def find_coincident_events(A, B, C):
    """
    taken from: https://stackoverflow.com/questions/43382056/detect-if-elements-are-within-pairs-of-interval-limits

    Find element of A and B such that C is C>=A and C<=B
    """
    
    # Use searchsorted and look for     
    m_AB = np.searchsorted(C, A, 'left') != np.searchsorted(C, B, 'right') # masks for the timestamp array
    return m_AB

@nb.njit
def assign_event(hit_index, lower, upper, data):
    event_number = np.empty(len(data))
    event_number[:] = np.nan
    start_index = 0
    for i in range(len(hit_index)):
        hit_num = hit_index[i]
        l = lower[hit_num]
        h = upper[hit_num]
        for j in range(start_index, len(data)):
            if l <= data[j] <= h:
                event_number[j] = hit_num
            elif data[j] < l:
                event_number[j] = np.nan
            elif data[j] > h:                
                start_index = j
                break
    return event_number


class EventBuilder():

    """
    Construct a builder that takes a 
    low and high time value, constructs intervals,
    removes overlapping intervals, then filters
    other data based on these intervals.

    Build a time stamp array by adding detector times
    with self.add_timestamps. Eventbuilder will make
    sure this array is time ordered.

    After all desired detectors have been added
    disjoint build windows are created using
    self.create_build_windows(low, high)

    low and high are the time before and after
    these events that are considered for correlations.
    They are in units of nanoseconds.
    
    """


    def __init__(self):
        self.lower = None
        self.upper = None
        self.pre_reduced_len = 0.0
        self.reduced_len = 0.0
        self.timestamps = []

    def add_timestamps(self, timestamps):
        timestamps = timestamps.to_numpy()
        self.timestamps = np.concatenate((self.timestamps, timestamps))
        # make sure it is sorted
        self.timestamps = np.sort(self.timestamps)
        
    def create_build_windows(self, low, high):

        # just in case you include a negative
        low = np.abs(low)
        
        low_stamps = self.timestamps - low
        high_stamps = self.timestamps + high

        self.pre_reduced_len = len(self.timestamps)
        
        drop_indx = reduce_intervals(low_stamps, high_stamps)

        low_stamps = np.delete(low_stamps, drop_indx)
        high_stamps = np.delete(high_stamps, drop_indx)
                    
        # combine (if there was any data before) and sort
        self.lower = low_stamps
        self.upper = high_stamps

        self.reduced_len = len(self.lower)
        
        self.calc_livetime()
        


    def calc_livetime(self):
        self.livetime = self.reduced_len/self.pre_reduced_len
        

    def filter_data(self, det):
        """
        For the given detector, look at each event and see if it can be assigned
        to an event based on the time stamp array.
        """        
        
        det_times = det.data['time_raw'].to_numpy()
        
        # get indices of events that contain at least one of the detectors time stamps
        mask = find_coincident_events(self.lower, self.upper, det_times)
        hit_index = np.arange(len(self.lower))[mask]

        # assign event numbers to every event (if no event give NaN)
        event_number = assign_event(hit_index, self.lower, self.upper, det_times)
        det.data['event'] = event_number

        # drop duplicate events keep first timestamp
        det.data = det.data.drop_duplicates('event', keep='first')
        
        return det


    
def same_event(det1, det2):

    """
    def same_event(df1, df2):

    df1: dataframe with the hits of interest
    df2: dataframe to correlate with df1

    sorting is based on event number, defined by the event builder
    time window. In theory function can be applied as many times as
    needed/desired.

    returns a new detector instance 
    """
    # These errors are if we have already called the function once
    # If that is the case, then the column names will not exist
    try:
        df1 = det1.data.rename(columns={'energy': 'energy'+'_'+det1.name,
                                        'time': 'time'+ '_'+det1.name}).copy()
    except KeyError:
        df1 = det1.data
        
    df1 = df1.dropna(subset=['event'])
    
    try:
        df2 = det2.data.rename(columns={'energy': 'energy'+'_'+det2.name,
                                              'time': 'time'+'_'+det2.name})
    except KeyError:
        df2 = det2.data

    df2 = df2.dropna(subset=['event'])

    df3 = pd.merge(df1, df2, on='event', suffixes=('_'+det1.name, '_'+det2.name))
    new_det = detectors.Detector(0, 0, 0, det1.name +'_' +det2.name)
    new_det.data = df3
    return new_det

