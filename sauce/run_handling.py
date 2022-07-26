import pandas as pd
import tables as tb
import numpy as np
import numba as nb


@nb.njit
def global_event_sort(times, build_window):

    t_i = times[0]
    t_f = t_i + build_window
    event = 0
    event_number = np.empty(len(times))
    event_number[:] = np.nan

    for i in range(len(times)):
        tc = times[i]
        
        if tc >= t_i and tc < t_f:
            event_number[i] = event
        else:
            t_i = tc
            t_f = tc + build_window
            event += 1
            event_number[i] = event
    return event_number
    

class Run():

    
    """
    Testing shows it is faster to 
    load the whole h5 file and query.
    This object will be passed around so that each detector can 
    select the data it needs.
    """
    
    
    def __init__(self, h5_filename, mode='r'):

        self.df = pd.read_hdf(h5_filename, './raw_data/basic_info')
        self.is_sorted = False
        if np.any(self.df['is_trace']):
            with tb.open_file(h5_filename, 'r') as f: 
                traces = f.root.raw_data.trace_array

                # get array indices from df
                trace_idx = self.df[self.df['is_trace'] == True]['trace_idx']
                # traces are indexed from 1
                self.df['trace'] = [traces[i-1] if is_trace else np.nan
                                           for (i, is_trace) in
                                           zip(self.df['trace_idx'], self.df['is_trace'])]
        else:
            self.df['trace'] = np.nan
        

    def global_event_builder(self, build_window):
        """
        If you really want to, you can have a global event index.
        """

        build_window = np.abs(build_window)

        if not self.is_sorted:
            # sort if you haven't before
            self.df = self.df.sort_values(by='time_raw')
            self.df['event'] = np.nan
            self.is_sorted = True
        
        times = self.df['time_raw'].to_numpy()

        event_idx = global_event_sort(times, build_window)
        self.df['event'] = event_idx 
