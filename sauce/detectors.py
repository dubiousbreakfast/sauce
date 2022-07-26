
"""
This file helps map the data to actual detectors and group them.

-Caleb Marshall, Ohio University 2022
"""

import numpy as np
import pandas as pd
import tables as tb
from matplotlib.path import Path
from .run_handling import Run

def smap(f, *args):
    # convenience function for multiprocessing
    return f(*args)

class Detector():

    """
    Class to hold data relevant to the specific channel. 
    """

    def __init__(self, crate, slot, channel, name):

        self.crate = crate
        self.slot = slot
        self.channel = channel
        self.name = name
        self.data = None
        
    def find_events(self, full_run_data):

        """
        After more usage, I think it is useful to either
        load the entire run (detailed analysis) or 
        pull from disk just the specific data. 
        
        As such this function is now more general, and
        calls two other methods to select the data depending
        on whether a Run object is passed or a path to an h5 file.
        """

        if isinstance(full_run_data, Run):
            self._events_from_run(full_run_data)
        elif isinstance(full_run_data, str):
            self._events_from_h5(full_run_data)
        else:
            print('Only Run objects or h5_file paths accepted!')
            

    def _events_from_run(self, run_obj):
        df = run_obj.df

        # pull the relevant data          
        self.data = df.loc[(df['crate'] == self.crate) &
                           (df['slot'] == self.slot) &
                           (df['channel'] == self.channel)]

                    
        # Drop all of the columns that are not needed anymore
        self.data = self.data.drop(columns=['crate', 'slot', 'channel', 'trace_idx', 'is_trace']) \
                             .reset_index(drop=True) \
                             .sort_values(by='time_raw')
        

    def _events_from_h5(self, h5_filename):
        with tb.open_file(h5_filename, 'r') as f:

            # string for the query 
            where_str = ("( crate == " + str(self.crate) +
                         ") & ( slot == " + str(self.slot) +
                         ") & ( channel == " + str(self.channel) + ")")

            table = f.root.raw_data.basic_info
            traces = f.root.raw_data.trace_array
            det_iter = table.where(where_str)

            # list of tuples into dataframe          
            self.data = pd.DataFrame.from_records([x.fetch_all_fields() for x in det_iter],
                                                  columns=table.colnames)

            if np.any(self.data['is_trace']):
                # traces are indexed from 1
                self.data['trace'] = [traces[i-1] if is_trace else np.nan
                                      for (i, is_trace) in
                                      zip(self.data['trace_idx'], self.data['is_trace'])]
            else:
                self.data['trace'] = np.nan
            
        # Drop all of the columns that are not needed anymore and sort
        self.data = self.data.drop(columns=['crate', 'slot', 'channel', 'trace_idx', 'is_trace']) \
                             .reset_index(drop=True) \
                             .sort_values(by='time_raw')

    def energy_calibrate(self, calibration_function):
        self.data['energy'] = calibration_function(self.data['energy'])
        
    def time_calibrate(self, calibration_function):
        self.data['time'] = calibration_function(self.data['time'])

    def apply_threshold(self, threshold, axis='energy'):
        self.data = self.data.loc[self.data['energy'] > threshold]

    def apply_cut(self, cut, axis='energy'):
        self.data = self.data.loc[(self.data[axis] > cut[0]) &
                                  (self.data[axis] < cut[1])]

    def apply_poly_cut(self, cut2d, gate_name=None):
        """
        Apply a 2D polygon cut to the data. Gate info is
        found in Cut2D object found in sauce.gates
        """
        points = cut2d.points
        x_axis = cut2d.x_axis
        y_axis = cut2d.y_axis
        
        poly = Path(points, closed=True)
        results = poly.contains_points(self.data[[x_axis, y_axis]])
        self.data = self.data[results]
        

    def hist(self, lower, upper, bins, axis='energy', centers=True):
        """
        Return a histrogram of the given axis.
        """

        counts, bin_edges = np.histogram(self.data[axis], bins=bins, range=(lower, upper))
        # to make fitting data
        if centers:
            centers = bin_edges[:-1]
            return centers, counts
        else:
            return counts, bin_edges



class DSSD(Detector):

    def __init__(self, map_filename, side, map_seperator='\s+'):
        """
        Creates dictonary of Detector objects that correspond to 
        the one side of the DSSD. map_filename provides the 
        channel ids
        """

        self.data = None
        self.side = side
        self.dssd_dic = {}
        self.name = 'dssd_'+side
        
        temp_file = pd.read_csv(map_filename, sep=map_seperator)
        for index, row in temp_file.iterrows():
            if row['side'] == side:
                det_name = str(row['side']) + ' ' + str(row['strip'])
                self.dssd_dic[row['strip']] = Detector(row['crate'],
                                                       row['slot'],
                                                       row['channel'],
                                                       det_name)

    def find_events(self, full_run_data):

        print('Finding ' + self.name + ' events')

        # assign the data, results from map are in original order 
        for k, v in self.dssd_dic.items():
            v.find_events(full_run_data)
            v.data['strip'] = int(k)
        self._make_data()
            
    def _make_data(self):
        # Make sure you clear the data frame first
        self.data = None
        
        frame = [x.data for i, x in self.dssd_dic.items()]
        self.data = pd.concat(frame, ignore_index=True) \
                      .sort_values(by='time_raw')
                      
