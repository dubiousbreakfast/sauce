import numpy as np
import pandas as pd
from matplotlib.path import Path
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from . import detectors


class Cut2D():

    def __init__(self, x_axis, y_axis):
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.points = []
    

class CreateGate2D(Cut2D):

    def __init__(self, det, x_axis, y_axis, xy_range=None):
        Cut2D.__init__(self, x_axis, y_axis)
        x = det.data[x_axis]
        y = det.data[y_axis]
        self.fig, self.ax = plt.subplots()
        if xy_range:
            self.ax.hist2d(x, y, bins=[1024, 1024], cmin=1,
                           range=[[0, xy_range[0]], [0, xy_range[1]]])
        else:
            self.ax.hist2d(x, y, bins=[1024, 1024], cmin=1,
                           range=[[0, x.max()], [0, y.max()+100]])

        self.ax.set_title('Click to set gate, press enter to finish')
        self.cid = plt.connect('button_press_event', self.on_click)
        self.cid2  = plt.connect('key_press_event', self.on_press)
        plt.show()

    def on_click(self, event):
        x, y = event.xdata, event.ydata
        if event.inaxes:
            # add a point on left click
            if event.button == 1:
                print(x, y)
                self.points.append((x, y))
                self.drawing_logic()
                plt.draw()
            elif event.button == 3:
                self.points.pop()
                self.drawing_logic()
                print('Deleting last point')
                plt.draw()

    def on_press(self, event):
        if event.key == 'enter':
            plt.disconnect(self.cid)
            self.points.append(self.points[0])
            self.patch_update(closed=True, facecolor='r', alpha=0.2)
            plt.draw()
            print(self.points)
            return self.points
                
    def patch_update(self, closed=False, facecolor='none', alpha=1.0):
        self.ax.patches = []                 
        path = Path(self.points, closed=closed)
        patch = patches.PathPatch(path, facecolor=facecolor,
                                  alpha=alpha)      
        self.ax.add_patch(patch)             

    def drawing_logic(self):
        if len(self.points) == 1:
            self.ax.scatter(self.points[0][0], self.points[0][1])
        elif len(self.points) >= 2:
            self.patch_update()
        else:
            pass
