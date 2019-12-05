"""Plotting results from REP and price targeting models"""

import os
import pickle

import numpy as np
from scipy import interpolate
from scipy.interpolate import griddata
from matplotlib.colors import BoundaryNorm

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

from analysis import AnalyseResults


class PlotData:
    def __init__(self, results_dir):
        self.results = self.load_model_results(results_dir)

    @staticmethod
    def load_model_results(directory):
        """Load model results"""

        with open(os.path.join(directory, 'model_results.pickle'), 'rb') as f:
            model_results = pickle.load(f)

        return model_results


class CreatePlots:
    def __init__(self, results_dir):
        # Object used to get plot data
        self.plot_data = PlotData(results_dir)


if __name__ == '__main__':
    results_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '3_model', 'linear', 'output', 'remote')
    tmp_directory = os.path.join(os.path.dirname(__file__), 'output', 'tmp', 'remote')
    figures_directory = os.path.join(os.path.dirname(__file__), 'output', 'figures')

    # Get plot data
    plot_data = PlotData(tmp_directory)
