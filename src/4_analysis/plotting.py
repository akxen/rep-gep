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
        self.analysis = AnalyseResults()

    @staticmethod
    def load_model_results(directory):
        """Load model results"""

        with open(os.path.join(directory, 'model_results.pickle'), 'rb') as f:
            model_results = pickle.load(f)

        return model_results


class CreatePlots:
    def __init__(self, results_dir, figures_dir):
        # Object used to get plot data
        self.plot_data = PlotData(results_dir)

        # Directory to place output figures
        self.figures_dir = figures_dir

    def get_surface_features(self, model_key, results_key, transition_year=None, price_target=None):
        """Get surface features for a given model"""

        # Results for a given model
        if model_key == 'ptar_diff':
            results = self.plot_data.results['ptar']
        else:
            results = self.plot_data.results[model_key]


        # Carbon prices
        if model_key in ['rep', 'tax']:
            x = list(results.keys())
            y = list(results[x[0]][results_key].keys())

        elif model_key in ['heuristic', 'mppdc', 'baudev', 'ptar', 'pdev', 'ptar_diff']:
            x = list(results[transition_year].keys())
            y = list(results[transition_year][x[0]][results_key].keys())

        else:
            raise Exception(f'Unexpected model_key: {model_key}')

        # Sort x and y values
        x.sort()
        y.sort()

        # Construct meshgrid
        X, Y = np.meshgrid(x, y)

        # Container for surface values
        Z = []
        for i in range(len(X)):
            for j in range(len(X[0])):
                try:
                    # Carbon price and year
                    x_cord, y_cord = X[i][j], Y[i][j]

                    # Lookup value corresponding to carbon price and year combination
                    if model_key in ['rep', 'tax']:
                        Z.append(results[x_cord][results_key][y_cord])

                    elif model_key in ['baudev', 'pdev', 'ptar']:
                        Z.append(results[transition_year][x_cord][results_key][y_cord])

                    elif model_key in ['ptar_diff']:
                        Z.append(results[transition_year][x_cord][results_key][y_cord] - price_target[y_cord])

                    else:
                        raise Exception(f'Unexpected model_key: {model_key}')

                except Exception as e:
                    print(e)
                    Z.append(-10)

        # Re-shape Z for plotting
        Z = np.array(Z).reshape(X.shape)

        return X, Y, Z

    def get_interpolated_surface(self, model_key, results_key, transition_year=None, price_target=None):
        """Construct interpolated surface based on model results"""

        # Original surface
        X, Y, Z = self.get_surface_features(model_key, results_key, transition_year=transition_year,
                                            price_target=price_target)

        # Interpolated grid
        Xi, Yi = np.meshgrid(np.arange(5, 100.1, 0.5), np.arange(2016, 2030.01, 0.1))

        # Interpolated surface
        Zi = interpolate.griddata((X.flatten(), Y.flatten()), Z.flatten(), (Xi, Yi), method='linear')

        return Xi, Yi, Zi

    def plot_tax_rep_comparison(self):
        """Compare price and emissions outcomes under a carbon tax and REP scheme"""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True)
        cmap = plt.get_cmap('plasma')

        # Extract surface information
        X1, Y1, Z1 = self.get_surface_features('tax', 'YEAR_AVERAGE_PRICE', transition_year=None)
        X1i, Y1i = np.meshgrid(np.arange(5, 100.1, 0.5), np.arange(2016, 2030.01, 0.1))
        Z1i = interpolate.griddata((X1.flatten(), Y1.flatten()), Z1.flatten(), (X1i, Y1i), method='linear')

        X2, Y2, Z2 = self.get_surface_features('rep', 'YEAR_AVERAGE_PRICE', transition_year=None)
        X2i, Y2i = np.meshgrid(np.arange(5, 100.1, 0.5), np.arange(2016, 2030.01, 0.1))
        Z2i = interpolate.griddata((X2.flatten(), Y2.flatten()), Z2.flatten(), (X2i, Y2i), method='linear')

        X3, Y3, Z3 = self.get_surface_features('tax', 'YEAR_EMISSIONS', transition_year=None)
        X3i, Y3i = np.meshgrid(np.arange(5, 100.1, 0.5), np.arange(2016, 2030.01, 0.1))
        Z3i = interpolate.griddata((X3.flatten(), Y3.flatten()), Z3.flatten(), (X3i, Y3i), method='linear')

        X4, Y4, Z4 = self.get_surface_features('rep', 'YEAR_EMISSIONS', transition_year=None)
        X4i, Y4i = np.meshgrid(np.arange(5, 100.1, 0.5), np.arange(2016, 2030.01, 0.1))
        Z4i = interpolate.griddata((X4.flatten(), Y4.flatten()), Z4.flatten(), (X4i, Y4i), method='linear')

        # Get min and max surface values across subplots on same row
        p1_vmin, p1_vmax = min(Z1i.min(), Z2i.min()), max(Z1i.max(), Z2i.max())
        p2_vmin, p2_vmax = min(Z3i.min(), Z4i.min()), max(Z3i.max(), Z4i.max())

        # Construct heatmaps
        im1 = ax1.pcolormesh(X1i, Y1i, Z1i, cmap=cmap, vmin=p1_vmin, vmax=p1_vmax, edgecolors='face')
        im2 = ax2.pcolormesh(X2i, Y2i, Z2i, cmap=cmap, vmin=p1_vmin, vmax=p1_vmax, edgecolors='face')
        im3 = ax3.pcolormesh(X3i, Y3i, Z3i, cmap=cmap, vmin=p2_vmin, vmax=p2_vmax, edgecolors='face')
        im4 = ax4.pcolormesh(X4i, Y4i, Z4i, cmap=cmap, vmin=p2_vmin, vmax=p2_vmax, edgecolors='face')

        # Add colour bars
        divider1 = make_axes_locatable(ax2)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        cb1 = fig.colorbar(im2, cax=cax1)
        cb1.set_label('Average price (\$/MWh)', fontsize=7)

        divider2 = make_axes_locatable(ax4)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        cb2 = fig.colorbar(im4, cax=cax2)
        cb2.set_label('Emissions price (tCO$_{2}$)', fontsize=7)

        cb2.formatter.set_powerlimits((6, 6))
        cb2.formatter.useMathText = True
        cb2.update_ticks()

        # Format axes
        ax1.yaxis.set_major_locator(MultipleLocator(3))
        ax1.yaxis.set_minor_locator(MultipleLocator(1))

        ax1.xaxis.set_major_locator(MultipleLocator(20))
        ax1.xaxis.set_minor_locator(MultipleLocator(5))

        ax3.set_xlabel('Emissions price (\$/tCO$_{2}$)', fontsize=7)
        ax4.set_xlabel('Emissions price (\$/tCO$_{2}$)', fontsize=7)

        ax1.set_ylabel('Year', fontsize=7)
        ax3.set_ylabel('Year', fontsize=7)

        ax1.set_title('Tax', fontsize=8, y=0.98)
        ax2.set_title('REP', fontsize=8, y=0.98)

        for a in [ax1, ax2, ax3, ax4]:
            a.tick_params(axis='both', which='major', labelsize=6)
            a.tick_params(axis='both', which='minor', labelsize=6)

        cb1.ax.tick_params(labelsize=6)
        cb2.ax.tick_params(labelsize=6)

        cb2.ax.yaxis.offsetText.set_fontsize(7)

        # Add text to denote subfigures
        text_style = {'verticalalignment': 'bottom', 'horizontalalignment': 'left', 'color': 'white', 'fontsize': 10,
                      'weight': 'bold'}
        ax1.text(7, 2016.5, 'a', **text_style)
        ax2.text(7, 2016.5, 'b', **text_style)
        ax3.text(7, 2016.5, 'c', **text_style)
        ax4.text(7, 2016.5, 'd', **text_style)

        fig.set_size_inches(6.5, 4)
        fig.subplots_adjust(left=0.09, bottom=0.10, right=0.92, top=0.95, wspace=0.1, hspace=0.16)

        # Save figure
        fig.savefig(os.path.join(self.figures_dir, 'tax_rep.pdf'))
        fig.savefig(os.path.join(self.figures_dir, 'tax_rep.png'), dpi=200)
        plt.show()

    def plot_transition_year_comparison(self, model_key):
        """Compare emissions, price, baseline, and scheme revenue outcomes under different transition years"""

        cmap = plt.get_cmap('plasma')
        fig, axs = plt.subplots(nrows=4, ncols=3, sharex=True, sharey=True)
        ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = axs

        layout = {'ax1': {'ax': ax1, 'transition_year': 2020, 'results_key': 'YEAR_EMISSIONS'},
                  'ax2': {'ax': ax2, 'transition_year': 2025, 'results_key': 'YEAR_EMISSIONS'},
                  'ax3': {'ax': ax3, 'transition_year': 2030, 'results_key': 'YEAR_EMISSIONS'},
                  'ax4': {'ax': ax4, 'transition_year': 2020, 'results_key': 'YEAR_AVERAGE_PRICE'},
                  'ax5': {'ax': ax5, 'transition_year': 2025, 'results_key': 'YEAR_AVERAGE_PRICE'},
                  'ax6': {'ax': ax6, 'transition_year': 2030, 'results_key': 'YEAR_AVERAGE_PRICE'},
                  'ax7': {'ax': ax7, 'transition_year': 2020, 'results_key': 'baseline'},
                  'ax8': {'ax': ax8, 'transition_year': 2025, 'results_key': 'baseline'},
                  'ax9': {'ax': ax9, 'transition_year': 2030, 'results_key': 'baseline'},
                  'ax10': {'ax': ax10, 'transition_year': 2020, 'results_key': 'YEAR_CUMULATIVE_SCHEME_REVENUE'},
                  'ax11': {'ax': ax11, 'transition_year': 2025, 'results_key': 'YEAR_CUMULATIVE_SCHEME_REVENUE'},
                  'ax12': {'ax': ax12, 'transition_year': 2030, 'results_key': 'YEAR_CUMULATIVE_SCHEME_REVENUE'},
                  }

        def get_limits(results_key):
            """Get vmin and vmax for colour bar"""

            # Initialise
            vmin, vmax = 1e11, -1e11

            for ty in [2020, 2025, 2030]:
                # Get surface features
                _, _, Z = self.get_interpolated_surface(model_key, results_key, transition_year=ty)

                if Z.min() < vmin:
                    vmin = Z.min()

                if Z.max() > vmax:
                    vmax = Z.max()

            return vmin, vmax

        def add_limits(layout_dict):
            """Add vmin and vmax for each plot"""

            for k in layout_dict.keys():
                # Get vmin and vmax
                vmin, vmax = get_limits(layout_dict[k]['results_key'])

                # Manually adjust some colour ranges
                if layout_dict[k]['results_key'] == 'baseline':
                    vmin, vmax = 0, 2

                elif layout_dict[k]['results_key'] == 'YEAR_AVERAGE_PRICE':
                    vmax = 120

                # Add to dictionary
                layout_dict[k]['vmin'] = vmin
                layout_dict[k]['vmax'] = vmax

            return layout_dict

        # Add vmin and vmax
        layout = add_limits(layout)

        for k, v in layout.items():
            X, Y, Z = self.get_interpolated_surface(model_key, v['results_key'], v['transition_year'])

            # Construct plot and keep track of it
            im = v['ax'].pcolormesh(X, Y, Z, cmap=cmap, vmin=v['vmin'], vmax=v['vmax'], edgecolors='face')
            layout[k]['im'] = im

        def add_dividers(layout_dict):
            """Add dividers so subplots are the the same size after adding colour bars"""

            for k in layout_dict.keys():
                divider = make_axes_locatable(layout_dict[k]['ax'])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cax.axis('off')
                layout_dict[k]['cax'] = cax

            return layout_dict

        # Add divider so subplots are the same size even when colour bars are included
        layout = add_dividers(layout)

        # Add colour bars
        layout['ax3']['cax'].axis('on')
        cb1 = fig.colorbar(layout['ax3']['im'], cax=layout['ax3']['cax'])
        cb1.ax.tick_params(labelsize=6)
        cb1.set_label('Emissions (tCO$_{2}$)', fontsize=7)
        cb1.formatter.set_powerlimits((6, 6))
        cb1.formatter.useMathText = True
        t1 = cb1.ax.yaxis.get_offset_text()
        t1.set_size(7)
        cb1.update_ticks()

        layout['ax6']['cax'].axis('on')
        cb2 = fig.colorbar(layout['ax6']['im'], cax=layout['ax6']['cax'])
        cb2.set_label('Price ($/MWh)', fontsize=7)
        cb2.ax.tick_params(labelsize=6)

        layout['ax9']['cax'].axis('on')
        cb3 = fig.colorbar(layout['ax9']['im'], cax=layout['ax9']['cax'])
        cb3.set_label('Baseline (tCO$_{2}$/MWh)', fontsize=7)
        cb3.ax.tick_params(labelsize=6)

        layout['ax12']['cax'].axis('on')
        cb4 = fig.colorbar(layout['ax12']['im'], cax=layout['ax12']['cax'])
        cb4.set_label('Revenue ($)', fontsize=7)
        cb4.ax.tick_params(labelsize=6)
        cb4.formatter.set_powerlimits((9, 9))
        cb4.formatter.useMathText = True
        t4 = cb4.ax.yaxis.get_offset_text()
        t4.set_size(7)
        cb4.update_ticks()

        # Format y-ticks and labels
        for a in ['ax1', 'ax4', 'ax7', 'ax10']:
            layout[a]['ax'].yaxis.set_major_locator(MultipleLocator(6))
            layout[a]['ax'].yaxis.set_minor_locator(MultipleLocator(2))

            layout[a]['ax'].tick_params(axis='both', which='major', labelsize=6)
            layout[a]['ax'].set_ylabel('Year', fontsize=7)

        # Format x-ticks
        for a in ['ax10', 'ax11', 'ax12']:
            layout[a]['ax'].xaxis.set_major_locator(MultipleLocator(20))
            layout[a]['ax'].xaxis.set_minor_locator(MultipleLocator(10))

            layout[a]['ax'].tick_params(axis='both', which='major', labelsize=6)
            layout[a]['ax'].set_xlabel('Emissions price (\$/tCO$_{2}$)', fontsize=7)

        # Add titles denoting transition years
        layout['ax1']['ax'].set_title('2020', fontsize=8, pad=2)
        layout['ax2']['ax'].set_title('2025', fontsize=8, pad=2)
        layout['ax3']['ax'].set_title('2030', fontsize=8, pad=2)

        # Add letters to differentiate plots
        text_x, text_y = 9, 2016.5
        text_style = {'verticalalignment': 'bottom', 'horizontalalignment': 'left', 'color': 'k', 'fontsize': 10,
                      'weight': 'bold'}
        ax1.text(text_x, text_y, 'a', )
        ax2.text(text_x, text_y, 'b', **text_style)
        ax3.text(text_x, text_y, 'c', **text_style)

        ax4.text(text_x, text_y, 'd', **text_style)
        ax5.text(text_x, text_y, 'e', **text_style)
        ax6.text(text_x, text_y, 'f', **text_style)

        ax7.text(text_x, text_y, 'g', **text_style)
        ax8.text(text_x, text_y, 'h', **text_style)
        ax9.text(text_x, text_y, 'i', **text_style)

        ax10.text(text_x, text_y, 'j', **text_style)
        ax11.text(text_x, text_y, 'k', **text_style)
        ax12.text(text_x, text_y, 'l', **text_style)

        # Format labels
        fig.set_size_inches(6.5, 6)
        fig.subplots_adjust(left=0.08, bottom=0.06, right=0.92, top=0.97, wspace=0.01)
        fig.savefig(os.path.join(figures_directory, f'transition_years_{model_key}.png'), dpi=200)
        fig.savefig(os.path.join(figures_directory, f'transition_years_{model_key}.pdf'))

        plt.show()

    def price_target_difference(self, price_target):
        """Plot difference between price target and realised prices"""

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)

        # Get interpolated surfaces
        X1, Y1, Z1 = self.get_interpolated_surface('ptar_diff', 'YEAR_AVERAGE_PRICE', 2020, price_target=price_target)
        X2, Y2, Z2 = self.get_interpolated_surface('ptar_diff', 'YEAR_AVERAGE_PRICE', 2025, price_target=price_target)
        X3, Y3, Z3 = self.get_interpolated_surface('ptar_diff', 'YEAR_AVERAGE_PRICE', 2030, price_target=price_target)

        # Construct plot and keep track of it
        plot_style = {'vmin': -10, 'vmax': 10, 'edgecolors': 'face'}
        im = ax1.pcolormesh(X1, Y1, Z1, **plot_style)
        im = ax2.pcolormesh(X2, Y2, Z2, **plot_style)
        im = ax3.pcolormesh(X3, Y3, Z3, **plot_style)

        plt.show()


if __name__ == '__main__':
    results_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '3_model', 'linear', 'output', 'remote')
    tmp_directory = os.path.join(os.path.dirname(__file__), 'output', 'tmp', 'remote')
    figures_directory = os.path.join(os.path.dirname(__file__), 'output', 'figures')

    # Object used to analyse results and get price target trajectory
    analysis = AnalyseResults()

    # Get plot data
    plot_data = PlotData(tmp_directory)
    plots = CreatePlots(tmp_directory, figures_directory)

    bau = analysis.load_results(results_directory, 'bau_case.pickle')
    bau_prices = analysis.get_year_average_price(bau['PRICES'], -1)
    bau_price_trajectory = bau_prices['average_price_real'].to_dict()
    bau_first_year_trajectory = {y: bau_price_trajectory[2016] for y in range(2016, 2031)}

    # plots.plot_tax_rep_comparison()
    # plots.plot_transition_year_comparison('baudev')
    # plots.plot_transition_year_comparison('ptar')
    # plots.plot_transition_year_comparison('pdev')
    plots.price_target_difference(bau_price_trajectory)

