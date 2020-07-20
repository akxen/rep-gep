"""Plotting results from REP and price targeting models"""

import os
import pickle

import numpy as np
import pandas as pd
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

    @staticmethod
    def cm_to_in(cm):
        """Convert cm to inches"""
        return cm / 2.54

    def get_surface_features(self, model_key, results_key, transition_year=None, price_target=None):
        """Get surface features for a given model"""

        # Results for a given model
        if model_key == 'ptar_diff':
            results = self.plot_data.results['ptar']

        elif model_key == 'baudev_diff':
            results = self.plot_data.results['baudev']

        else:
            results = self.plot_data.results[model_key]

        # Carbon prices
        if model_key in ['rep', 'tax']:
            x = list(results.keys())
            y = list(results[x[0]][results_key].keys())

        elif model_key in ['heuristic', 'mppdc', 'baudev', 'ptar', 'pdev', 'ptar_diff', 'baudev_diff']:
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

                    elif model_key in ['ptar_diff', 'baudev_diff']:
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
        cb2.set_label('Emissions (tCO$_{2}$)', fontsize=7)

        cb2.formatter.set_powerlimits((6, 6))
        cb2.formatter.useMathText = True
        cb2.update_ticks()

        # Format axes
        ax1.yaxis.set_major_locator(MultipleLocator(3))
        ax1.yaxis.set_minor_locator(MultipleLocator(1))

        ax1.xaxis.set_major_locator(MultipleLocator(20))
        ax1.xaxis.set_minor_locator(MultipleLocator(5))

        ax3.set_xlabel('Emissions price (\$/tCO$_{2}$)')
        ax4.set_xlabel('Emissions price (\$/tCO$_{2}$)')

        ax1.set_ylabel('Year')
        ax3.set_ylabel('Year')

        for a in [ax1, ax2, ax3, ax4]:
            a.tick_params(axis='both', which='major', labelsize=6)
            a.tick_params(axis='both', which='minor', labelsize=6)

        # Set font size
        ax1.set_title('Tax', fontsize=7, y=0.97)
        ax2.set_title('REP', fontsize=7, y=0.97)

        cb1.ax.tick_params(labelsize=6)
        cb2.ax.tick_params(labelsize=6)

        cb2.ax.yaxis.offsetText.set_fontsize(7)

        ax1.xaxis.label.set_size(7)
        ax1.yaxis.label.set_size(7)

        ax2.xaxis.label.set_size(7)
        ax2.yaxis.label.set_size(7)

        ax3.xaxis.label.set_size(7)
        ax3.yaxis.label.set_size(7)

        ax4.xaxis.label.set_size(7)
        ax4.yaxis.label.set_size(7)

        # Add text to denote subfigures
        text_style = {'verticalalignment': 'bottom', 'horizontalalignment': 'left', 'fontsize': 8, 'weight': 'bold'}
        ax1.text(7, 2016.25, 'a', color='white', **text_style)
        ax2.text(7, 2016.25, 'b', color='white', **text_style)
        ax3.text(7, 2016.25, 'c', color='k', **text_style)
        ax4.text(7, 2016.25, 'd', color='k', **text_style)

        fig.set_size_inches(self.cm_to_in(11.5), self.cm_to_in(7.6))
        fig.subplots_adjust(left=0.11, bottom=0.12, right=0.90, top=0.95, wspace=0.1, hspace=0.16)

        # Save figure
        fig.savefig(os.path.join(self.figures_dir, 'defense', 'tax_rep.pdf'), transparent=True)
        fig.savefig(os.path.join(self.figures_dir, 'defense', 'tax_rep.png'), transparent=True, dpi=300)
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

                # Manually adjust min an max for some colour ranges
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
        cb1 = fig.colorbar(layout['ax3']['im'], cax=layout['ax3']['cax'], ticks=MultipleLocator(40e6))
        cb1.ax.tick_params(labelsize=5)
        # cb1.set_label('Emissions (tCO$_{2}$)', fontsize=5)
        cb1.formatter.set_powerlimits((6, 6))
        cb1.formatter.useMathText = True
        t1 = cb1.ax.yaxis.get_offset_text()
        t1.set_size(5.5)
        layout['ax3']['cax'].yaxis.get_offset_text().set_visible(False)
        layout['ax3']['ax'].text(95, 2030.2, r"$\times 10^{6}$", fontsize=5.5)

        cb1.update_ticks()

        layout['ax6']['cax'].axis('on')
        cb2 = fig.colorbar(layout['ax6']['im'], cax=layout['ax6']['cax'])
        # cb2.set_label('Price ($/MWh)', fontsize=5)
        cb2.ax.tick_params(labelsize=5)

        layout['ax9']['cax'].axis('on')
        cb3 = fig.colorbar(layout['ax9']['im'], cax=layout['ax9']['cax'])
        # cb3.set_label('Baseline (tCO$_{2}$/MWh)', fontsize=5)
        cb3.ax.tick_params(labelsize=5)

        layout['ax12']['cax'].axis('on')
        cb4 = fig.colorbar(layout['ax12']['im'], cax=layout['ax12']['cax'])
        # cb4.set_label('Revenue ($)', fontsize=5)
        cb4.ax.tick_params(labelsize=5)
        cb4.formatter.set_powerlimits((9, 9))
        cb4.formatter.useMathText = True
        t4 = cb4.ax.yaxis.get_offset_text()
        t4.set_size(5.5)
        layout['ax12']['cax'].yaxis.get_offset_text().set_visible(False)
        layout['ax12']['ax'].text(95, 2030.2, r"$\times 10^{9}$", fontsize=5.5)
        cb4.update_ticks()

        # Set y-lim for all plots
        for a in [f'ax{i}' for i in range(1, 13)]:
            layout[a]['ax'].set_ylim([2016, 2030])

        # Add lines to denote transition years and set y-lim
        for a in ['ax1', 'ax4', 'ax7', 'ax10']:
            layout[a]['ax'].plot([5, 100], [2020, 2020], color='w', linestyle='--', linewidth=0.8)

        for a in ['ax2', 'ax5', 'ax8', 'ax11']:
            layout[a]['ax'].plot([5, 100], [2025, 2025], color='w', linestyle='--', linewidth=0.8)

        for a in ['ax3', 'ax6']:
            layout[a]['ax'].plot([5, 100], [2029.9, 2029.9], color='w', linestyle='--', linewidth=0.8)

        for a in ['ax9', 'ax12']:
            layout[a]['ax'].plot([5, 100], [2029.85, 2029.85], color='w', linestyle='--', linewidth=0.8)

        # Format y-ticks and labels
        for a in ['ax1', 'ax4', 'ax7', 'ax10']:
            layout[a]['ax'].yaxis.set_major_locator(MultipleLocator(6))
            layout[a]['ax'].yaxis.set_minor_locator(MultipleLocator(2))

            layout[a]['ax'].tick_params(axis='both', which='major', labelsize=5)
            layout[a]['ax'].set_ylabel('Year', fontsize=6, labelpad=-.1)

        # Format x-ticks
        for a in ['ax10', 'ax11', 'ax12']:
            layout[a]['ax'].xaxis.set_major_locator(MultipleLocator(20))
            layout[a]['ax'].xaxis.set_minor_locator(MultipleLocator(10))

            layout[a]['ax'].tick_params(axis='both', which='major', labelsize=5)
            layout[a]['ax'].set_xlabel('Emissions price (\$/tCO$_{2}$)', fontsize=6, labelpad=-0.01)

        # Add titles denoting transition years
        layout['ax1']['ax'].set_title('2020', fontsize=7, pad=2)
        layout['ax2']['ax'].set_title('2025', fontsize=7, pad=2)
        layout['ax3']['ax'].set_title('2030', fontsize=7, pad=2)

        # Add letters to differentiate plots
        text_x, text_y = 7.5, 2016.5
        text_style = {'verticalalignment': 'bottom', 'horizontalalignment': 'left', 'fontsize': 6, 'weight': 'bold'}
        ax1.text(text_x, text_y, 'a', color='k', **text_style)
        ax2.text(text_x, text_y, 'b', color='k', **text_style)
        ax3.text(text_x, text_y, 'c', color='k', **text_style)

        ax4.text(text_x, text_y, 'd', color='w', **text_style)
        ax5.text(text_x, text_y, 'e', color='w', **text_style)
        ax6.text(text_x, text_y, 'f', color='w', **text_style)

        ax7.text(text_x, text_y, 'g', color='w', **text_style)
        ax8.text(text_x, text_y, 'h', color='w', **text_style)
        ax9.text(text_x, text_y, 'i', color='w', **text_style)

        ax10.text(text_x, text_y, 'j', color='w', **text_style)
        ax11.text(text_x, text_y, 'k', color='w', **text_style)
        ax12.text(text_x, text_y, 'l', color='w', **text_style)

        # Aligned colorbar labels
        layout['ax3']['ax'].text(132.5, 2023, 'Emissions (tCO$_{2}$)', va='center', fontsize=4.7, rotation=90)
        layout['ax6']['ax'].text(132.5, 2023, 'Price ($/MWh)', va='center', fontsize=4.7, rotation=90)
        layout['ax9']['ax'].text(132.5, 2023, 'Baseline (tCO$_{2}$/MWh)', va='center', fontsize=4.7, rotation=90)
        layout['ax12']['ax'].text(132.5, 2023, 'Revenue ($)', va='center', fontsize=4.7, rotation=90)

        # Format labels
        fig.set_size_inches(self.cm_to_in(11.5), self.cm_to_in(7.6))
        fig.subplots_adjust(left=0.09, bottom=0.1, right=0.91, top=0.96, wspace=0.01)
        fig.savefig(os.path.join(self.figures_dir, 'defense', f'transition_years_{model_key}.png'), dpi=400,
                    transparent=True)
        fig.savefig(os.path.join(self.figures_dir, 'defense', f'transition_years_{model_key}.pdf'), transparent=True)

        plt.show()

    def plot_price_target_difference(self, **kwds):
        """Plot difference between price target and realised prices"""

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)

        # Get interpolated surfaces
        X1, Y1, Z1 = self.get_interpolated_surface('baudev_diff', 'YEAR_AVERAGE_PRICE', 2020,
                                                   price_target=kwds['bau_price'])
        X2, Y2, Z2 = self.get_interpolated_surface('baudev_diff', 'YEAR_AVERAGE_PRICE', 2025,
                                                   price_target=kwds['bau_price'])
        X3, Y3, Z3 = self.get_interpolated_surface('baudev_diff', 'YEAR_AVERAGE_PRICE', 2030,
                                                   price_target=kwds['bau_price'])

        X4, Y4, Z4 = self.get_interpolated_surface('ptar_diff', 'YEAR_AVERAGE_PRICE', 2020,
                                                   price_target=kwds['price_trajectory'])
        X5, Y5, Z5 = self.get_interpolated_surface('ptar_diff', 'YEAR_AVERAGE_PRICE', 2025,
                                                   price_target=kwds['price_trajectory'])
        X6, Y6, Z6 = self.get_interpolated_surface('ptar_diff', 'YEAR_AVERAGE_PRICE', 2030,
                                                   price_target=kwds['price_trajectory'])

        # Construct plot and keep track of it
        plot_style_trajectory = {'vmin': -30, 'vmax': 30, 'edgecolors': 'face', 'cmap': 'bwr'}
        im1 = ax1.pcolormesh(X1, Y1, Z1, **plot_style_trajectory)
        im2 = ax2.pcolormesh(X2, Y2, Z2, **plot_style_trajectory)
        im3 = ax3.pcolormesh(X3, Y3, Z3, **plot_style_trajectory)

        plot_style_bau = {'vmin': -30, 'vmax': 30, 'edgecolors': 'face', 'cmap': 'bwr'}
        im4 = ax4.pcolormesh(X4, Y4, Z4, **plot_style_bau)
        im5 = ax5.pcolormesh(X5, Y5, Z5, **plot_style_bau)
        im6 = ax6.pcolormesh(X6, Y6, Z6, **plot_style_bau)

        # Add dividers so subplots are the the same size after adding colour bars
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            divider = make_axes_locatable(ax)

            if ax == ax3:
                cax3 = divider.append_axes("right", size="5%", pad=0.05)
            elif ax == ax6:
                cax6 = divider.append_axes("right", size="5%", pad=0.05)
            else:
                divider.append_axes("right", size="5%", pad=0.05).axis('off')

        # Add colour bars
        cb3 = fig.colorbar(im3, cax=cax3)
        cb3.ax.tick_params(labelsize=5.5)
        cb3.set_label('Price difference ($)', fontsize=6.5)

        cb6 = fig.colorbar(im6, cax=cax6)
        cb6.ax.tick_params(labelsize=5.5)
        cb6.set_label('Price difference ($)', fontsize=6.5)

        # Set y-lim for all plots
        for a in [ax1, ax2, ax3, ax4, ax5, ax6]:
            a.set_ylim([2016, 2030])

        # Add lines to denote transition years
        for a in [ax1, ax4]:
            a.plot([5, 100], [2020, 2020], color='k', linestyle='--', linewidth=0.8, alpha=0.9)

        for a in [ax2, ax5]:
            a.plot([5, 100], [2025, 2025], color='k', linestyle='--', linewidth=0.8, alpha=0.9)

        ax3.plot([5, 100], [2029.9, 2029.9], color='k', linestyle='--', linewidth=0.8, alpha=0.9)
        ax6.plot([5, 100], [2029.85, 2029.85], color='k', linestyle='--', linewidth=0.8, alpha=0.9)

        # Format y-ticks and labels
        for ax in [ax1, ax4]:
            ax.yaxis.set_major_locator(MultipleLocator(6))
            ax.yaxis.set_minor_locator(MultipleLocator(2))

            ax.tick_params(axis='both', which='major', labelsize=5.5)
            ax.set_ylabel('Year', fontsize=6.5)

        # Format x-ticks
        for ax in [ax4, ax5, ax6]:
            ax.xaxis.set_major_locator(MultipleLocator(20))
            ax.xaxis.set_minor_locator(MultipleLocator(10))

            ax.tick_params(axis='both', which='major', labelsize=5.5)
            ax.set_xlabel('Emissions price (\$/tCO$_{2}$)', fontsize=6.5)

        # Add titles denoting transition years
        ax1.set_title('2020', fontsize=6.5, pad=2)
        ax2.set_title('2025', fontsize=6.5, pad=2)
        ax3.set_title('2030', fontsize=6.5, pad=2)

        # Add letters to differentiate plots
        text_x, text_y = 9, 2016.5
        text_style = {'verticalalignment': 'bottom', 'horizontalalignment': 'left', 'fontsize': 8, 'weight': 'bold'}
        ax1.text(text_x, text_y, 'a', color='k', **text_style)
        ax2.text(text_x, text_y, 'b', color='k', **text_style)
        ax3.text(text_x, text_y, 'c', color='k', **text_style)

        ax4.text(text_x, text_y, 'd', color='k', **text_style)
        ax5.text(text_x, text_y, 'e', color='k', **text_style)
        ax6.text(text_x, text_y, 'f', color='k', **text_style)

        # Set figure size
        fig.set_size_inches(self.cm_to_in(11.5), self.cm_to_in(7.6))
        fig.subplots_adjust(left=0.11, bottom=0.12, right=0.90, top=0.96, wspace=0.01)
        fig.savefig(os.path.join(self.figures_dir, 'defense', 'price_difference.pdf'), transparent=True)
        fig.savefig(os.path.join(self.figures_dir, 'defense', 'price_difference.png'), dpi=400, transparent=True)

        plt.show()

    def plot_tax_rep_comparison_first_year(self):
        """REP scheme and tax comparison"""

        x = [t for t in range(5, 101, 5)]
        p_tax = [self.plot_data.results['tax'][t]['YEAR_AVERAGE_PRICE'][2016] for t in x]
        p_rep = [self.plot_data.results['rep'][t]['YEAR_AVERAGE_PRICE'][2016] for t in x]
        p_bau = self.plot_data.results['bau']['YEAR_AVERAGE_PRICE'][2016]

        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        bau_style = {'color': 'k', 'linestyle': ':', 'alpha': 0.8, 'linewidth': 0.7}
        ax.plot([5, 100], [p_bau, p_bau], **bau_style)

        # Plot lines
        line_properties = {'alpha': 0.8, 'linewidth': 1.1, 'markersize': 3.5, 'fillstyle': 'none', 'markeredgewidth': 0.5}
        ax.plot(x, p_tax, 'o--', color='#d91818', **line_properties)
        ax.plot(x, p_rep, 'o--', color='#4263f5', **line_properties)
        ax.legend(['BAU', 'Tax', 'REP'], fontsize=7, frameon=False)

        # Installed gas capacity
        g_c = [sum(v for k, v in self.plot_data.results['tax'][t]['x_c'].items()
                   if (('OCGT' in k[0]) or ('CCGT' in k[0])) and (k[1] == 2016)) for t in x]
        ax2.plot(x, g_c, 'o--', color='#4fa83d', **line_properties)

        # Labels
        ax.set_ylabel('Average price ($/MWh)')
        ax.set_xlabel('Emissions price (tCO$_{2}$/MWh')

        # Format axes
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_minor_locator(MultipleLocator(5))

        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_minor_locator(MultipleLocator(5))

        ax2.set_ylabel('New gas capacity (MW)')
        ax2.legend(['Gas'], fontsize=7, frameon=False, loc='lower right',
                   # bbox_to_anchor=(0.5, 0.5),
                   )

        ax2.yaxis.set_major_locator(MultipleLocator(2000))
        ax2.yaxis.set_minor_locator(MultipleLocator(1000))

        ax.set_ylim([0, 120])

        # Set font size
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)

        ax2.xaxis.label.set_size(8)
        ax2.yaxis.label.set_size(8)

        ax.tick_params(axis='both', which='major', labelsize=7.4)
        ax2.tick_params(axis='y', which='major', labelsize=7.4)
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(3, 3), useMathText=True)
        ax2.yaxis.offsetText.set_fontsize(8)

        # Adjust figures
        fig.set_size_inches(self.cm_to_in(11.5), self.cm_to_in(7.6))
        fig.subplots_adjust(left=0.11, bottom=0.14, right=0.91, top=0.94)

        fig.savefig(os.path.join(self.figures_dir, 'defense', 'price_sensitivity.pdf'), transparent=True)
        fig.savefig(os.path.join(self.figures_dir, 'defense', 'price_sensitivity.png'), dpi=400, transparent=True)

        plt.show()

    def baseline_slice(self):
        """Plot slice of baseline and average emissions intensity for permit price 50 $/tCO2"""

        x = range(2016, 2031)
        y1 = [self.plot_data.results['pdev'][2025][60]['baseline'][y] for y in x]
        y2 = [self.plot_data.results['pdev'][2025][60]['YEAR_SCHEME_EMISSIONS_INTENSITY'][y] for y in x]
        y12 = [self.plot_data.results['pdev'][2025][60]['YEAR_AVERAGE_PRICE'][y] for y in x]

        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.plot(x, y1, color='r')
        ax.plot(x, y2, color='b')
        ax2.plot(x, y12, color='k')
        plt.show()


if __name__ == '__main__':
    results_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '3_model', 'linear', 'output', 'local')
    tmp_directory = os.path.join(os.path.dirname(__file__), 'output', 'tmp', 'local')
    figures_directory = os.path.join(os.path.dirname(__file__), 'output', 'figures')

    # Object used to analyse results and get price target trajectory
    analysis = AnalyseResults()

    # Get plot data
    plot_data = PlotData(tmp_directory)
    plots = CreatePlots(tmp_directory, figures_directory)

    # Get BAU price trajectories
    bau = analysis.load_results(results_directory, 'bau_case.pickle')
    bau_prices = analysis.get_year_average_price(bau['PRICES'], -1)
    bau_price_trajectory = bau_prices['average_price_real'].to_dict()
    bau_first_year_trajectory = {y: bau_price_trajectory[2016] for y in range(2016, 2031)}

    # Create plots
    plots.plot_tax_rep_comparison()
    plots.plot_transition_year_comparison('baudev')
    plots.plot_transition_year_comparison('ptar')
    plots.plot_transition_year_comparison('pdev')

    plot_params = {'price_trajectory': bau_price_trajectory, 'bau_price': bau_first_year_trajectory}
    plots.plot_price_target_difference(**plot_params)
    plots.plot_tax_rep_comparison_first_year()
