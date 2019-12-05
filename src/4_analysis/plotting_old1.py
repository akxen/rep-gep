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

v = {'carbon_tax': {'carbon_price': {'emissions': None, 'average_price': None, 'baselines': None}}}
w = {'rep': {'carbon_price': {'emissions': None, 'average_price': None, 'baselines': None}}}
x = {'heuristic': {'transition_year': {'carbon_price': {'emissions': None, 'average_price': None, 'baselines': None}}}}
y = {'mppdc': {'transition_year': {'carbon_price': {'emissions': None, 'average_price': None, 'baselines': None}}}}


def extract_bau_results(results_dir, keys):
    """Extract REP model results"""

    # Object used to parse results
    analysis = AnalyseResults()

    # All REP files
    filenames = [f for f in os.listdir(results_dir) if 'bau_case' in f]

    # Container for results
    results = {}

    for f in filenames:

        # Extract information for each  key
        for k in keys:
            if k == 'YEAR_AVERAGE_PRICE':
                r = analysis.get_average_prices(results_dir, f, None, 'PRICES', -1)
                r = r.loc[:, 'average_price_real']

            else:
                r = analysis.extract_results(results_dir, f, k, stage=None, iteration='max', model=None)

            # Append results to main container
            results[k] = r.to_dict()

    return results


def extract_rep_results(results_dir, keys):
    """Extract REP model results"""

    # Object used to parse results
    analysis = AnalyseResults()

    # All REP files
    filenames = [f for f in os.listdir(results_dir) if 'rep' in f]

    # Container for results
    results = {}

    for f in filenames:
        # Get carbon price from filename
        carbon_price = int(f.split('-')[1].replace('.pickle', ''))

        if carbon_price not in results.keys():
            results[carbon_price] = {}

        # Extract information for each  key
        for k in keys:
            if k == 'YEAR_AVERAGE_PRICE':
                r = analysis.get_average_prices(results_dir, f, 'stage_2_rep', 'PRICES', -1)
                r = r.loc[:, 'average_price_real']

            else:
                r = analysis.extract_results(results_dir, f, k, stage='stage_2_rep', iteration='max', model=None)

            # Append results to main container
            results[carbon_price][k] = r.to_dict()

    return results


def extract_carbon_results(results_dir, keys):
    """Extract REP model results"""

    # Object used to parse results
    analysis = AnalyseResults()

    # All REP files
    filenames = [f for f in os.listdir(results_dir) if 'rep' in f]

    # Container for results
    results = {}

    for f in filenames:
        # Get carbon price from filename
        carbon_price = int(f.split('-')[1].replace('.pickle', ''))

        if carbon_price not in results.keys():
            results[carbon_price] = {}

        # Extract information for each  key
        for k in keys:
            if k == 'YEAR_AVERAGE_PRICE':
                r = analysis.get_average_prices(results_dir, f, 'stage_1_carbon_tax', 'PRICES', -1)
                r = r.loc[:, 'average_price_real']

            else:
                r = analysis.extract_results(results_dir, f, k, stage='stage_1_carbon_tax', iteration='max', model=None)

            # Append results to main container
            results[carbon_price][k] = r.to_dict()

    return results


def extract_price_targeting_results(results_dir, keys):
    """Extract REP model results"""

    # Object used to parse results
    analysis = AnalyseResults()

    # All REP files
    filenames = [f for f in os.listdir(results_dir) if 'heuristic' in f]

    # Container for results
    results = {}

    for f in filenames:
        # Get carbon price and transition year from filename
        transition_year = int(f.split('-')[1].replace('_cp', ''))
        carbon_price = int(f.split('-')[2].replace('.pickle', ''))

        if transition_year not in results.keys():
            results[transition_year] = {}

        if carbon_price not in results[transition_year].keys():
            results[transition_year][carbon_price] = {}

        # Extract information for each  key
        for k in keys:
            if k == 'YEAR_AVERAGE_PRICE':
                r = analysis.get_average_prices(results_dir, f, 'stage_3_price_targeting', 'PRICES', -1)
                r = r.loc[:, 'average_price_real']

            else:
                r = analysis.extract_results(results_dir, f, k, stage='stage_3_price_targeting', iteration='max',
                                             model='primal')

            # Append results to main container
            results[transition_year][carbon_price][k] = r.to_dict()

    return results


def extract_mppdc_results(results_dir, keys):
    """Extract REP model results"""

    # Object used to parse results
    analysis = AnalyseResults()

    # All REP files
    filenames = [f for f in os.listdir(results_dir) if 'mppdc' in f]

    # Container for results
    results = {}

    for f in filenames:
        # Get carbon price and transition year from filename
        transition_year = int(f.split('-')[1].replace('_cp', ''))
        carbon_price = int(f.split('-')[2].replace('.pickle', ''))

        if transition_year not in results.keys():
            results[transition_year] = {}

        if carbon_price not in results[transition_year].keys():
            results[transition_year][carbon_price] = {}

        # Extract information for each  key
        for k in keys:
            # Check that results are available by extracting the baseline (should exist for all models)
            raw = analysis.extract_results(results_dir, f, 'baseline', stage='stage_3_price_targeting', iteration='max')
            if raw is None:
                r = None

            elif k == 'YEAR_AVERAGE_PRICE':
                r = analysis.get_average_prices(results_dir, f, 'stage_3_price_targeting', 'lamb', 1)
                r = r.loc[:, 'average_price_real']

            else:
                r = analysis.extract_results(results_dir, f, k, stage='stage_3_price_targeting', iteration='max')

            # Append results to main container
            if r is not None:
                results[transition_year][carbon_price][k] = r.to_dict()
            else:
                results[transition_year][carbon_price][k] = None

    return results


def extract_model_results(results_dir, output_dir):
    """Extract and parse model results. Save to file."""

    # Extract data
    result_keys = ['YEAR_EMISSIONS', 'baseline', 'YEAR_AVERAGE_PRICE', 'YEAR_CUMULATIVE_SCHEME_REVENUE', 'x_c']

    print('Extracting BAU results')
    bau_results = extract_bau_results(results_dir, ['YEAR_EMISSIONS', 'baseline', 'YEAR_AVERAGE_PRICE', 'x_c'])

    print('Extracting tax results')
    tax_results = extract_carbon_results(results_dir, result_keys)

    print('Extracting REP results')
    rep_results = extract_rep_results(results_dir, result_keys)

    print('Extracting heuristic results')
    heuristic_results = extract_price_targeting_results(results_dir, result_keys)

    print('Extracting MPPDC results')
    mppdc_results = extract_mppdc_results(results_dir, result_keys)

    # Combine into single dictionary
    combined_results = {'bau': bau_results, 'tax': tax_results, 'rep': rep_results, 'heuristic': heuristic_results,
                        'mppdc': mppdc_results}

    # Save to pickle file
    with open(os.path.join(output_dir, 'model_results.pickle'), 'wb') as f:
        pickle.dump(combined_results, f)

    return combined_results


def load_model_results(directory):
    """Load model results"""

    with open(os.path.join(directory, 'model_results.pickle'), 'rb') as f:
        model_results = pickle.load(f)

    return model_results


def get_surface_features(model_results, model_key, results_key, transition_year=None):
    """Get surface features for a given model"""

    # Results for a given model
    results = model_results[model_key]

    # Carbon prices
    if model_key in ['rep', 'tax']:
        x = list(results.keys())
        y = list(results[x[0]][results_key].keys())

    elif model_key in ['heuristic', 'mppdc']:
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

                elif model_key == 'heuristic':
                    Z.append(results[transition_year][x_cord][results_key][y_cord])

                else:
                    raise Exception(f'Unexpected model_key: {model_key}')

            except Exception as e:
                print(e)
                Z.append(-10)

    # Re-shape Z for plotting
    Z = np.array(Z).reshape(X.shape)

    return X, Y, Z


def plot_emissions_surface(results, model_key, transition_year=None):
    """Plot emissions surface"""

    # Get surface features
    X, Y, Z = get_surface_features(results, model_key, 'YEAR_EMISSIONS', transition_year=transition_year)

    # Interpolating points
    Xi, Yi = np.meshgrid(np.arange(5, 100.1, 0.5), np.arange(2016, 2030.01, 0.5))
    Zi = interpolate.griddata((X.flatten(), Y.flatten()), Z.flatten(), (Xi, Yi), method='linear')

    # Plotting figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    # ax.plot_surface(X, Y, Z, linewidth=0, cmap=cm.coolwarm, antialiased=True)
    ax.plot_surface(Xi, Yi, Zi, linewidth=0, cmap=cm.coolwarm, rstride=1, cstride=1, alpha=1)
    ax.set_title(f'Emissions: {model_key} {transition_year}')

    plt.show()


def plot_baseline_surface(results, model_key, transition_year=None):
    """Plot emissions surface"""

    # Get surface features
    X, Y, Z = get_surface_features(results, model_key, 'baseline', transition_year=transition_year)

    # Interpolating points
    Xi, Yi = np.meshgrid(np.arange(5, 100.1, 0.5), np.arange(2016, 2030.01, 0.5))
    Zi = interpolate.griddata((X.flatten(), Y.flatten()), Z.flatten(), (Xi, Yi), method='linear')

    # Plotting figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    ax.plot_surface(Xi, Yi, Zi, linewidth=0, cmap=cm.coolwarm, rstride=1, cstride=1, alpha=1)
    ax.set_title(f'Baselines: {model_key} {transition_year}')

    plt.show()


def plot_price_surface(results, model_key, transition_year=None):
    """Plot emissions surface"""

    # Get surface features
    X, Y, Z = get_surface_features(results, model_key, 'YEAR_AVERAGE_PRICE', transition_year=transition_year)

    # Interpolating points
    Xi, Yi = np.meshgrid(np.arange(5, 100.1, 0.5), np.arange(2016, 2030.01, 0.5))
    Zi = interpolate.griddata((X.flatten(), Y.flatten()), Z.flatten(), (Xi, Yi), method='linear')

    # Plotting figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    ax.plot_surface(Xi, Yi, Zi, linewidth=0, cmap=cm.plasma, rstride=1, cstride=1, alpha=1)
    ax.set_title(f'Prices: {model_key} {transition_year}')

    return fig, ax


def plot_revenue_surface(results, model_key, transition_year=None):
    """Plot emissions surface"""

    # Get surface features
    X, Y, Z = get_surface_features(results, model_key, 'YEAR_CUMULATIVE_SCHEME_REVENUE',
                                   transition_year=transition_year)

    # Interpolating points
    Xi, Yi = np.meshgrid(np.arange(5, 100.1, 0.5), np.arange(2016, 2030.01, 0.5))
    Zi = interpolate.griddata((X.flatten(), Y.flatten()), Z.flatten(), (Xi, Yi), method='linear')

    # Plotting figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    ax.plot_surface(Xi, Yi, Zi, linewidth=0, cmap=cm.coolwarm, rstride=1, cstride=1, alpha=1)
    ax.set_title(f'Revenue: {model_key} {transition_year}')

    plt.show()


def plot_baseline_heatmap(results, model_key, transition_year=None):
    """Plot heatmap of emissions intensity baselines"""
    X, Y, Z = get_surface_features(results, model_key, 'baseline', transition_year=transition_year)

    # Interpolating points
    Xi, Yi = np.meshgrid(np.arange(5, 100.1, 0.5), np.arange(2016, 2030.01, 0.1))
    Zi = interpolate.griddata((X.flatten(), Y.flatten()), Z.flatten(), (Xi, Yi), method='linear')

    fig, ax = plt.subplots()
    cmap = plt.get_cmap('plasma')
    im = ax.pcolormesh(Xi, Yi, Zi, cmap=cmap)
    fig.colorbar(im, ax=ax)
    ax.set_title(f'Baselines: {model_key} {transition_year}')

    plt.show()


def plot_price_heatmap(results, model_key, transition_year=None):
    """Plot heatmap of average prices"""
    X, Y, Z = get_surface_features(results, model_key, 'YEAR_AVERAGE_PRICE', transition_year=transition_year)

    # Interpolating points
    Xi, Yi = np.meshgrid(np.arange(5, 100.1, 0.5), np.arange(2016, 2030.01, 0.1))
    Zi = interpolate.griddata((X.flatten(), Y.flatten()), Z.flatten(), (Xi, Yi), method='linear')

    fig, ax = plt.subplots()
    cmap = plt.get_cmap('plasma')
    im = ax.pcolormesh(Xi, Yi, Zi, cmap=cmap)
    fig.colorbar(im, ax=ax)
    ax.set_title(f'Prices: {model_key} {transition_year}')

    plt.show()


def plot_emissions_heatmap(results, model_key, transition_year=None):
    """Plot heatmap of average prices"""

    X, Y, Z = get_surface_features(results, model_key, 'YEAR_EMISSIONS', transition_year=transition_year)

    # Interpolating points
    Xi, Yi = np.meshgrid(np.arange(5, 100.1, 0.5), np.arange(2016, 2030.01, 0.1))
    Zi = interpolate.griddata((X.flatten(), Y.flatten()), Z.flatten(), (Xi, Yi), method='linear')

    fig, ax = plt.subplots()
    cmap = plt.get_cmap('plasma')
    im = ax.pcolormesh(Xi, Yi, Zi, cmap=cmap)
    fig.colorbar(im, ax=ax)
    ax.set_title(f'Emissions: {model_key} {transition_year}')
    plt.show()


def plot_revenue_heatmap(results, model_key, transition_year=None):
    """Plot heatmap of average prices"""

    X, Y, Z = get_surface_features(results, model_key, 'YEAR_CUMULATIVE_SCHEME_REVENUE',
                                   transition_year=transition_year)

    # Interpolating points
    Xi, Yi = np.meshgrid(np.arange(5, 60, 0.5), np.arange(2016, 2029, 0.1))
    Zi = interpolate.griddata((X.flatten(), Y.flatten()), Z.flatten(), (Xi, Yi), method='linear')

    fig, ax = plt.subplots()
    cmap = plt.get_cmap('plasma')
    im = ax.pcolormesh(Xi, Yi, Zi, cmap=cmap)
    fig.colorbar(im, ax=ax)
    ax.set_title(f'Revenue: {model_key} {transition_year}')
    plt.show()


def price_comparison(results):
    """Comparing prices under MPPDC and heuristic approaches"""

    # Transition year and carbon price
    ty, cp = 2025, 40

    # Extract results
    p_r = results['rep'][cp]['YEAR_AVERAGE_PRICE']
    p_m = results['mppdc'][ty][cp]['YEAR_AVERAGE_PRICE']
    p_h = results['heuristic'][ty][cp]['YEAR_AVERAGE_PRICE']

    fig, ax = plt.subplots()

    # X-axis
    x = list(p_m.keys())
    x.sort()

    # Y-axis plots
    y_r = [p_r[i] for i in x]
    y_m = [p_m[i] for i in x]
    y_h = [p_h[i] for i in x]

    ax.plot(x, y_r, color='green', linewidth=6, alpha=0.5)
    ax.plot(x, y_m, color='red', linewidth=3.5, alpha=0.7)
    ax.plot(x, y_h, color='blue', linewidth=1, linestyle='--')
    ax.legend(['REP', 'MPPDC', 'Heuristic'])
    plt.show()


def baseline_comparison(results):
    """Comparing prices under MPPDC and heuristic approaches"""

    # Transition year and carbon price
    ty, cp = 2025, 40

    # Extract results
    b_r = results['rep'][cp]['baseline']
    b_m = results['mppdc'][ty][cp]['baseline']
    b_h = results['heuristic'][ty][cp]['baseline']

    fig, ax = plt.subplots()

    # X-axis
    x = list(b_m.keys())
    x.sort()

    # Y-axis plots
    y_r = [b_r[i] for i in x]
    y_m = [b_m[i] for i in x]
    y_h = [b_h[i] for i in x]

    ax.plot(x, y_r, color='green', linewidth=6, alpha=0.5)
    ax.plot(x, y_m, color='red', linewidth=3.5, alpha=0.7)
    ax.plot(x, y_h, color='blue', linewidth=1, linestyle='--')
    ax.legend(['REP', 'MPPDC', 'Heuristic'])
    plt.show()


def revenue_comparison(results):
    """Comparing cumulative scheme revenue under different price targeting protocols"""

    # Transition year and carbon price
    ty, cp = 2025, 40

    # Extract results
    r_m = results['mppdc'][ty][cp]['YEAR_CUMULATIVE_SCHEME_REVENUE']
    r_h = results['heuristic'][ty][cp]['YEAR_CUMULATIVE_SCHEME_REVENUE']

    fig, ax = plt.subplots()

    # X-axis
    x = list(r_m.keys())
    x.sort()

    # Y-axis plots
    y_m = [r_m[i] for i in x]
    y_h = [r_h[i] for i in x]

    ax.plot(x, y_m, color='red', linewidth=3.5, alpha=0.7)
    ax.plot(x, y_h, color='blue', linewidth=1, linestyle='--')
    ax.legend(['MPPDC', 'Heuristic'])
    plt.show()


def plot_tax_rep_comparison(results, figures_dir):
    """Compare price and emissions outcomes under a carbon tax and REP scheme"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True)
    cmap = plt.get_cmap('plasma')

    # Extract surface information
    X1, Y1, Z1 = get_surface_features(results, 'tax', 'YEAR_AVERAGE_PRICE', transition_year=None)
    X1i, Y1i = np.meshgrid(np.arange(5, 100.1, 0.5), np.arange(2016, 2030.01, 0.1))
    Z1i = interpolate.griddata((X1.flatten(), Y1.flatten()), Z1.flatten(), (X1i, Y1i), method='linear')

    X2, Y2, Z2 = get_surface_features(results, 'rep', 'YEAR_AVERAGE_PRICE', transition_year=None)
    X2i, Y2i = np.meshgrid(np.arange(5, 100.1, 0.5), np.arange(2016, 2030.01, 0.1))
    Z2i = interpolate.griddata((X2.flatten(), Y2.flatten()), Z2.flatten(), (X2i, Y2i), method='linear')

    X3, Y3, Z3 = get_surface_features(results, 'tax', 'YEAR_EMISSIONS', transition_year=None)
    X3i, Y3i = np.meshgrid(np.arange(5, 100.1, 0.5), np.arange(2016, 2030.01, 0.1))
    Z3i = interpolate.griddata((X3.flatten(), Y3.flatten()), Z3.flatten(), (X3i, Y3i), method='linear')

    X4, Y4, Z4 = get_surface_features(results, 'rep', 'YEAR_EMISSIONS', transition_year=None)
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

    # Add text
    ax1.text(7, 2016.5, 'a', verticalalignment='bottom', horizontalalignment='left', color='white', fontsize=10,
             weight='bold')
    ax2.text(7, 2016.5, 'b', verticalalignment='bottom', horizontalalignment='left', color='white', fontsize=10,
             weight='bold')
    ax3.text(7, 2016.5, 'c', verticalalignment='bottom', horizontalalignment='left', fontsize=10, weight='bold')
    ax4.text(7, 2016.5, 'd', verticalalignment='bottom', horizontalalignment='left', fontsize=10, weight='bold')

    fig.set_size_inches(6.5, 4)
    fig.subplots_adjust(left=0.09, bottom=0.10, right=0.92, top=0.95, wspace=0.1, hspace=0.16)

    # Save figure
    fig.savefig(os.path.join(figures_dir, 'tax_rep.pdf'))
    fig.savefig(os.path.join(figures_dir, 'tax_rep.png'), dpi=200)
    plt.show()


def get_interpolated_surface(results, model_key, results_key, transition_year=None):
    """Construct interpolated surface based on model results"""

    # Original surface
    X, Y, Z = get_surface_features(results, model_key, results_key, transition_year=transition_year)

    # Interpolated grid
    Xi, Yi = np.meshgrid(np.arange(5, 100.1, 0.5), np.arange(2016, 2030.01, 0.1))

    # Interpolated surface
    Zi = interpolate.griddata((X.flatten(), Y.flatten()), Z.flatten(), (Xi, Yi), method='linear')

    return Xi, Yi, Zi


def plot_transition_year_comparison(results):
    """Compare emissions, price, baseline, and cumulative scheme revenue outcomes under different transition years"""

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
            _, _, Z = get_interpolated_surface(results, 'heuristic', results_key, transition_year=ty)

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
        X, Y, Z = get_interpolated_surface(results, 'heuristic', v['results_key'], v['transition_year'])

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
    text_x = 9
    text_y = 2016.5
    ax1.text(text_x, text_y, 'a', verticalalignment='bottom', horizontalalignment='left', color='black', fontsize=10,
             weight='bold')
    ax2.text(text_x, text_y, 'b', verticalalignment='bottom', horizontalalignment='left', color='black', fontsize=10,
             weight='bold')
    ax3.text(text_x, text_y, 'c', verticalalignment='bottom', horizontalalignment='left', color='black', fontsize=10,
             weight='bold')

    ax4.text(text_x, text_y, 'd', verticalalignment='bottom', horizontalalignment='left', color='white', fontsize=10,
             weight='bold')
    ax5.text(text_x, text_y, 'e', verticalalignment='bottom', horizontalalignment='left', color='white', fontsize=10,
             weight='bold')
    ax6.text(text_x, text_y, 'f', verticalalignment='bottom', horizontalalignment='left', color='white', fontsize=10,
             weight='bold')

    ax7.text(text_x, text_y, 'g', verticalalignment='bottom', horizontalalignment='left', color='white', fontsize=10,
             weight='bold')
    ax8.text(text_x, text_y, 'h', verticalalignment='bottom', horizontalalignment='left', color='white', fontsize=10,
             weight='bold')
    ax9.text(text_x, text_y, 'i', verticalalignment='bottom', horizontalalignment='left', color='white', fontsize=10,
             weight='bold')

    ax10.text(text_x, text_y, 'j', verticalalignment='bottom', horizontalalignment='left', color='black', fontsize=10,
              weight='bold')
    ax11.text(text_x, text_y, 'k', verticalalignment='bottom', horizontalalignment='left', color='black', fontsize=10,
              weight='bold')
    ax12.text(text_x, text_y, 'l', verticalalignment='bottom', horizontalalignment='left', color='black', fontsize=10,
              weight='bold')

    # Format labels
    fig.set_size_inches(6.5, 6)
    fig.subplots_adjust(left=0.08, bottom=0.06, right=0.92, top=0.97, wspace=0.01)
    fig.savefig(os.path.join(figures_directory, 'transition_years.png'), dpi=200)
    fig.savefig(os.path.join(figures_directory, 'transition_years.pdf'))

    plt.show()


def plot_mppdc_heuristic_comparison(results):
    """Compare baselines between MPPDC and heuristic solution protocols"""

    # Comparing heuristic and MPPDC solutions
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True)

    # Transition year and carbon price
    cp = 100

    # Extract results
    b_r = results['rep'][cp]['baseline']

    for ty, ax in [(2020, ax1), (2025, ax2), (2030, ax3)]:
        b_m = results['mppdc'][ty][cp]['baseline']
        b_h = results['heuristic'][ty][cp]['baseline']

        # X-axis
        try:
            x = list(b_m.keys())
        except:
            continue
        x.sort()

        # Y-axis plots
        y_r = [b_r[i] for i in x]
        y_m = [b_m[i] for i in x]
        y_h = [b_h[i] for i in x]

        ax.plot(x, y_r, color='green', linewidth=6, alpha=0.5)
        ax.plot(x, y_m, color='red', linewidth=3.5, alpha=0.7)
        ax.plot(x, y_h, color='blue', linewidth=1, linestyle='--')

    ax1.legend(['REP', 'MPPDC', 'Heuristic'], frameon=False, fontsize=7, loc='lower left')
    ax1.set_ylim([0, 1.6])

    for a in [ax1, ax2, ax3]:
        a.xaxis.set_major_locator(MultipleLocator(4))
        a.xaxis.set_minor_locator(MultipleLocator(2))
        a.tick_params(labelsize=6)
        a.set_xlabel('Year', fontsize=7)

    ax1.set_ylabel('Baseline (tCO$_{2}$/MWh)', fontsize=7)

    fig.set_size_inches(6.5, 2.4)
    fig.subplots_adjust(bottom=0.15, left=0.07, top=0.98, right=0.98)
    fig.savefig(os.path.join(figures_directory, 'mppdc_heuristic.png'), dpi=200)
    fig.savefig(os.path.join(figures_directory, 'mppdc_heuristic.pdf'))

    plt.show()


def plot_mppdc_heuristic_check(results, key, transition_year, carbon_price):
    """Check MPPDC and heuristic results"""

    fig, ax = plt.subplots()
    r_h = results['heuristic'][transition_year][carbon_price][key]
    r_m = results['mppdc'][transition_year][carbon_price][key]

    lists_h = sorted(r_h.items())
    x_h, y_h = zip(*lists_h)
    ax.plot(x_h, y_h, color='red', alpha=0.8)

    lists_m = sorted(r_m.items())
    x_m, y_m = zip(*lists_m)
    ax.plot(x_m, y_m, color='blue', alpha=0.8)

    ax.legend(['heuristic', 'mppdc'])

    ax.set_title(f'{key} {transition_year} {carbon_price}')
    plt.show()


def plot_price_surface_formatted(results):
    """Plot a formatted price surface"""

    # Plot REP surfaces
    fig, ax = plot_price_surface(results, 'rep')
    ax.view_init(20, -90)

    ax.w_xaxis.set_major_locator(MultipleLocator(20))
    ax.w_yaxis.set_major_locator(MultipleLocator(6))
    ax.w_zaxis.set_major_locator(MultipleLocator(20))

    fig.canvas.draw()

    x_labels = ax.w_xaxis.get_ticklabels()
    ax.w_xaxis.set_ticklabels(x_labels, ha='center', va='center', fontsize=6)

    y_labels = ax.w_yaxis.get_ticklabels()
    ax.w_yaxis.set_ticklabels(y_labels, ha='center', va='center', fontsize=6)

    z_labels = ax.w_zaxis.get_ticklabels()
    ax.w_zaxis.set_ticklabels(z_labels, ha='center', va='center', fontsize=6)

    ax.w_xaxis.set_tick_params(pad=0)
    ax.w_yaxis.set_tick_params(pad=0)
    ax.w_zaxis.set_tick_params(pad=-3)

    ax.set_xlabel('Carbon price (\$/tCO$_{2}$)', fontsize=6, labelpad=-5)
    ax.set_ylabel('Year', fontsize=6, labelpad=-5)
    ax.set_zlabel('Price (\$/MWh)', fontsize=6, labelpad=-10)
    ax.set_title('')

    fig.set_size_inches(2.5, 2.5)
    fig.subplots_adjust(left=0, bottom=0.11, right=0.95, top=1, wspace=0, hspace=0)

    fig.savefig(os.path.join(figures_directory, 'rep_surface.png'), dpi=200)
    plt.show()


def plot_tax_rep_comparison_first_year(results, figures_dir):
    """REP scheme and tax comparison"""

    x = [t for t in range(5, 101, 5)]
    p_tax = [results['tax'][t]['YEAR_AVERAGE_PRICE'][2016] for t in x]
    p_rep = [results['rep'][t]['YEAR_AVERAGE_PRICE'][2016] for t in x]
    p_bau = results['bau']['YEAR_AVERAGE_PRICE'][2016]

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot([5, 100], [p_bau, p_bau], color='k', linestyle='--', alpha=0.5, linewidth=0.9)

    # Plot lines
    ax.plot(x, p_tax, 'o--', color='#d91818', alpha=0.8, linewidth=0.7, markersize=2, fillstyle='none',
            markeredgewidth=0.5)
    ax.plot(x, p_rep, 'o--', color='#4263f5', alpha=0.8, linestyle='--', linewidth=0.7, markersize=2, fillstyle='none',
            markeredgewidth=0.5)
    ax.legend(['BAU', 'Tax', 'REP'], fontsize=6, frameon=False)

    # Installed gas capacity
    g_c = [sum(v for k, v in m_results['tax'][t]['x_c'].items() if (('OCGT' in k[0]) or ('CCGT' in k[0]))
               and (k[1] == 2016)) for t in x]
    ax2.plot(x, g_c, 'o--', color='#4fa83d', alpha=0.8, linewidth=0.7, markersize=2, fillstyle='none',
             markeredgewidth=0.5)

    # Labels
    ax.set_ylabel('Average price ($/MWh)', fontsize=6)
    ax.set_xlabel('Emissions price (tCO$_{2}$/MWh', fontsize=6)

    # Format axes
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(10))

    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(5))

    ax.tick_params(axis='both', which='major', labelsize=6)
    ax2.tick_params(axis='y', which='major', labelsize=6)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(3, 3), useMathText=True)
    ax2.yaxis.offsetText.set_fontsize(7)

    ax2.set_ylabel('New gas capacity (MW)', fontsize=6)
    ax2.legend(['Gas'], fontsize=6, frameon=False, loc='center', bbox_to_anchor=(0.5, 0.94))

    ax2.yaxis.set_major_locator(MultipleLocator(2000))
    ax2.yaxis.set_minor_locator(MultipleLocator(1000))

    fig.set_size_inches(3, 2.5)
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.85, top=0.92)

    fig.savefig(os.path.join(figures_dir, 'price_sensitivity.png'), dpi=200)
    fig.savefig(os.path.join(figures_dir, 'price_sensitivity.pdf'))

    plt.show()


if __name__ == '__main__':
    results_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '3_model', 'linear', 'output', 'local')
    output_directory = os.path.join(os.path.dirname(__file__), 'output', 'tmp', 'local')
    figures_directory = os.path.join(os.path.dirname(__file__), 'output', 'figures')

    # Extract model results
    # c_results = extract_model_results(results_directory, output_directory)
    m_results = load_model_results(output_directory)

    # # Plot surfaces
    # plot_emissions_surface(m_results, 'tax')
    # plot_emissions_surface(m_results, 'rep')
    # plot_emissions_surface(m_results, 'heuristic', transition_year=2020)
    # plot_emissions_surface(m_results, 'heuristic', transition_year=2025)
    # plot_emissions_surface(m_results, 'heuristic', transition_year=2030)
    #
    # plot_baseline_surface(m_results, 'rep')
    # plot_baseline_surface(m_results, 'heuristic', transition_year=2020)
    # plot_baseline_surface(m_results, 'heuristic', transition_year=2025)
    # plot_baseline_surface(m_results, 'heuristic', transition_year=2030)
    #
    # plot_price_surface(m_results, 'tax')
    # plot_price_surface(m_results, 'rep')
    # plot_price_surface(m_results, 'heuristic', transition_year=2020)
    # plot_price_surface(m_results, 'heuristic', transition_year=2025)
    # plot_price_surface(m_results, 'heuristic', transition_year=2030)
    #
    # plot_revenue_surface(m_results, 'rep')
    # plot_revenue_surface(m_results, 'heuristic', transition_year=2020)
    # plot_revenue_surface(m_results, 'heuristic', transition_year=2025)
    # plot_revenue_surface(m_results, 'heuristic', transition_year=2030)
    #
    # # Plot heatmaps
    # plot_emissions_heatmap(m_results, 'tax')
    # plot_emissions_heatmap(m_results, 'rep')
    # plot_emissions_heatmap(m_results, 'heuristic', transition_year=2020)
    # plot_emissions_heatmap(m_results, 'heuristic', transition_year=2025)
    # plot_emissions_heatmap(m_results, 'heuristic', transition_year=2030)
    #
    # plot_price_heatmap(m_results, 'tax')
    # plot_price_heatmap(m_results, 'rep')
    # plot_price_heatmap(m_results, 'heuristic', transition_year=2020)
    # plot_price_heatmap(m_results, 'heuristic', transition_year=2025)
    # plot_price_heatmap(m_results, 'heuristic', transition_year=2030)
    #
    # plot_baseline_heatmap(m_results, 'rep')
    # plot_baseline_heatmap(m_results, 'heuristic', transition_year=2020)
    # plot_baseline_heatmap(m_results, 'heuristic', transition_year=2025)
    # plot_baseline_heatmap(m_results, 'heuristic', transition_year=2030)
    #
    # plot_revenue_heatmap(m_results, 'rep')
    # plot_revenue_heatmap(m_results, 'heuristic', transition_year=2020)
    # plot_revenue_heatmap(m_results, 'heuristic', transition_year=2025)
    # plot_revenue_heatmap(m_results, 'heuristic', transition_year=2030)
    #
    # # MPPDC and heuristic comparison
    # price_comparison(m_results)
    # baseline_comparison(m_results)
    # revenue_comparison(m_results)

    # Comparing emissions and average wholesale prices under a REP and carbon tax
    # plot_tax_rep_comparison(m_results, figures_directory)

    # Comparing transition years
    # plot_transition_year_comparison(m_results)

    # Compare baselines from MPPDC and heuristic solution protocols
    # plot_mppdc_heuristic_comparison(m_results)

    # Compare carbon tax and REP scheme
    # plot_tax_rep_comparison_first_year(m_results, figures_directory)

    # for c in [25, 50, 75]:
    #     for y in [2020, 2025, 2030]:
    #         try:
    #             plot_mppdc_heuristic_check(m_results, key='baseline', transition_year=y, carbon_price=c)
    #         except:
    #             pass
