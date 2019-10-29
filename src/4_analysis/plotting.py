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
            if k == 'YEAR_AVERAGE_PRICE':
                r = analysis.get_average_prices(results_dir, f, 'stage_3_price_targeting', 'lamb', 1)
                r = r.loc[:, 'average_price_real']

            else:
                r = analysis.extract_results(results_dir, f, k, stage='stage_3_price_targeting', iteration='max')

            # Append results to main container
            results[transition_year][carbon_price][k] = r.to_dict()

    return results


def extract_model_results(results_dir, output_dir):
    """Extract and parse model results. Save to file."""

    # Extract data
    result_keys = ['YEAR_EMISSIONS', 'baseline', 'YEAR_AVERAGE_PRICE', 'YEAR_CUMULATIVE_SCHEME_REVENUE']

    bau_results = extract_bau_results(results_dir, ['YEAR_EMISSIONS', 'baseline', 'YEAR_AVERAGE_PRICE'])
    tax_results = extract_carbon_results(results_dir, result_keys)
    rep_results = extract_rep_results(results_dir, result_keys)
    heuristic_results = extract_price_targeting_results(results_dir, result_keys)
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
    Xi, Yi = np.meshgrid(np.arange(5, 60, 0.5), np.arange(2016, 2029, 0.5))
    Zi = interpolate.griddata((X.flatten(), Y.flatten()), Z.flatten(), (Xi, Yi), method='cubic')

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
    Xi, Yi = np.meshgrid(np.arange(5, 60, 0.5), np.arange(2016, 2029, 0.5))
    Zi = interpolate.griddata((X.flatten(), Y.flatten()), Z.flatten(), (Xi, Yi), method='cubic')

    # Plotting figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    # ax.plot_surface(X, Y, Z, linewidth=0, cmap=cm.coolwarm, antialiased=True)
    ax.plot_surface(Xi, Yi, Zi, linewidth=0, cmap=cm.coolwarm, rstride=1, cstride=1, alpha=1)
    ax.set_title(f'Baselines: {model_key} {transition_year}')

    plt.show()


def plot_price_surface(results, model_key, transition_year=None):
    """Plot emissions surface"""

    # Get surface features
    X, Y, Z = get_surface_features(results, model_key, 'YEAR_AVERAGE_PRICE', transition_year=transition_year)

    # Interpolating points
    Xi, Yi = np.meshgrid(np.arange(5, 60, 0.5), np.arange(2016, 2029, 0.5))
    Zi = interpolate.griddata((X.flatten(), Y.flatten()), Z.flatten(), (Xi, Yi), method='cubic')

    # Plotting figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    # ax.plot_surface(X, Y, Z, linewidth=0, cmap=cm.coolwarm, antialiased=True)
    ax.plot_surface(Xi, Yi, Zi, linewidth=0, cmap=cm.coolwarm, rstride=1, cstride=1, alpha=1)
    ax.set_title(f'Prices: {model_key} {transition_year}')

    plt.show()


def plot_revenue_surface(results, model_key, transition_year=None):
    """Plot emissions surface"""

    # Get surface features
    X, Y, Z = get_surface_features(results, model_key, 'YEAR_CUMULATIVE_SCHEME_REVENUE',
                                   transition_year=transition_year)

    # Interpolating points
    Xi, Yi = np.meshgrid(np.arange(5, 60, 0.5), np.arange(2016, 2029, 0.5))
    Zi = interpolate.griddata((X.flatten(), Y.flatten()), Z.flatten(), (Xi, Yi), method='cubic')

    # Plotting figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    # ax.plot_surface(X, Y, Z, linewidth=0, cmap=cm.coolwarm, antialiased=True)
    ax.plot_surface(Xi, Yi, Zi, linewidth=0, cmap=cm.coolwarm, rstride=1, cstride=1, alpha=1)
    ax.set_title(f'Revenue: {model_key} {transition_year}')

    plt.show()


def plot_baseline_heatmap(results, model_key, transition_year=None):
    """Plot heatmap of emissions intensity baselines"""
    X, Y, Z = get_surface_features(results, model_key, 'baseline', transition_year=transition_year)

    # Interpolating points
    Xi, Yi = np.meshgrid(np.arange(5, 60, 0.5), np.arange(2016, 2029, 0.1))
    Zi = interpolate.griddata((X.flatten(), Y.flatten()), Z.flatten(), (Xi, Yi), method='cubic')

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
    Xi, Yi = np.meshgrid(np.arange(5, 60, 0.5), np.arange(2016, 2029, 0.1))
    Zi = interpolate.griddata((X.flatten(), Y.flatten()), Z.flatten(), (Xi, Yi), method='cubic')

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
    Xi, Yi = np.meshgrid(np.arange(5, 60, 0.5), np.arange(2016, 2029, 0.1))
    Zi = interpolate.griddata((X.flatten(), Y.flatten()), Z.flatten(), (Xi, Yi), method='cubic')

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
    Zi = interpolate.griddata((X.flatten(), Y.flatten()), Z.flatten(), (Xi, Yi), method='cubic')

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
    b_r = m_results['rep'][cp]['baseline']
    b_m = m_results['mppdc'][ty][cp]['baseline']
    b_h = m_results['heuristic'][ty][cp]['baseline']

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


if __name__ == '__main__':
    results_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '3_model', 'linear', 'output', 'remote')
    output_directory = os.path.join(os.path.dirname(__file__), 'output')

    # Extract model results
    # c_results = extract_model_results(results_directory, output_directory)
    m_results = load_model_results(output_directory)

    # Plot surfaces
    plot_emissions_surface(m_results, 'tax')
    plot_emissions_surface(m_results, 'rep')
    plot_emissions_surface(m_results, 'heuristic', transition_year=2020)
    plot_emissions_surface(m_results, 'heuristic', transition_year=2025)
    plot_emissions_surface(m_results, 'heuristic', transition_year=2030)

    plot_baseline_surface(m_results, 'rep')
    plot_baseline_surface(m_results, 'heuristic', transition_year=2020)
    plot_baseline_surface(m_results, 'heuristic', transition_year=2025)
    plot_baseline_surface(m_results, 'heuristic', transition_year=2030)

    plot_price_surface(m_results, 'tax')
    plot_price_surface(m_results, 'rep')
    plot_price_surface(m_results, 'heuristic', transition_year=2020)
    plot_price_surface(m_results, 'heuristic', transition_year=2025)
    plot_price_surface(m_results, 'heuristic', transition_year=2030)

    plot_revenue_surface(m_results, 'rep')
    plot_revenue_surface(m_results, 'heuristic', transition_year=2020)
    plot_revenue_surface(m_results, 'heuristic', transition_year=2025)
    plot_revenue_surface(m_results, 'heuristic', transition_year=2030)

    # Plot heatmaps
    plot_emissions_heatmap(m_results, 'tax')
    plot_emissions_heatmap(m_results, 'rep')
    plot_emissions_heatmap(m_results, 'heuristic', transition_year=2020)
    plot_emissions_heatmap(m_results, 'heuristic', transition_year=2025)
    plot_emissions_heatmap(m_results, 'heuristic', transition_year=2030)

    plot_price_heatmap(m_results, 'tax')
    plot_price_heatmap(m_results, 'rep')
    plot_price_heatmap(m_results, 'heuristic', transition_year=2020)
    plot_price_heatmap(m_results, 'heuristic', transition_year=2025)
    plot_price_heatmap(m_results, 'heuristic', transition_year=2030)

    plot_baseline_heatmap(m_results, 'rep')
    plot_baseline_heatmap(m_results, 'heuristic', transition_year=2020)
    plot_baseline_heatmap(m_results, 'heuristic', transition_year=2025)
    plot_baseline_heatmap(m_results, 'heuristic', transition_year=2030)

    plot_revenue_heatmap(m_results, 'rep')
    plot_revenue_heatmap(m_results, 'heuristic', transition_year=2020)
    plot_revenue_heatmap(m_results, 'heuristic', transition_year=2025)
    plot_revenue_heatmap(m_results, 'heuristic', transition_year=2030)

    # MPPDC and heuristic comparison
    price_comparison(m_results)
    baseline_comparison(m_results)
    revenue_comparison(m_results)
