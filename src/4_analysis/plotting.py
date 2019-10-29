"""Plotting results from REP and price targeting models"""

import os
import pickle

import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from analysis import AnalyseResults


v = {'carbon_tax': {'carbon_price': {'emissions': None, 'average_price': None, 'baselines': None}}}
w = {'rep': {'carbon_price': {'emissions': None, 'average_price': None, 'baselines': None}}}
x = {'heuristic': {'transition_year': {'carbon_price': {'emissions': None, 'average_price': None, 'baselines': None}}}}
y = {'mppdc': {'transition_year': {'carbon_price': {'emissions': None, 'average_price': None, 'baselines': None}}}}


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
    result_keys = ['YEAR_EMISSIONS', 'baseline', 'YEAR_AVERAGE_PRICE']
    rep_results = extract_rep_results(results_dir, result_keys)
    heuristic_results = extract_price_targeting_results(results_dir, result_keys)
    mppdc_results = extract_mppdc_results(results_dir, result_keys)

    # Combine into single dictionary
    combined_results = {'rep': rep_results, 'heuristic': heuristic_results, 'mppdc': mppdc_results}

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
    if model_key == 'rep':
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
                if model_key == 'rep':
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


if __name__ == '__main__':
    results_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '3_model', 'linear', 'output', 'remote')
    output_directory = os.path.join(os.path.dirname(__file__), 'output')

    # Extract model results
    # c_results = extract_model_results(results_directory, output_directory)
    m_results = load_model_results(output_directory)

    # Get surface features
    X_s, Y_s, Z_s = get_surface_features(m_results, 'heuristic', 'baseline', transition_year=2025)

    # Plot figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X_s, Y_s, Z_s, linewidth=0, cmap=cm.coolwarm, antialiased=True)
    ax.set_zlim([0, 1.5])
    plt.show()

    for i in range(Z_s.shape[0]):
        print(min(Z_s[i]))
