"""Check results"""

import os
import pickle

import matplotlib.pyplot as plt

from analysis import AnalyseResults


def load_results(output_dir, filename):
    """Load results"""

    with open(os.path.join(output_directory, filename), 'rb') as f:
        r = pickle.load(f)

    return r


def plot_results(title, *args):
    """Plot results"""

    # Container for legend labels
    legend_labels = []

    # Initialise figure
    fig, ax = plt.subplots()

    for arg in args:
        # Extract results
        lists = sorted(arg['values'].items())
        x, y = zip(*lists)

        # Plot results
        ax.plot(x, y, color=arg['color'], alpha=0.8)
        legend_labels.append(arg['label'])

    # Set legend and title
    ax.legend(legend_labels)
    ax.set_title(title)
    plt.show()


if __name__ == '__main__':
    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '3_model', 'linear', 'output', 'local')

    analysis = AnalyseResults()

    # Load results
    # r_bau = load_results(output_directory, 'bau_case.pickle')
    # r_m = load_results(output_directory, 'mppdc_ptar_ty-2025_cp-25.pickle')
    # r_h = load_results(output_directory, 'heuristic_ptar_ty-2025_cp-25.pickle')
    #
    # # Compute average prices
    # p_m = analysis.get_year_average_price(r_m['stage_3_price_targeting'][max(r_m['stage_3_price_targeting'].keys())]['lamb'], factor=1)
    # p_h = analysis.get_year_average_price(r_h['stage_3_price_targeting'][max(r_h['stage_3_price_targeting'].keys())]['primal']['PRICES'], factor=-1)
    # p_bau = analysis.get_year_average_price(r_bau['PRICES'], factor=-1)
    #
    # # Extract baselines
    # b_m = r_m['stage_3_price_targeting'][max(r_m['stage_3_price_targeting'].keys())]['baseline']
    # b_h = r_h['stage_3_price_targeting'][max(r_h['stage_3_price_targeting'].keys())]['primal']['baseline']
    #
    # # Compare results
    # b_m_info = {'label': 'MPPDC', 'color': 'blue', 'values': b_m}
    # b_h_info = {'label': 'Heuristic', 'color': 'red', 'values': b_h}
    # plot_results('Baseline', b_m_info, b_h_info)
    #
    # p_m_info = {'label': 'MPPDC', 'color': 'blue', 'values': p_m['average_price_real'].to_dict()}
    # p_h_info = {'label': 'Heuristic', 'color': 'red', 'values': p_h['average_price_real'].to_dict()}
    # p_bau = {'label': 'BAU', 'color': 'green', 'values': p_bau['average_price_real'].to_dict()}
    # plot_results('Prices', p_m_info, p_h_info, p_bau)
    # plot_results(p_m['average_price_real'].to_dict(), p_h['average_price_real'].to_dict(), 'Prices')

    r = load_results(output_directory, 'heuristic_pdev_ty-2025_cp-40.pickle')
    p = analysis.get_year_average_price(r['stage_3_price_targeting'][max(r['stage_3_price_targeting'].keys())]['primal']['PRICES'], factor=-1)
