"""Creating plots"""

import os

import pandas as pd
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.collections import PatchCollection
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

from analysis import AnalyseResults


def merit_order():
    """Plot base image of generator merit order"""
    fig, ax = plt.subplots()

    rectangles_1 = []
    rectangles_2 = []
    rectangles_3 = []
    rect_1 = Rectangle((0, 0), 150, 20, color='red')
    rect_2 = Rectangle((150, 0), 125, 45, color='blue')
    rect_3 = Rectangle((275, 0), 100, 60, color='grey')

    rectangles_1.append(rect_1)
    rectangles_2.append(rect_2)
    rectangles_3.append(rect_3)

    pc_1 = PatchCollection(rectangles_1, facecolors='#8a6d3d', alpha=0.8, edgecolor='k', linewidth=0.5)
    pc_2 = PatchCollection(rectangles_2, facecolors='#3d728a', alpha=0.8, edgecolor='k', linewidth=0.5)
    pc_3 = PatchCollection(rectangles_3, facecolors='#9c001f', alpha=0.8, edgecolor='k', linewidth=0.5)
    ax.add_collection(pc_1)
    ax.add_collection(pc_2)
    ax.add_collection(pc_3)
    ax.set_xlim([0, 400])
    ax.set_ylim([0, 75])
    ax.tick_params(labelsize=6)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(20))
    ax.set_xlabel('Energy offers (MWh)', fontsize=7)
    ax.set_ylabel('Price ($/MWh)', fontsize=7, labelpad=-0.1)

    fig.set_size_inches(1.9685, 2.756)
    fig.subplots_adjust(left=0.17, bottom=0.13, top=0.99, right=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig, ax


def merit_order_demand(fig, ax):
    """Add demand"""

    ax.plot([200, 200], [0, 68], color='k', linestyle='--', alpha=0.9, linewidth=0.7)
    ax.plot([0, 150], [45, 45], color='k', linestyle='--', alpha=0.9, linewidth=0.6)
    ax.text(150, 70, 'Demand', fontsize=6)
    ax.text(-40, 44, '$\lambda$', fontsize=7, color='red')

    return fig, ax


def merit_order_price_setter(fig, ax):
    """Add demand"""

    ax.plot([0, 150], [45, 45], color='k', linestyle='--', alpha=0.9, linewidth=0.6)
    ax.text(-40, 44, '$\lambda$', fontsize=7, color='red')
    ax.text(80, 46.5, r'$C_{g^{\star}}+(E_{g^{\star}} - \phi)\tau$', fontsize=7)

    ax.text(80, 66, r'$\lambda = C_{g^{\star}}+(E_{g^{\star}} - \phi)\tau$', fontsize=7)

    ax.collections[0].set_alpha(0.2)
    ax.collections[2].set_alpha(0.2)

    return fig, ax


def plot_merit_order(output_dir):
    """Plot and save base merit order image"""

    fig, ax = merit_order()
    fig.savefig(os.path.join(output_dir, 'price_setters_1.png'), dpi=800)
    fig.savefig(os.path.join(output_dir, 'price_setters_1.pdf'), transparent=True)


def plot_merit_order_demand(output_dir):
    """Plot first image with merit order + demand"""

    # Plot base merit order image
    fig, ax = merit_order()

    # Include demand curve
    fig, ax = merit_order_demand(fig, ax)

    # Save figure
    fig.savefig(os.path.join(output_dir, 'price_setters_2.png'), dpi=800)
    fig.savefig(os.path.join(output_dir, 'price_setters_2.pdf'), transparent=True)

    return fig, ax


def plot_merit_order_price_setter(output_dir):
    """Plot first image with merit + demand"""

    # Plot base image
    fig, ax = merit_order()

    # Include demand curve
    fig, ax = merit_order_price_setter(fig, ax)

    # Save figure
    fig.savefig(os.path.join(output_dir, 'price_setters_3.png'), dpi=800)
    fig.savefig(os.path.join(output_dir, 'price_setters_3.pdf'), transparent=True)

    return fig, ax


def plot_gep_prices(results_dir, output_dir):
    """Prices from generation expansion planning model"""

    # Prices from different models
    p_bau = analysis.get_average_prices(results_dir, 'bau_case.pickle', None, 'PRICES', -1)
    p_rep = analysis.get_average_prices(results_dir, 'rep_case.pickle', 'stage_2_rep', 'PRICES', -1)
    p_tax = analysis.get_average_prices(results_dir, 'rep_case.pickle', 'stage_1_carbon_tax', 'PRICES', -1)
    p_price_dev_mppdc = analysis.get_average_prices(results_dir, 'mppdc_price_change_deviation_case.pickle', 'stage_3_price_targeting', 'lamb', 1)
    p_price_dev_heuristic = analysis.get_average_prices(results_dir, 'heuristic_price_change_deviation_case.pickle', 'stage_3_price_targeting', 'PRICES', -1)

    fig, ax = plt.subplots()
    x = p_bau.index.values

    ax.plot(x.tolist(), p_bau['average_price_real'].values.tolist(), color='blue', alpha=0.7)
    ax.plot(x.tolist(), p_tax['average_price_real'].values.tolist(), color='red', alpha=0.7)
    ax.plot(x.tolist(), p_rep['average_price_real'].values.tolist(), color='green', alpha=0.4)
    ax.plot(x.tolist(), p_price_dev_mppdc['average_price_real'].values.tolist(), color='purple', alpha=0.4)
    ax.plot(x.tolist(), p_price_dev_heuristic['average_price_real'].values.tolist(), color='orange', alpha=0.4)
    ax.yaxis.set_minor_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.tick_params(labelsize=7)
    ax.set_xlabel('Year', fontsize=9)
    ax.set_ylabel('Price ($/MWh)', fontsize=9)
    ax.legend(['BAU', 'Tax', 'REP', 'Target (MPPDC)', 'Target (heuristic)'], ncol=2, fontsize=7)
    fig.set_size_inches(4.5, 2.8)
    fig.subplots_adjust(left=0.12, bottom=0.15, top=0.99, right=0.99)
    fig.savefig(os.path.join(output_dir, 'gep_prices.png'))
    fig.savefig(os.path.join(output_dir, 'gep_prices.pdf'), transparent=True)

    plt.show()


def plot_gep_baselines(results_dir, output_dir):
    """Plot baseline results from different GEP model runs"""

    # Baselines from different models
    b_price_dev_mppdc = analysis.get_baselines(results_directory, 'mppdc_price_change_deviation_case.pickle')
    b_price_dev_heuristic = analysis.get_baselines(results_directory, 'heuristic_price_change_deviation_case.pickle')

    # Load REP results
    r = analysis.load_results(results_dir, 'rep_case.pickle')
    b_rep = list(r['stage_2_rep'][max(r['stage_2_rep'].keys())]['baseline'].values())

    fig, ax = plt.subplots()
    x = b_price_dev_mppdc.index.values
    ax.plot(x, b_price_dev_mppdc.values.tolist(), color='purple', alpha=0.8)
    ax.plot(x, b_price_dev_heuristic.values.tolist(), color='orange', alpha=0.5)

    ax.plot(x, b_rep, color='green', linestyle='--', alpha=0.7)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(labelsize=8)
    ax.set_ylabel('Baseline (tCO$_{2}$/MWh)', fontsize=9)
    ax.set_xlabel('Year', fontsize=9)

    fig.set_size_inches(4.5, 2.8)
    fig.subplots_adjust(left=0.12, bottom=0.15, top=0.99, right=0.99)
    ax.legend(['MPPDC', 'Heuristic', 'REP'], fontsize=8)
    fig.savefig(os.path.join(output_dir, 'gep_baselines.png'))
    fig.savefig(os.path.join(output_dir, 'gep_baselines.pdf'), transparent=True)

    plt.show()


if __name__ == '__main__':
    # Output directory
    output_directory = os.path.join(os.path.dirname(__file__), 'output', 'figures')

    # Results directory
    results_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '3_model', 'linear', 'output', 'remote')

    # Object used to analyse results
    analysis = AnalyseResults()

    # Plot merit order (base image)
    plot_merit_order(output_directory)
    plt.show()

    # Plot merit order + demand
    plot_merit_order_demand(output_directory)
    plt.show()

    # Plot merit order + price setter
    plot_merit_order_price_setter(output_directory)
    plt.show()

    # Plot generation expansion planning model prices
    plot_gep_prices(results_directory, output_directory)
    plt.show()

    # Plot generation expansion planning model baselines
    plot_gep_baselines(results_directory, output_directory)
    plt.show()

    def get_installed_cumulative_capacity_technology(results_dir, case):
        """Get installed cumulative capacity for a given case (by technology type)"""

        # Load results
        results = analysis.load_results(results_dir, case)

        # Extract investment results
        x_c = results['stage_2_rep'][max(results['stage_2_rep'].keys())]['x_c']

        # Convert to pandas Series
        df_cap = pd.Series(x_c).rename_axis(['generator', 'year']).rename('capacity')

        # Compute cumulative capacity
        df_cumulative_cap = df_cap.reset_index().pivot(index='year', columns='generator', values='capacity').cumsum()

        # Take transpose and extract technology type for each technology
        df_cumulative_cap_t = df_cumulative_cap.T
        df_cumulative_cap_t['technology'] = df_cumulative_cap_t.apply(lambda x: x.name.split('-')[1], axis=1)

        # Groupby technology type
        df_tech = df_cumulative_cap_t.groupby('technology').sum().T

        return df_tech

    df_inv = get_installed_cumulative_capacity_technology(results_directory, 'rep_case.pickle')

    def get_year_existing_capacity_technology(year):
        """Get existing capacity by technology type"""

        # Existing unit retirement
        df_r = pd.Series(analysis.data.unit_retirement)

        # Generator capacity for a given year
        df_g = analysis.data.existing_units

        # Retired generators
        retired_gens = df_r.loc[df_r <= year].index

        # Generators that could potentially be included in model (scheduled and semi-scheduled)
        included_gens = df_g.loc[df_g[('PARAMETERS', 'SCHEDULE_TYPE')].isin(['SCHEDULED', 'SEMI-SCHEDULED'])].index

        # Available generators
        available_gens = df_g.index.intersection(included_gens).difference(retired_gens)
        df_ga = df_g.reindex(available_gens)

        # Installed capacity for existing generators
        df_c = df_ga.groupby(('PARAMETERS', 'TECHNOLOGY_CAT_PRIMARY')).sum()[('PARAMETERS', 'REG_CAP')]

        # Installed capacity for a given year
        capacity = {(year, k): v for k, v in df_c.to_dict().items()}

        return capacity

    existing_cap = {}
    for y in df_inv.index:
        existing_cap = {**existing_cap, **get_year_existing_capacity_technology(y)}

    df_ec = (pd.Series(existing_cap).rename_axis(['year', 'technology']).rename('capacity')
             .reset_index().pivot(index='year', columns='technology', values='capacity'))

    # Assume liquid + gas generators are same type of facility (GAS)
    df_tc = df_ec.copy()
    df_tc['GAS'] = df_tc['GAS'] + df_tc['LIQUID']
    df_tc = df_tc.drop('LIQUID', axis=1)

    # Combine OCGT and CCGT into a single category (GAS)
    df_invc = df_inv.copy()
    df_invc['GAS'] = df_invc['CCGT'] + df_invc['OCGT']
    df_invc = df_invc.drop(['OCGT', 'CCGT'], axis=1)

    # Total incumbent + installed capacity
    df_total = df_invc + df_tc
    df_total['HYDRO'].update(df_tc['HYDRO'])
    df_total['STORAGE'].update(df_invc['STORAGE'])

    # Total installed capacity in each year
    df_total['TOTAL'] = df_total.sum(axis=1)

    # Proportions of each technology
    df_p = df_total.apply(lambda x: x / x['TOTAL'], axis=1)
    df_p = df_total.copy()

    # Cumulative proportions
    df_pc = df_p.drop(['STORAGE', 'TOTAL'], axis=1).cumsum(axis=1)

    # Plot cumulative capacities
    fig, ax = plt.subplots()

    df_pc.plot(ax=ax, color='k', linewidth=0.5, legend=False)
    x = df_p.index.to_list()
    y0 = [0 for _ in x]
    ax.fill_between(x, df_pc['COAL'].to_list(), y0, facecolor='brown', alpha=0.7)
    ax.fill_between(x, df_pc['GAS'].to_list(), df_pc['COAL'].to_list(), facecolor='green', alpha=0.7)
    ax.fill_between(x, df_pc['HYDRO'].to_list(), df_pc['GAS'].to_list(), facecolor='blue', alpha=0.7)
    ax.fill_between(x, df_pc['SOLAR'].to_list(), df_pc['HYDRO'].to_list(), facecolor='orange', alpha=0.7)
    ax.fill_between(x, df_pc['WIND'].to_list(), df_pc['SOLAR'].to_list(), facecolor='purple', alpha=0.7)

    ax.set_ylabel('Share of cumulative capacity', fontsize=9)
    ax.set_xlabel('Year', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    # ax.yaxis.set_major_locator(MultipleLocator(0.2))
    # ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    # ax.set_ylim([0, 1])
    fig.set_size_inches(4.5, 2.8)

    legend_elements = [Patch(facecolor='brown', label='Coal', alpha=0.7),
                       Patch(facecolor='green', label='Gas', alpha=0.7),
                       Patch(facecolor='blue', label='Hydro', alpha=0.7),
                       Patch(facecolor='orange', label='Solar', alpha=0.7),
                       Patch(facecolor='purple', label='Wind', alpha=0.7)]

    ax.legend(handles=legend_elements, fontsize=9, ncol=2, loc='lower left')
    fig.subplots_adjust(left=0.11, bottom=0.14, top=0.97, right=0.99)
    fig.savefig(os.path.join(output_directory, 'rep_cumulative_capacity.png'))
    fig.savefig(os.path.join(output_directory, 'rep_cumulative_capacity.pdf'), transparent=True)

    plt.show()
