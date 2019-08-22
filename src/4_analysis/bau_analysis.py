"""Analyse output from BAU case"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, '3_model', 'linear', 'components', 'base'))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, '3_model', 'linear', 'components'))

import pickle
import pandas as pd
import matplotlib.pyplot as plt

from data import ModelData
from utils import ParseOutput
from gep import Primal


class AnalyseResults:
    def __init__(self):
        self.data = ModelData()
        self.parser = ParseOutput()

        self.results_dir = os.path.join(os.path.dirname(__file__), os.path.pardir, '3_model', 'linear', 'output', 'remote')

        # BAU results
        self.bau_results = pd.read_pickle(os.path.join(self.results_dir, 'primal_bau_results.pickle'))

    def plot_system_emissions_intensities(self):
        """Plot system emissions intensities for each year in model horizon"""

        # Plot emissions intensities
        pd.DataFrame.from_dict(self.bau_results['YEAR_EMISSIONS_INTENSITY'], orient='index').plot()
        plt.show()

    def plot_installed_capacity(self):
        """Plot installed capacity for each candidate unit"""

        # Total installed candidate capacity for each year in model horizon
        df_c = (pd.Series(self.bau_results['x_c']).rename_axis(['unit', 'year'])
                .reset_index(name='capacity').pivot(index='year', columns='unit', values='capacity').cumsum())

        # Plot of all candidate units with positive capacity installed over model horizon
        df_c.loc[:, ~df_c.eq(0).all(axis=0)].plot()
        plt.show()

    def get_lost_load(self):
        """Analyse lost-load for given years in model horizon"""

        # Load lost
        df_l = (pd.Series(self.bau_results['p_V']).rename_axis(['zone', 'year', 'scenario', 'hour'])
                .rename('lost_load').reset_index())

        return df_l

    def check_input_traces(self, years):
        """Check input traces for wind, solar, and demand for given years"""

        for trace in ['WIND', 'DEMAND', 'SOLAR']:
            for y in years:
                self.data.input_traces.loc[y, trace].T.plot(title=f'{trace} - {y}')
                plt.show()

    def get_generator_year_output(self):
        """Get yearly energy output from each generator"""

        # Generator output
        df_go = (pd.Series(self.bau_results['p']).rename_axis(['generator', 'year', 'scenario', 'hour'])
                 .rename('output').reset_index())

        # Merge fuel type for each generator
        df_go_1 = pd.merge(df_go, self.data.existing_units.loc[:, ('PARAMETERS', 'FUEL_TYPE')], how='left',
                           left_on='generator',
                           right_index=True).rename(columns={('PARAMETERS', 'FUEL_TYPE'): 'fuel_type'})

        df_go_2 = pd.merge(df_go, self.data.candidate_units.loc[:, ('PARAMETERS', 'FUEL_TYPE_PRIMARY')], how='left',
                           left_on='generator',
                           right_index=True).rename(columns={('PARAMETERS', 'FUEL_TYPE_PRIMARY'): 'fuel_type'})

        # Merging fuel types for existing and candidate generators
        df_go_1.update(df_go_2)

        return df_go_1

    def plot_generator_year_output(self):
        """Plot generator energy output for each year in model horizon"""

        # Output for each year in model horizon
        df = self.get_generator_year_output()

        # Check output by generator and fuel type
        df.groupby(['year', 'fuel_type'])['output'].sum().unstack().plot()
        plt.show()

    def get_average_prices(self):
        """Get average price for each year in model horizon"""

        # Load prices
        df = self.parser.parse_prices(mode='primal')

        # Compute average price
        avg_price = (df.groupby(['year', 'zone'])['revenue_scaled'].sum()
                     .div(df.groupby(['year', 'zone'])['demand_scaled'].sum()))

        return avg_price

    def plot_real_prices(self):
        """Plot prices (non-discounted) for each year in model horizon"""

        # Scaling factor used to convert discounted average prices to real values
        scaling_factor = pd.Series({y: (1 + 0.06) ** (y - 2016) for y in range(2016, 2051)})

        # Plotting prices
        average_prices = self.get_average_prices()

        # Scale prices and plot
        average_prices.mul(scaling_factor).plot()
        plt.show()

    def plot_demand_duration_curve(self, year):
        """Compare demand duration curve used in model to actual demand duration curve for given year"""

        # Model demand duration curve
        # ---------------------------
        # Duration for each scenario
        duration = (self.data.input_traces.loc[year, ('K_MEANS', 'METRIC', 'DURATION')].rename_axis(['scenario'])
                    .rename('duration'))

        # Scenario demand
        demand = (self.data.input_traces.loc[:, 'DEMAND'].stack().sum(axis=1).loc[year]
                  .rename_axis(['scenario', 'hour']).rename('demand').reset_index())

        # Combining demand and scenario duration information into a single DataFrame
        demand_duration = pd.merge(demand, duration, how='left', left_on='scenario', right_index=True)

        # Sort by demand (highest at top of frame)
        demand_duration = demand_duration.sort_values(by='demand', ascending=False)

        # Cumulative duration
        demand_duration['duration_cumsum'] = demand_duration['duration'].cumsum()

        # Plot approximate load duration curve
        ax = demand_duration.plot.scatter(x='duration_cumsum', y='demand', color='b')

        # Actual demand duration curve
        # ----------------------------
        # Path to dataset files
        path = os.path.join(os.path.dirname(__file__), os.path.pardir, '2_input_traces', 'output')

        # Load dataset
        dataset = pd.read_hdf(os.path.join(path, 'dataset.h5'))

        # Actual demand
        actual_demand = (dataset['DEMAND'].loc[dataset['DEMAND'].index.year == year, :].sum(axis=1)
                         .sort_values(ascending=False).to_frame(name='demand'))

        # Cumulative sum of duration
        actual_demand['duration_cumsum'] = range(1, actual_demand.shape[0] + 1)

        # Plot demand duration curve
        actual_demand.plot.scatter(x='duration_cumsum', y='demand', ax=ax, color='r')

        # Add title
        ax.set_title(f'Demand duration curve - {year}')
        plt.show()


if __name__ == '__main__':
    # Object used to analyse results
    analysis = AnalyseResults()

    # Plot demand duration curves for given years
    # for y in range(2040, 2051):
    #     analysis.plot_demand_duration_curve(y)

    df = analysis.parser.parse_prices(mode='primal')

    primal = Primal(final_year=2050, scenarios_per_year=1)
    model = primal.construct_model()
    marginal_costs = (pd.Series({k: v for k, v in model.C_MC.items()}).rename_axis(['generator', 'year'])
                      .rename('marginal_cost').sort_values(ascending=False))

    df_l = analysis.get_lost_load()

    inspect_lost_load = df_l.loc[(df_l.year == 2044) & (df_l.scenario == 6) & (df_l.hour == 18), :]

    inspect_prices = df.loc[(2044, 6, slice(None), 18)]

