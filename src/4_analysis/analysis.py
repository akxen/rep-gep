"""Analyse output from BAU case"""

import os
import sys
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, '3_model', 'linear', 'components'))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, '3_model', 'linear', 'components', 'base'))

import pandas as pd
import matplotlib.pyplot as plt

from data import ModelData


class AnalyseResults:
    def __init__(self):
        self.data = ModelData()

    @staticmethod
    def load_results(results_dir, filename):
        """Load model results"""

        # Load results
        with open(os.path.join(results_dir, filename), 'rb') as f:
            results = pickle.load(f)

        return results

    def parse_prices(self, prices, factor):
        """Get price information - keys will be different for primal and MPPDC models

        Parameters
        ----------
        prices : dict
            Prices for each zone, year, scenario and interval

        factor : int
            Scale input prices by either 1 or -1 (depends if solver returns prices as negative values)
        """

        # Prices
        df_p = (pd.Series(prices).rename_axis(['zone', 'year', 'scenario', 'interval'])
                .to_frame(name='price').mul(factor))

        # Demand
        df_d = (self.data.input_traces['DEMAND'].stack().stack().rename_axis(['year', 'scenario', 'interval', 'zone'])
                .reset_index(name='demand').set_index(['zone', 'year', 'scenario', 'interval']))

        # Scenario duration
        df_t = (self.data.input_traces.loc[:, ('K_MEANS', 'METRIC', 'DURATION')]
                .rename_axis(['year', 'scenario']).rename('duration'))

        # Combine price, demand, and duration into single DataFrame
        df_c = pd.concat([df_p, df_d], axis=1).join(df_t, how='left')

        # Remove rows with missing values
        df_c = df_c.dropna(how='any')

        # Final year in model horizon
        final_year = df_c.index.get_level_values(0).max()

        # Factor by which prices must be scaled to take into account end-of-horizon effect
        df_c['eoh_scaling_factor'] = df_c.apply(lambda x: (1 + (1 / self.data.WACC)) if x.name[0] == final_year else 1,
                                                axis=1)

        # Scaled prices
        df_c['price_scaled'] = df_c['price'].div(df_c['duration']).div(df_c['eoh_scaling_factor'])

        # Interval revenue
        df_c['revenue'] = df_c['price_scaled'].mul(df_c['demand'])

        # Revenue scaled by scenario duration
        df_c['revenue_scaled'] = df_c['revenue'].mul(df_c['duration'])

        # Demand scaled by scenario duration
        df_c['demand_scaled'] = df_c['demand'].mul(df_c['duration'])

        # Average price for each scenario - discounted
        df_c['average_price_discounted'] = df_c['revenue_scaled'].div(df_c['demand_scaled'])

        # Discount factor
        df_c['discount_factor'] = df_c.apply(lambda x: 1.06 ** (x.name[0] - 2016), axis=1)

        # Average price for each scenario - discounted
        df_c['average_price_real'] = df_c['average_price_discounted'].mul(df_c['discount_factor'])

        return df_c

    def get_year_system_emissions_intensities(self, results_dir, filename):
        """Get yearly emissions intensities for they system"""

        # Results dictionary
        results = self.load_results(results_dir, filename)

        # Emissions intensities for each year in model horizon
        df = pd.Series(results['YEAR_EMISSIONS_INTENSITY']).rename_axis('year').to_frame(name='emissions_intensity')

        return df

    def plot_year_system_emissions_intensities(self, results_dir, filename):
        """Plot system emissions intensities for each year in model horizon"""

        # Get emissions intensities for each year in model horizon
        df = self.get_year_system_emissions_intensities(results_dir, filename)

        # Plot emissions intensities
        df.plot()
        plt.show()

    def get_installed_capacity(self, results_dir, filename):
        """Get installed capacity for each year of model horizon (candidate units)"""

        # Results dictionary
        results = self.load_results(results_dir, filename)

        # Total installed candidate capacity for each year in model horizon
        df = (pd.Series(results['x_c']).rename_axis(['unit', 'year'])
              .reset_index(name='capacity').pivot(index='year', columns='unit', values='capacity').cumsum())

        return df

    def plot_installed_capacity(self, results_dir, filename):
        """Plot installed capacity for each candidate unit"""

        # Total installed candidate capacity for each year in model horizon
        df = self.get_installed_capacity(results_dir, filename)

        # Plot of all candidate units with positive capacity installed over model horizon
        df.loc[:, ~df.eq(0).all(axis=0)].plot()
        plt.show()

    def get_lost_load(self, results_dir, filename):
        """Analyse lost-load for given years in model horizon"""

        # Results dictionary
        results = self.load_results(results_dir, filename)

        # Load lost
        df = pd.Series(results['p_V']).rename_axis(['zone', 'year', 'scenario', 'hour']).to_frame('lost_load')

        return df

    def get_year_input_traces(self, year, trace):
        """
        Check input traces for wind, solar, and demand for given years

        Parameters
        ----------
        year : int
            Year for which trace should be analysed

        trace : str
            Name of trace to extract. Either 'WIND', 'DEMAND', 'SOLAR'
        """

        assert trace in ['WIND', 'DEMAND', 'SOLAR'], f'Unexpected trace: {trace}'

        return self.data.input_traces.loc[year, trace].T.rename_axis(['zone', 'interval'])

    def get_year_average_price(self, prices, factor=-1):
        """Compute average prices for each year"""

        # Get discounted and scaled prices for each scenario (with demand and duration)
        parsed_prices = self.parse_prices(prices, factor=factor)

        # Average discounted prices
        df = (parsed_prices.groupby('year').apply(lambda x: x['revenue_scaled'].sum() / x['demand_scaled']
                                                  .sum()).to_frame(name='average_price_discounted'))

        # Discount factor
        df['discount_factor'] = df.apply(lambda x: 1.06 ** (x.name - 2016), axis=1)

        # Average real prices (corrected for discount factor)
        df['average_price_real'] = df['average_price_discounted'] * df['discount_factor']

        return df

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

    def get_interval_generator_output(self, results_dir, filename):
        """Get generator output for each interval along with capacity, zone, region, and fuel type information"""

        # Results from primal model
        results = self.load_results(results_dir, filename)

        # Generator output
        df = pd.Series(results['p']).rename_axis(['generator', 'year', 'scenario', 'interval']).to_frame('output')

        # Scenario duration
        duration_map = (self.data.input_traces.loc[:, ('K_MEANS', 'METRIC', 'DURATION')]
                        .rename_axis(['year', 'scenario']).to_frame('duration'))

        # Zone map
        zone_map = (pd.concat([self.data.existing_units.loc[:, ('PARAMETERS', 'NEM_ZONE')],
                               self.data.candidate_units.loc[:, ('PARAMETERS', 'ZONE')]]).to_frame(name='zone')
                    .rename_axis('generator'))

        # Fuel category
        fuel_map = (pd.concat([self.data.existing_units.loc[:, ('PARAMETERS', 'FUEL_CAT')],
                               self.data.candidate_units.loc[:, ('PARAMETERS', 'FUEL_TYPE_PRIMARY')]])
                    .to_frame(name='fuel'))

        # Create standardised assignment
        fuel_categories = {'WIND': 'RENEWABLE', 'HYDRO': 'RENEWABLE', 'SOLAR': 'RENEWABLE', 'GAS': 'FOSSIL',
                           'COAL': 'FOSSIL', 'FOSSIL': 'FOSSIL'}
        fuel_map_standardised = (fuel_map.apply(lambda x: fuel_categories[x['fuel'].upper()], axis=1)
                                 .to_frame(name='fuel').rename_axis('generator'))

        # Existing capacity (doesn't take into account unit retirement)
        existing_capacity_map = self.data.existing_units.loc[:, ('PARAMETERS', 'REG_CAP')].to_frame(
            'capacity').rename_axis('generator')

        # Candidate capacity available in each year
        candidate_capacity_map = (pd.Series(results['x_c']).rename_axis(['generator', 'year']).unstack().T.cumsum()
                                  .stack().to_frame(name='capacity'))

        # Merge existing capacity information
        df_merged_1 = df.join(existing_capacity_map, how='left')

        # Merge candidate capacity information
        df_merged_2 = pd.merge(df.reset_index(), candidate_capacity_map.reset_index(), how='left',
                               left_on=['year', 'generator'], right_on=['year', 'generator']).set_index(
            ['generator', 'year', 'scenario', 'interval'])

        # Update values
        df_merged_1.update(df_merged_2, overwrite=False)
        df_merged = df_merged_1.copy()

        # Merge zone and capacity information with generator output
        df_merged = df_merged.join(zone_map, how='left').join(fuel_map_standardised, how='left').join(duration_map,
                                                                                                      how='left')

        # Map between NEM zones and regions
        region_map = (self.data.existing_units.drop_duplicates(subset=[('PARAMETERS', 'NEM_ZONE'),
                                                                       ('PARAMETERS', 'NEM_REGION')])
                      .set_index(('PARAMETERS', 'NEM_ZONE')).loc[:, ('PARAMETERS', 'NEM_REGION')]
                      .rename_axis('NEM_ZONE').to_frame(name='region'))

        # Merge region information
        df_merged = pd.merge(df_merged, region_map, left_on='zone', right_index=True)

        return df_merged

    def get_total_emissions(self, results_dir, filename):
        """Compute total emissions over model horizon"""

        # Results from model
        results = self.load_results(results_dir, filename)

        return sum(results['YEAR_EMISSIONS'].values())

    def get_year_emissions(self, results_dir, filename):
        """Get total emissions in each year of model horizon"""

        # Results from model
        results = self.load_results(results_dir, filename)

        return results['YEAR_EMISSIONS']

    def get_baselines(self, results_dir, filename):
        """Get emissions intensity baselines from MPPDC model"""

        # Load results
        results = self.load_results(results_dir, filename)

        # Largest key = results from last iteration
        max_key = max(results['stage_3_price_targeting'].keys())

        # Extract baselines
        if 'mppdc' in filename:
            baselines = results['stage_3_price_targeting'][max_key]['baseline']
        elif 'heuristic' in filename:
            baselines = results['stage_3_price_targeting'][max_key]['primal']['baseline']
        else:
            raise Exception(f"""Expected 'mppdc' or 'heuristic' in filename: {filename}""")

        return pd.Series(baselines).rename_axis('year').rename('baseline')

    def get_average_prices(self, results_dir, filename, stage, price_key, price_factor):
        """Get average prices for a given results file"""

        with open(os.path.join(results_dir, filename), 'rb') as f:
            results = pickle.load(f)

        if filename == 'bau_case.pickle':
            prices = self.get_year_average_price(results[price_key], factor=price_factor)

        else:
            # Max key = last iteration for given stage
            max_key = max(results[stage].keys())

            if type(max_key) == int:
                if 'heuristic' in filename:
                    prices = self.get_year_average_price(results[stage][max_key]['primal'][price_key], factor=price_factor)
                else:
                    prices = self.get_year_average_price(results[stage][max_key][price_key], factor=price_factor)
            else:
                prices = self.get_year_average_price(results[stage][price_key], factor=price_factor)

        return prices


if __name__ == '__main__':
    # Path where results can be found
    results_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '3_model', 'linear', 'output', 'local')

    # Object used to analyse results
    analysis = AnalyseResults()

    # Load model results given a filename
    r = analysis.load_results(results_directory, 'heuristic_price_change_deviation_case.pickle')

    # Prices from different models
    p_bau = analysis.get_average_prices(results_directory, 'bau_case.pickle', None, 'PRICES', -1)
    p_rep = analysis.get_average_prices(results_directory, 'rep_case.pickle', 'stage_2_rep', 'PRICES', -1)
    p_tax = analysis.get_average_prices(results_directory, 'rep_case.pickle', 'stage_1_carbon_tax', 'PRICES', -1)
    # p_bau_dev_mppdc = analysis.get_average_prices(results_directory, 'mppdc_bau_deviation_case.pickle', 'stage_3_price_targeting', 'lamb', 1)
    # p_bau_dev_heuristic = analysis.get_average_prices(results_directory, 'heuristic_bau_deviation_case.pickle', 'stage_3_price_targeting', 'PRICES', -1)
    p_price_dev_mppdc = analysis.get_average_prices(results_directory, 'mppdc_price_change_deviation_case.pickle', 'stage_3_price_targeting', 'lamb', 1)
    p_price_dev_heuristic = analysis.get_average_prices(results_directory, 'heuristic_price_change_deviation_case.pickle', 'stage_3_price_targeting', 'PRICES', -1)

    # Baselines from different models
    b_price_dev_mppdc = analysis.get_baselines(results_directory, 'mppdc_price_change_deviation_case.pickle')
    b_price_dev_heuristic = analysis.get_baselines(results_directory, 'heuristic_price_change_deviation_case.pickle')
    # b_bau_dev_mppdc = analysis.get_baselines(results_directory, 'mppdc_bau_deviation_case.pickle')
    # b_bau_dev_heuristic = analysis.get_baselines(results_directory, 'heuristic_bau_deviation_case.pickle')

    # Plotting baselines - price deviation objective
    fig, ax = plt.subplots()
    b_price_dev_mppdc.plot(ax=ax, color='red')
    b_price_dev_heuristic.plot(ax=ax, color='blue')
    ax.set_title('Price deviation objective')
    plt.show()

    fig, ax = plt.subplots()
    p_price_dev_mppdc['average_price_real'].plot(ax=ax, color='red')
    p_price_dev_heuristic['average_price_real'].plot(ax=ax, color='blue')
    ax.set_title('Price deviation objective')
    plt.show()

    # # Plotting
    # fig, ax = plt.subplots()
    # b_bau_dev_mppdc.plot(ax=ax)
    # b_bau_dev_heuristic.plot(ax=ax)
    # ax.set_title('BAU deviation objective')
    # plt.show()
