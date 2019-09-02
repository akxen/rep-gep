"""Generate emissions intensity target based on on BAU results"""

import os
import json

import pandas as pd
import matplotlib.pyplot as plt

from analysis import AnalyseResults


class Targets:
    def __init__(self, results_dir):
        # Object used to analyse results
        self.analysis = AnalyseResults(results_dir)

    @staticmethod
    def get_year_emission_intensity_target(initial_emissions_intensity, half_life, year, start_year):
        """Get half-life emissions intensity target for each year in model horizon"""

        # Re-index such that first year in model horizon is t = 0
        t = year - start_year

        exponent = (-t / half_life)

        return initial_emissions_intensity * (2 ** exponent)

    def get_emissions_intensity_target(self, half_life):
        """Get sequence of yearly emissions intensity targets"""

        # Get emissions intensities for each year of model horizon - BAU case
        df_bau = self.analysis.get_year_system_emissions_intensities('primal_bau_results.pickle')
        df_bau = df_bau.rename(columns={'emissions_intensity': 'bau_emissions_intensity'})

        # First and last years of model horizon
        start, end = df_bau.index[[0, -1]]

        # Initial emissions intensity
        E_0 = df_bau.loc[start, 'bau_emissions_intensity']

        # Emissions intensity target sequence
        target_sequence = {y: self.get_year_emission_intensity_target(E_0, half_life, y, start)
                           for y in range(start, end + 1)}

        # Convert to DataFrame
        df_sequence = pd.Series(target_sequence).rename_axis('year').to_frame('emissions_intensity_target')

        # Combine with bau emissions intensities
        df_c = pd.concat([df_bau, df_sequence], axis=1)

        return df_c

    def get_first_year_average_real_bau_price(self):
        """Get average price in first year of model horizon"""

        # Get average price in first year of model horizon (real price)
        prices = self.analysis.get_year_average_price('primal_bau_results.pickle')

        return prices.iloc[0]['average_price_real']

    @staticmethod
    def load_emissions_intensity_target(filename):
        """Load emissions intensity target"""

        # Check that emissions target loads correctly
        with open(os.path.join(os.path.dirname(__file__), 'output', filename), 'r') as f:
            target = json.load(f)

        # Convert keys from strings to integers
        target = {int(k): v for k, v in target.items()}

        return target

    @staticmethod
    def load_first_year_average_bau_price(filename):
        """Load average price in first year - BAU scenario"""

        # Check that price loads correctly
        with open(os.path.join(os.path.dirname(__file__), 'output', filename), 'r') as f:
            price = json.load(f)

        return price['first_year_average_price']

    def get_cumulative_emissions_target(self, filename, frac):
        """
        Load emissions target

        Parameters
        ----------
        filename : str
            Name of results file on which emissions target will be based

        frac : float
            Target emissions reduction. E.g. 0.5 would imply emissions should be less than or equal to 50% of total
            emissions observed in results associated with 'filename'
        """

        return float(self.analysis.get_total_emissions(filename) * frac)

    @staticmethod
    def load_cumulative_emissions_target():
        """Load cumulative emissions target"""

        with open(os.path.join(os.path.dirname(__file__), 'output', 'cumulative_emissions_target.json'), 'r') as f:
            emissions_target = json.load(f)

        return emissions_target['cumulative_emissions_target']

    def get_interim_emissions_target(self, filename):
        """Load total emissions in each year when pursuing a cumulative emissions cap"""

        # Get emissions in each year of model horizon when pursuing cumulative target
        year_emissions = self.analysis.get_year_emissions(filename)

        return year_emissions

    @staticmethod
    def load_interim_emissions_target():
        """Load interim emissions target"""

        with open(os.path.join(os.path.dirname(__file__), 'output', 'interim_emissions_target.json'), 'r') as f:
            emissions_target = json.load(f)

        # Convert years to integers
        emissions_target = {int(k): v for k, v in emissions_target.items()}

        return emissions_target

    def get_cumulative_emissions_cap_carbon_price(self):
        """Get carbon price from cumulative emissions cap model results"""

        # Results
        results = self.analysis.load_results('cumulative_emissions_cap_results.pickle')

        return results['CUMULATIVE_EMISSIONS_CAP_CONS_DUAL']

    def get_interim_emissions_cap_carbon_price(self):
        """Get carbon price from interim emissions cap model results"""

        # Results
        results = self.analysis.load_results('interim_emissions_cap_results.pickle')

        return results['INTERIM_EMISSIONS_CAP_CONS_DUAL']


if __name__ == '__main__':
    # Directory containing model results
    results_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '3_model', 'linear', 'output', 'remote')

    # Object used to get model targets
    targets = Targets(results_directory)

    # # Cumulative emissions target - based on fraction of BAU emissions
    # f = 'primal_bau_results.pickle'
    # cumulative_target = {'cumulative_emissions_target': targets.get_cumulative_emissions_target(f, 0.5)}
    # with open(os.path.join(os.path.dirname(__file__), 'output', 'cumulative_emissions_target.json'), 'w') as g:
    #     json.dump(cumulative_target, g)
    # cumulative_target_loaded = targets.load_cumulative_emissions_target()
    #
    # # Interim emissions target - based on emissions in each year when pursuing cumulative target
    # f = 'cumulative_emissions_cap_results.pickle'
    # interim_target = targets.get_interim_emissions_target(f)
    # with open(os.path.join(os.path.dirname(__file__), 'output', 'interim_emissions_target.json'), 'w') as g:
    #     json.dump(interim_target, g)
    # interim_target_loaded = targets.load_interim_emissions_target()

    # # Emissions intensity target - assumes system emissions intensity will halve every 25 years
    # df_emissions_target = get_emissions_intensity_target(half_life=25)
    #
    # # Save the emissions target as a json file
    # emissions_target_path = os.path.join(os.path.dirname(__file__), 'output', 'emissions_target.json')
    # df_emissions_target['emissions_intensity_target'].to_json(emissions_target_path)
    #
    # # Average real BAU price in first year of model horizon
    # first_year_average_real_bau_price = {'first_year_average_price': get_first_year_average_real_bau_price()}
    #
    # # Save first year average price information
    # with open(os.path.join(os.path.dirname(__file__), 'output', 'first_year_average_price.json'), 'w') as f:
    #     json.dump(first_year_average_real_bau_price, f)
    #
    # # Check that emissions target loads correctly
    # emissions_target = load_emissions_intensity_target('emissions_target.json')
    #
    # # Check that average price in first year of BAU scenario loads correctly
    # first_year_average_bau_price = load_first_year_average_bau_price('first_year_average_price.json')

    cumulative_cap_carbon_price = targets.get_cumulative_emissions_cap_carbon_price()
    interim_cap_carbon_price = targets.get_interim_emissions_cap_carbon_price()
