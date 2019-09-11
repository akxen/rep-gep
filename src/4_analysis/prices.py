"""Identify price setting generators in each dispatch interval"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
from pyomo.environ import *

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, '3_model', 'linear', 'components'))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, '3_model', 'linear', 'components', 'base'))

from gep import Dual
from data import ModelData
from analysis import AnalyseResults
from components import CommonComponents


class PriceSetter:
    def __init__(self, final_year=2040, scenarios_per_year=5):
        self.dual = Dual(final_year, scenarios_per_year)
        self.data = ModelData()
        self.analysis = AnalyseResults()

        # Common model components. May need to update these values
        self.common = CommonComponents(final_year=final_year, scenarios_per_year=scenarios_per_year)

        # Model sets
        self.sets = self.get_model_sets()

    def convert_to_frame(self, results_dir, filename, index_name, variable_name):
        """Convert dict to pandas DataFrame"""

        # Load results dictionary
        results = self.analysis.load_results(results_dir, filename)

        # Convert dictionary to DataFrame
        df = pd.Series(results[variable_name]).rename_axis(index_name).to_frame(name=variable_name)

        return df

    def get_model_sets(self):
        """Define sets used in model"""

        # Get all sets used within the model
        m = ConcreteModel()
        m = self.common.define_sets(m)

        return m

    def get_eligible_generators(self):
        """Find generators which are eligible for rebates / penalties"""

        # Eligible generators
        eligible_generators = [g for g in self.sets.G_THERM.union(self.sets.G_C_WIND, self.sets.G_C_SOLAR)]

        return eligible_generators

    def get_generator_cost_parameters(self, results_dir, filename, eligible_generators):
        """
        Get parameters affecting generator marginal costs, and compute the net marginal cost for a given policy
        """

        # Model results
        results = self.analysis.load_results(results_dir, filename)

        # Price setting algorithm
        costs = pd.Series(results['C_MC']).rename_axis(['generator', 'year']).to_frame(name='marginal_cost')

        # Add emissions intensity baseline
        costs = costs.join(pd.Series(results['baseline']).rename_axis('year').to_frame(name='baseline'), how='left')

        # Add permit price
        costs = (costs.join(pd.Series(results['permit_price']).rename_axis('year')
                            .to_frame(name='permit_price'), how='left'))

        # Emissions intensities for existing and candidate units
        existing_emissions = self.analysis.data.existing_units.loc[:, ('PARAMETERS', 'EMISSIONS')]
        candidate_emissions = self.analysis.data.candidate_units.loc[:, ('PARAMETERS', 'EMISSIONS')]

        # Combine emissions intensities into a single DataFrame
        emission_intensities = (pd.concat([existing_emissions, candidate_emissions]).rename_axis('generator')
                                .to_frame('emissions_intensity'))

        # Join emissions intensities
        costs = costs.join(emission_intensities, how='left')

        # Total marginal cost (taking into account net cost under policy)
        costs['net_marginal_cost'] = (costs['marginal_cost']
                                      + (costs['emissions_intensity'] - costs['baseline']) * costs['permit_price'])

        def correct_for_ineligible_generators(row):
            """Update costs so only eligible generators have new costs (ineligible generators have unchanged costs)"""

            if row.name[0] in eligible_generators:
                return row['net_marginal_cost']
            else:
                return row['marginal_cost']

        # Correct for ineligible generators
        costs['net_marginal_cost'] = costs.apply(correct_for_ineligible_generators, axis=1)

        return costs

    def get_price_setting_generators(self, results_dir, filename, eligible_generators):
        """Find price setting generators"""

        # Prices
        prices = self.analysis.parse_prices(results_dir, filename)

        # Generator SRMC and cost parameters (emissions intensities, baselines, permit prices)
        generator_costs = self.get_generator_cost_parameters(results_dir, filename, eligible_generators)

        def get_price_setting_generator(row):
            """Get price setting generator, price difference, and absolute real price"""

            # Year and average real price for a given row
            year, price = row.name[0], row['average_price_real']

            # Absolute difference between price and all generator SRMCs
            abs_price_difference = generator_costs.loc[(slice(None), year), 'net_marginal_cost'].subtract(price).abs()

            # Price setting generator and absolute price difference for that generator
            generator, difference = abs_price_difference.idxmin()[0], abs_price_difference.min()

            # Generator SRMC
            srmc = generator_costs.loc[(generator, year), 'net_marginal_cost']

            # Update generator name to load shedding if price very high (indicative of load shedding)
            if difference > 9000:
                generator = 'LOAD-SHEDDING'
                difference = np.nan
                srmc = np.nan

            return pd.Series({'generator': generator, 'difference': difference, 'price': price, 'srmc': srmc})

        # Find price setting generators
        price_setters = prices.apply(get_price_setting_generator, axis=1)

        # Combine output into single dictionary
        output = {'price_setters': price_setters, 'prices': prices, 'generator_costs': generator_costs}

        return output

    def get_dual_component_existing_thermal(self, results_dir, filename):
        """Get dual variable component of dual constraint for existing thermal units"""

        # def get_existing_thermal_unit_dual_information():
        dfs = []

        for v in ['SIGMA_1', 'SIGMA_2', 'SIGMA_20', 'SIGMA_23']:
            print(v)
            index = ('generator', 'year', 'scenario', 'interval')
            dfs.append(self.convert_to_frame(results_dir, filename, index, v))

        # Place all information in a single DataFrame
        df_c = pd.concat(dfs, axis=1).dropna()

        # Get offset values
        df_c['SIGMA_20_PLUS_1'] = df_c['SIGMA_20'].shift(-1)
        df_c['SIGMA_23_PLUS_1'] = df_c['SIGMA_23'].shift(-1)

        return df_c

    def k(self, g):
        """Mapping generator to the NEM zone to which it belongs"""

        if g in self.sets.G_E:
            return self.data.existing_units_dict[('PARAMETERS', 'NEM_ZONE')][g]

        elif g in self.sets.G_C.difference(self.sets.G_STORAGE):
            return self.data.candidate_units_dict[('PARAMETERS', 'ZONE')][g]

        elif g in self.sets.G_STORAGE:
            return self.data.battery_properties_dict['NEM_ZONE'][g]

        else:
            raise Exception(f'Unexpected generator: {g}')

    @staticmethod
    def merge_generator_node_prices(dual_component, zone_prices):
        """Get prices at the node to which a generator is connected"""

        # Merge price information
        df = (pd.merge(dual_component.reset_index(), zone_prices.reset_index(), how='left',
                       left_on=['zone', 'year', 'scenario', 'interval'],
                       right_on=['zone', 'year', 'scenario', 'interval'])
              .set_index(['generator', 'year', 'scenario', 'interval', 'zone']))

        return df

    def get_generator_cost_information(self, results_dir, filename):
        """Merge generator cost information"""

        # Load results
        delta = self.convert_to_frame(results_dir, filename, 'year', 'DELTA')
        rho = self.convert_to_frame(results_dir, filename, ('year', 'scenario'), 'RHO')
        emissions_rate = self.convert_to_frame(results_dir, filename, 'generator', 'EMISSIONS_RATE')
        baseline = self.convert_to_frame(results_dir, filename, 'year', 'baseline')
        permit_price = self.convert_to_frame(results_dir, filename, 'year', 'permit_price')
        marginal_cost = self.convert_to_frame(results_dir, filename, ('generator', 'year'), 'C_MC')

        # Join information into single dataFrame
        df_c = marginal_cost.join(emissions_rate, how='left')
        df_c = df_c.join(baseline, how='left')
        df_c = df_c.join(permit_price, how='left')
        df_c = df_c.join(delta, how='left')
        df_c = df_c.join(rho, how='left')

        # Add a scaling factor for the final year
        final_year = df_c.index.levels[0][-1]
        df_c['scaling_factor'] = df_c.apply(lambda x: 1 if int(x.name[0]) < final_year else 1 + (1 / 0.06), axis=1)

        return df_c

    def merge_generator_cost_information(self, df, results_dir, filename):
        """Merge generator cost information from model"""

        # Get generator cost information
        generator_cost_info = self.get_generator_cost_information(results_dir, filename)

        df = (pd.merge(df.reset_index(), generator_cost_info.reset_index(), how='left')
              .set_index(['generator', 'year', 'scenario', 'interval', 'zone']))

        return df

    def get_constraint_body_existing_thermal(self, results_dir, filename):
        """Get body of dual power output constraint for existing thermal generators"""

        # Components of dual power output constraint
        duals = self.get_dual_component_existing_thermal(results_dir, filename)

        # Map between generators and zones
        generators = duals.index.levels[0]
        generator_zone_map = (pd.DataFrame.from_dict({g: self.k(g) for g in generators}, orient='index',
                                                     columns=['zone']).rename_axis('generator'))

        # Add NEM zone to index
        duals = duals.join(generator_zone_map, how='left').set_index('zone', append=True)

        # Power balance dual variables
        var_index = ('zone', 'year', 'scenario', 'interval')
        prices = self.convert_to_frame(results_dir, filename, var_index, 'PRICES')

        # Merge price information
        c = self.merge_generator_node_prices(duals, prices)

        # Merge operating cost information
        c = price_setter.merge_generator_cost_information(c, results_dir, filename)

        return c

    def evaluate_constraint_body_existing_thermal(self, results_dir, filename):
        """Evaluate constraint body information for existing thermal units (should = 0)"""

        # Get values of terms constituting the constraint
        c = self.get_constraint_body_existing_thermal(results_dir, filename)

        # Correct for all intervals excluding the last interval of each scenario
        s_1 = (- c['SIGMA_1'].abs() + c['SIGMA_2'].abs() - c['PRICES'].abs()
               + (c['DELTA'] * c['RHO'] * c['scaling_factor'] * (c['C_MC'] + (c['EMISSIONS_RATE'] - c['baseline'])
                                                                 * c['permit_price']))
               + c['SIGMA_20'].abs() - c['SIGMA_20_PLUS_1'].abs()
               - c['SIGMA_23'].abs() + c['SIGMA_23_PLUS_1'].abs())

        # Set last interval to NaN
        s_1.loc[(slice(None), slice(None), slice(None), 24, slice(None))] = np.nan

        # Last interval of each scenario
        s_2 = (- c['SIGMA_1'].abs() + c['SIGMA_2'].abs() - c['PRICES'].abs()
               + (c['DELTA'] * c['RHO'] * c['scaling_factor'] * (c['C_MC'] + (c['EMISSIONS_RATE'] - c['baseline'])
                                                                 * c['permit_price']))
               + c['SIGMA_20'].abs() - c['SIGMA_23'].abs())

        # Update so corrected values for last interval are accounted for
        s_3 = s_1.to_frame(name='body')
        s_3.update(s_2.to_frame(name='body'), overwrite=False)

        return s_3

    def evaluate_constraint_dual_component_existing_thermal(self, results_dir, filename):
        """Evaluate dual component of constraint"""

        # Get values of terms constituting the constraint
        c = self.get_constraint_body_existing_thermal(results_dir, filename)

        # Dual component - correct for intervals excluding the last interval of each scenario
        s_1 = (- c['SIGMA_1'].abs() + c['SIGMA_2'].abs() + c['SIGMA_20'].abs() - c['SIGMA_20_PLUS_1'].abs()
               - c['SIGMA_23'].abs() + c['SIGMA_23_PLUS_1'].abs())

        # Set last interval to NaN
        s_1.loc[(slice(None), slice(None), slice(None), 24, slice(None))] = np.nan

        # Dual component - correct for last interval of each scenario
        s_2 = - c['SIGMA_1'].abs() + c['SIGMA_2'].abs() + c['SIGMA_20'].abs() - c['SIGMA_23'].abs()

        # Combine components
        s_3 = s_1.to_frame(name='body')
        s_3.update(s_2.to_frame(name='body'), overwrite=False)

        return s_3

    def get_price_setting_generators_from_model_results(self, results_dir, filename):
        """Find price setting generators"""

        # Generators eligible for a rebate / penalty under the scheme
        eligible_generators = self.get_eligible_generators()

        # Generator costs
        generator_costs = self.get_generator_cost_information(results_dir, filename)

        # Get prices in each zone for each dispatch interval
        index = ('zone', 'year', 'scenario', 'interval')
        zone_price = self.convert_to_frame(results_dir, filename, index, 'PRICES')

        def correct_permit_prices(row):
            """Only eligible generators face a non-zero permit price"""

            if row.name[1] in eligible_generators:
                return row['permit_price']
            else:
                return 0

        # Update permit prices
        generator_costs['permit_price'] = generator_costs.apply(correct_permit_prices, axis=1)

        # Net marginal costs
        generator_costs['net_marginal_cost'] = (generator_costs['scaling_factor'] * generator_costs['DELTA']
                                                * generator_costs['RHO'] * (generator_costs['C_MC']
                                                                            + (generator_costs['EMISSIONS_RATE']
                                                                               - generator_costs['baseline'])
                                                                            * generator_costs['permit_price']))

        def get_price_setter(row):
            """Find generator whose marginal cost is closest to interval marginal cost"""

            # Extract zone, year, scenario, and interval information
            z, y, s, t = row.name

            # Power balance constraint marginal cost for given interval (related to price)
            p = abs(row['PRICES'])

            # Net marginal costs of all generators for the given interval
            generator_mc = generator_costs.loc[(y, slice(None), s), :]

            # Scenario duration and discount factor (arbitrarily selecting YWPS4 to get a single row)
            rho, delta = generator_costs.loc[(y, 'YWPS4', s), ['RHO', 'DELTA']].values

            # Difference between marginal cost in given interval and all generator marginal costs for that interval
            diff = generator_mc['net_marginal_cost'].subtract(p).abs()

            # Extract generator ID and absolute cost difference
            g, cost_diff = diff.idxmin(), diff.min()

            # Details of the price setting generator
            cols = ['EMISSIONS_RATE', 'baseline', 'permit_price', 'C_MC', 'net_marginal_cost', 'scaling_factor']
            emissions_rate, baseline, permit_price, marginal_cost, net_marginal_cost, scaling_factor = generator_costs.loc[g, cols]

            # Compute normalised price and cost differences
            price_normalised = p / (delta * rho * scaling_factor)
            cost_diff_normalised = cost_diff / (delta * rho)
            net_marginal_cost_normalised = net_marginal_cost / (delta * rho * scaling_factor)

            return (g[1], p, price_normalised, cost_diff, cost_diff_normalised, emissions_rate, baseline, permit_price,
                    marginal_cost, net_marginal_cost, net_marginal_cost_normalised)

        # Get price setting generator information
        ps = zone_price.apply(get_price_setter, axis=1)

        # Convert column of tuples to DataFrame with separate columns
        columns = ['generator', 'price_abs', 'price_normalised', 'difference_abs', 'difference_normalised',
                   'emissions_rate', 'baseline', 'permit_price', 'marginal_cost', 'net_marginal_cost',
                   'net_marginal_cost_normalised']
        ps = pd.DataFrame(ps.to_list(), columns=columns, index=zone_price.index)

        return ps


if __name__ == '__main__':
    # Path and filename
    results_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '3_model', 'linear', 'output', 'local')
    results_filename = 'carbon_tax_fixed_capacity_case.pickle'

    # Object used to parse model results and identify price setting generators
    price_setter = PriceSetter()

    # DataFrame of price setting generators
    psg = price_setter.get_price_setting_generators_from_model_results(results_directory, results_filename)

    # Save price setting generator results
    with open(os.path.join(os.path.dirname(__file__), 'output', 'price_setting_generators.pickle'), 'wb') as f:
        pickle.dump(psg, f)
