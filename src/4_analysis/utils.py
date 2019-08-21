"""Utilities used to parse model results"""

import os
import sys
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, '3_model', 'linear', 'components'))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, '3_model', 'linear', 'components', 'base'))

import pandas as pd
from data import ModelData


class ParseOutput:
    def __init__(self):
        self.data = ModelData()

    @staticmethod
    def load_results(filename):
        """Load model results"""

        # Path to model results
        path = os.path.join(os.path.dirname(__file__), os.path.pardir, '3_model', 'linear', 'output', 'remote', filename)

        # Load results
        with open(path, 'rb') as f:
            results = pickle.load(f)

        return results

    def parse_prices(self, mode):
        """Parse prices obtained from primal model"""
        
        # Get price information - keys will be different for primal and MPPDC models
        if mode == 'primal':
            # Load results
            results = self.load_results('primal_bau_results.pickle')

            # Extract price information
            prices = results['PRICES']

            # Prices
            df_p = (pd.Series(prices).rename_axis(['zone', 'year', 'scenario', 'interval'])
                    .to_frame(name='price').mul(-1))

        elif mode == 'mppdc':
            # Load results
            results = self.load_results('mppdc_bau_results.pickle')

            # Extract price information
            prices = results['lamb']

            # Prices
            df_p = pd.Series(prices).rename_axis(['zone', 'year', 'scenario', 'interval']).to_frame(name='price')

        else:
            raise Exception(f'Unexpected mode: {mode}')

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
        df_c['scaling_factor'] = df_c.apply(lambda x: (1 + (1 / self.data.WACC)) if x.name[0] == final_year else 1,
                                            axis=1)

        # Scaled prices
        df_c['price_scaled'] = df_c['price'].div(df_c['duration']).div(df_c['scaling_factor'])

        # Interval revenue
        df_c['revenue'] = df_c['price_scaled'].mul(df_c['demand'])

        # Revenue scaled by scenario duration
        df_c['revenue_scaled'] = df_c['revenue'].mul(df_c['duration'])

        # Demand scaled by scenario duration
        df_c['demand_scaled'] = df_c['demand'].mul(df_c['duration'])

        # Average price = total revenue collected from wholesale electricity sales / total energy demand
        df_c['average_price'] = (df_c.groupby(['year'])['revenue_scaled'].sum()
                                 .div(df_c.groupby(['year'])['demand_scaled'].sum()))

        # Average price for a given year
        year_average_price = (df_c.groupby(['year'])['revenue_scaled'].sum()
                              .div(df_c.groupby(['year'])['demand_scaled'].sum()))

        return year_average_price


if __name__ == '__main__':
    # Object used to parse model results
    res = ParseOutput()

    # Average yearly prices obtained from primal and MPPDC models
    primal_prices = res.parse_prices(mode='primal')
    # mppdc_prices = res.parse_prices(mode='mppdc')
