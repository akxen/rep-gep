"""Analyse output from BAU case"""

import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt

from utils import ParseOutput

if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), os.path.pardir, '3_model', 'linear', 'output', 'remote')
    bau_results = pd.read_pickle(os.path.join(results_dir, 'primal_bau_results.pickle'))

    # Plot emissions intensities
    pd.DataFrame.from_dict(bau_results['YEAR_EMISSIONS_INTENSITY'], orient='index').plot()
    plt.show()

    df_c = (pd.Series(bau_results['x_c']).rename_axis(['unit', 'year'])
            .reset_index(name='capacity').pivot(index='year', columns='unit', values='capacity').cumsum())

    # Plot of all candidate units with positive capacity installed over model horizon
    df_c.loc[:, ~df_c.eq(0).all(axis=0)].plot()

    for c in df_c.loc[:, ~df_c.eq(0).all(axis=0)].columns:
        df_c.loc[:, c].plot(title=c)
        plt.show()

    # Scaling factor used to convert average prices to nominal values
    scaling_factor = pd.Series({y: (1 + 0.06) ** (y - 2016) for y in range(2016, 2051)})

    # Plotting prices
    res = ParseOutput()
    primal_prices = res.parse_prices(mode='primal')
    primal_prices.mul(scaling_factor).plot()
    plt.show()



