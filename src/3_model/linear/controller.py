"""Controller used to run primal, dual, and MPPDC programs"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'components'))

from components import gep


if __name__ == '__main__':
    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')

    # Final year in model horizon
    final_year = 2018

    # Run business-as-usual scenario
    primal_bau_results = gep.run_bau(output_dir, final_year=final_year, scenarios_per_year=10, mode='primal')
    mppdc_bau_results = gep.run_bau(output_dir, final_year=final_year, scenarios_per_year=10, mode='mppdc')

    # Parameters for price smoothing model
    baselines = {y: 0.8 for y in range(2016, final_year + 1)}
    permit_prices = {y: 30 for y in range(2016, final_year + 1)}

    # Run price smoothing model
    price_smoothing_results = gep.run_mppdc_price_smoothing(output_dir, baselines, permit_prices, final_year=final_year,
                                                            scenarios_per_year=10)
