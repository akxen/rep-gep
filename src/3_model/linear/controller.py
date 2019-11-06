"""Controller used to run primal, dual, and MPPDC programs"""

import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'components'))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, '4_analysis'))

from cases import ModelCases


if __name__ == '__main__':
    log_file_name = 'case_logger'
    output_directory = os.path.join(os.path.dirname(__file__), 'output', 'remote')

    # Object used to run model cases
    cases = ModelCases(output_directory, log_file_name)

    # Common model parameters
    start, end, scenarios = 2016, 2030, 5

    # Define case parameters and run model
    case_params = {'mode': 'bau_deviation_minimisation'}

    # Run BAU case
    cases.run_bau_case(start, end, scenarios, output_directory)

    # Run models with different carbon prices
    for c in range(5, 101, 5):
        # Permit prices to be used in REP and price targeting models
        permit_prices_model = {y: float(c) for y in range(start, end + 1)}

        # Run REP case with given permit price
        cases.run_rep_case(start, end, scenarios, permit_prices_model, output_directory)

        # Run price targeting models with different transition years
        for transition_year in [2020, 2025, 2030]:
            print(f'Running models with transition year: {transition_year}')

            # Update transition year
            case_params['transition_year'] = transition_year
            case_params['carbon_price'] = c
            case_params['rep_filename'] = f'rep_cp-{c:.0f}.pickle'

            # Update scheme price weights to be used in objective function
            case_params['price_weights'] = {y: 1.0 if y <= transition_year else 0.0 for y in range(start, end + 1)}

            # Target prices using auxiliary model
            cases.run_price_smoothing_heuristic_case(case_params, output_directory)

            # Only run MPPDC if following condition(s) met. Used to compare MPPDC to heuristic solution.
            if c in [25, 50, 75, 100]:
                # Target prices using MPPDC model
                cases.run_price_smoothing_mppdc_case(case_params, output_directory)
