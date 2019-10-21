"""Controller used to run primal, dual, and MPPDC programs"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'components'))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, '4_analysis'))

from cases import ModelCases
from targets import Targets


if __name__ == '__main__':
    log_file_name = 'case_logger'
    output_directory = os.path.join(os.path.dirname(__file__), 'output', 'local')

    # Object used to run model cases
    cases = ModelCases(output_directory, log_file_name)

    # Object used to assist in calculation of target trajectories (e.g. price weights)
    targets = Targets()

    # Common model parameters
    start, end, scenarios = 2016, 2040, 5

    # Permit prices for carbon pricing scenarios
    permit_prices_model = {y: float(40) for y in range(start, end + 1)}

    # Define case parameters and run model
    case_params = {'rep_filename': 'rep_case.pickle', 'mode': 'price_change_minimisation'}

    # Run BAU case
    # cases.run_bau_case(start, end, scenarios, output_directory)

    # Run REP case
    # cases.run_rep_case(start, end, scenarios, permit_prices_model, output_directory)

    # Run price targeting models with different transition years
    for transition_year in [2021, 2028]:
        print(f'Running models with transition year: {transition_year}')

        # Update transition year
        case_params['transition_year'] = transition_year

        # Update scheme price weights to be used in objective function
        case_params['price_weights'] = {y: 1.0 if y <= transition_year else 0.0 for y in range(start, end + 1)}

        # Target prices using auxiliary model
        cases.run_price_smoothing_heuristic_case(case_params, output_directory)

        # Target prices using MPPDC model
        cases.run_price_smoothing_mppdc_case(case_params, output_directory)
