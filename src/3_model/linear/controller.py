"""Controller used to run primal, dual, and MPPDC programs"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'components'))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, '4_analysis'))

from cases import ModelCases
from targets import Targets

import matplotlib.pyplot as plt


if __name__ == '__main__':
    log_file_name = 'case_logger'
    output_directory = os.path.join(os.path.dirname(__file__), 'output', 'remote')

    # Object used to run model cases
    cases = ModelCases(output_directory, log_file_name)

    # Object used to assist in calculation of target trajectories (e.g. price weights)
    targets = Targets()

    # Common model parameters
    start, end, scenarios = 2016, 2040, 5

    # Year when scheme transitions to a Refunded Emissions Payment (REP) scheme
    transition_year = 2021

    # Permit prices for carbon pricing scenarios
    permit_prices_model = {y: float(40) for y in range(start, end + 1)}

    # Cumulative scheme revenue cannot go below this envelope
    scheme_revenue_envelope_lo = {y: targets.get_envelope(-100e6, 4, start, y) if y < transition_year else float(0)
                                  for y in range(start, end + 1)}

    # Price weights
    scheme_price_weights = {y: targets.get_envelope(1, 2, start, y) if y <= transition_year else 0
                            for y in range(start, end + 1)}

    # Define case parameters and run model
    case_params = {'rep_filename': 'rep_case.pickle',
                   'revenue_envelope_lo': scheme_revenue_envelope_lo,
                   'price_weights': scheme_price_weights,
                   'transition_year': transition_year}

    # Run BAU case
    # r_bau = cases.run_bau_case(start, end, scenarios, output_directory)
    cases.run_bau_case(start, end, scenarios, output_directory)

    # Run REP case
    # r_rep = cases.run_rep_case(start, end, scenarios, permit_prices_model, output_directory)
    cases.run_rep_case(start, end, scenarios, permit_prices_model, output_directory)

    # Run price case targeting model using MPPDC model
    # r_ptm = cases.run_price_smoothing_mppdc_case(case_params, output_directory)
    cases.run_price_smoothing_mppdc_case(case_params, output_directory)

    # Run price case targeting model using auxiliary model
    # r_pth = cases.run_price_smoothing_heuristic_case(case_params, output_directory)
    cases.run_price_smoothing_heuristic_case(case_params, output_directory)

    # # BAU prices
    # p_bau = cases.analysis.get_year_average_price(r_bau['results']['PRICES'], factor=-1)
    #
    # # Carbon tax prices
    # p_ct = cases.analysis.get_year_average_price(r_rep['results']['stage_1_carbon_tax']['PRICES'], factor=-1)
    #
    # # MPPDC price targeting prices
    # p_m = cases.analysis.get_year_average_price(r_ptm['results']['stage_3_price_targeting'][1]['lamb'], factor=1)
    #
    # # Heuristic price targeting prices
    # p_h = cases.analysis.get_year_average_price(r_pth['results']['stage_3_price_targeting'][1]['primal']['PRICES'],
    #                                             factor=-1)
    #
    # # Baselines
    # fig, ax = plt.subplots()
    # ax.plot(list(r_ptm['mppdc_model'].baseline.get_values().values()))
    # ax.plot(list(r_pth['auxiliary_model'].baseline.get_values().values()))
    # plt.show()
