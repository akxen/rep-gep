"""Controller used to run primal, dual, and MPPDC programs"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'components'))

from components import gep


if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(__file__), 'output', 'remote')
    final_year = 2018
    scenarios_per_year = 2
    gep.setup_logger('controller')

    # BAU case
    primal_results = gep.run_bau_case(output_dir, final_year, scenarios_per_year, mode='primal')
    # mppdc_results = gep.run_bau_case(output_dir, final_year, scenarios_per_year, mode='mppdc')
    #
    # # Carbon tax
    # target_emissions_intensities = {y: 0.7 for y in range(2016, final_year + 1)}
    # carbon_tax_results = gep.run_carbon_tax_case(output_dir, final_year, scenarios_per_year,
    #                                              target_emissions_intensities,
    #                                              permit_price_tol=2)
    #
    # # Non-negative scheme revenue
    # non_neg_revenue_results = gep.run_algorithm(output_dir, final_year, scenarios_per_year,
    #                                             target_emissions_intensities,
    #                                             baseline_tol=0.05, permit_price_tol=10,
    #                                             case='price_smoothing_non_negative_revenue')
    #
    # # Revenue neutral scheme
    # neutral_revenue_results = gep.run_algorithm(output_dir, final_year, scenarios_per_year,
    #                                             target_emissions_intensities,
    #                                             baseline_tol=0.05, permit_price_tol=10,
    #                                             case='price_smoothing_neutral_revenue')
    #
    # # Revenue neutral scheme with lower bound
    # neutral_revenue_with_lb_results = gep.run_algorithm(output_dir, final_year, scenarios_per_year,
    #                                                     target_emissions_intensities, baseline_tol=0.05,
    #                                                     permit_price_tol=10,
    #                                                     case='price_smoothing_neutral_revenue_lower_bound')
    #
    # # Refunded Emissions Payment (REP) scheme
    # rep_results = gep.run_algorithm(output_dir, final_year, scenarios_per_year, target_emissions_intensities,
    #                                 baseline_tol=0.05, permit_price_tol=10,
    #                                 case='rep')
    #
    # # Transitional scheme
    # transition_results = gep.run_algorithm(output_dir, final_year, scenarios_per_year, target_emissions_intensities,
    #                                        baseline_tol=0.05, permit_price_tol=10,
    #                                        case='transition')
