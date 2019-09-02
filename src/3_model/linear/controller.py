"""Controller used to run primal, dual, and MPPDC programs"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'components'))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, '4_analysis'))

from components import gep
from targets import Targets


if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(__file__), 'output', 'remote')
    final_year = 2050
    scenarios_per_year = 10
    gep.setup_logger(output_dir, 'controller')

    # BAU case
    # primal_results = gep.run_bau_case(output_dir, final_year, scenarios_per_year, mode='primal')
    # mppdc_results = gep.run_bau_case(output_dir, final_year, scenarios_per_year, mode='mppdc')

    # Object used to handle emissions targets - uses BAU results to formulate targets
    target = Targets(output_dir)

    # # Emissions cap - target is 50% of BAU emissions
    # emissions_cap = target.get_cumulative_emissions_target('primal_bau_results.pickle', 0.5)
    # emissions_cap_results = gep.run_cumulative_emissions_cap_case(output_dir, final_year, scenarios_per_year,
    #                                                               emissions_cap)
    #
    # # # Interim target - replicate emissions profile based on cumulative cap
    # interim_emissions_cap = target.get_interim_emissions_target('cumulative_emissions_cap_results.pickle')
    # interim_emissions_cap_results = gep.run_interim_emissions_cap_case(output_dir, final_year, scenarios_per_year,
    #                                                                    interim_emissions_cap)

    # Carbon tax - use dual variable from cumulative emissions cap case as permit price
    permit_prices = {y: float(target.get_cumulative_emissions_cap_carbon_price()) for y in range(2016, final_year + 1)}
    carbon_tax_results = gep.run_carbon_tax_case(output_dir, final_year, scenarios_per_year, permit_prices)

    # # Non-negative scheme revenue
    # non_neg_revenue_results = gep.run_algorithm(output_dir, final_year, scenarios_per_year,
    #                                             target_emissions_intensities,
    #                                             baseline_tol=0.05, permit_price_tol=10, permit_price_cap=500,
    #                                             case='price_smoothing_non_negative_revenue')
    #
    # # Revenue neutral scheme
    # neutral_revenue_results = gep.run_algorithm(output_dir, final_year, scenarios_per_year,
    #                                             target_emissions_intensities,
    #                                             baseline_tol=0.05, permit_price_tol=10, permit_price_cap=500,
    #                                             case='price_smoothing_neutral_revenue')
    #
    # # Revenue neutral scheme with lower bound
    # neutral_revenue_with_lb_results = gep.run_algorithm(output_dir, final_year, scenarios_per_year,
    #                                                     target_emissions_intensities, baseline_tol=0.05,
    #                                                     permit_price_tol=10, permit_price_cap=500,
    #                                                     case='price_smoothing_neutral_revenue_lower_bound')
    #
    # # Refunded Emissions Payment (REP) scheme
    # rep_results = gep.run_algorithm(output_dir, final_year, scenarios_per_year, target_emissions_intensities,
    #                                 baseline_tol=0.05, permit_price_tol=10, permit_price_cap=500,
    #                                 case='rep')
    #
    # # Transitional scheme
    # transition_results = gep.run_algorithm(output_dir, final_year, scenarios_per_year, target_emissions_intensities,
    #                                        baseline_tol=0.05, permit_price_tol=10, permit_price_cap=500,
    #                                        case='transition')
