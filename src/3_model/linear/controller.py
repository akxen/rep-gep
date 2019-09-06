"""Controller used to run primal, dual, and MPPDC programs"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'components'))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, '4_analysis'))

from cases import ModelCases
from analysis import AnalyseResults


if __name__ == '__main__':
    output_directory = os.path.join(os.path.dirname(__file__), 'output', 'local')
    log_file_name = 'case_logger'
    cases = ModelCases(output_directory, log_file_name)
    analysis = AnalyseResults(output_directory)

    start_year = 2016
    final_model = 2040
    scenarios_per_year = 5
    permit_prices = {y: float(40) for y in range(start_year, final_model + 1)}

    # Business-as-usual case
    bau_res = cases.run_bau_case(output_directory, final_model, scenarios_per_year, mode='primal')

    # # BAU average price in first year (used in price smoothing algorithm)
    # bau_prices = analysis.get_year_average_price('primal_bau_case.pickle')
    # initial_average_price = bau_prices.loc[2016, 'average_price_real']
    #
    # # Run REP case algorithm
    # # rep_res = cases.run_rep_case(output_directory, final_model, scenarios_per_year, permit_prices)
    #
    # # Run MPPDC price smoothing algorithm - non-negative revenue constraint
    # non_neg_rev = cases.run_price_smoothing_mppdc_case(output_directory, final_model, scenarios_per_year,
    #                                                    permit_prices, initial_average_price, 'non_negative_revenue')
    #
    # # Run primal model with baselines and permit prices obtained from MPPDC to check solution
    # non_neg_rev_check = cases.run_primal_to_check_mppdc_solution(output_directory, final_model,
    #                                                              scenarios_per_year, 'non_negative_revenue')
    #
    # # Max price difference
    # non_neg_rev_price_diff = cases.compare_absolute_values(non_neg_rev_check['PRICES'],
    #                                                        non_neg_rev['stage_3_mppdc']['lamb'])
    # print(f'Max price difference for non-negative revenue case: {non_neg_rev_price_diff[1]}')
    #
    # # Run MPPDC price smoothing algorithm - neutral revenue constraint
    # neutral_rev = cases.run_price_smoothing_mppdc_case(output_directory, final_model, scenarios_per_year,
    #                                                    permit_prices, initial_average_price, 'neutral_revenue')
    #
    # # Run primal model with baselines and permit prices obtained from MPPDC to check solution
    # neutral_rev_check = cases.run_primal_to_check_mppdc_solution(output_directory, final_model,
    #                                                              scenarios_per_year, 'neutral_revenue')
    #
    # # Max price difference
    # neutral_rev_price_diff = cases.compare_absolute_values(neutral_rev_check['PRICES'],
    #                                                        neutral_rev['stage_3_mppdc']['lamb'])
    # print(f'Max price difference for neutral revenue case: {neutral_rev_price_diff[1]}')
    #
    # # Lower bound on cumulative scheme revenue over model horizon
    # scheme_revenue_lb = float(-50e6)
    #
    # # Run MPPDC price smoothing algorithm - neutral revenue constraint with lower bound constraint on cumulative revenue
    # neutral_rev_lb = cases.run_price_smoothing_mppdc_case(output_directory, final_model, scenarios_per_year,
    #                                                       permit_prices, initial_average_price,
    #                                                       'neutral_revenue_lower_bound', scheme_revenue_lb)
    #
    # # Run primal model with baselines and permit prices obtained from MPPDC to check solution
    # neutral_rev_lb_check = cases.run_primal_to_check_mppdc_solution(output_directory, final_model,
    #                                                                 scenarios_per_year, 'neutral_revenue_lower_bound')
    #
    # # Max price difference
    # neutral_rev_lb_price_diff = cases.compare_absolute_values(neutral_rev_lb_check['PRICES'],
    #                                                           neutral_rev_lb['stage_3_mppdc']['lamb'])
    # print(f'Max price difference for neutral revenue case with lower revenue bound: {neutral_rev_lb_price_diff[1]}')
    #
    # # Transitional scheme - relax revenue neutrality for several years following introduction, then implement REP scheme
    # transition = cases.run_price_smoothing_mppdc_case(output_directory, final_model, scenarios_per_year, permit_prices,
    #                                                   initial_average_price, 'transition')
    #
    # # Run primal model with baselines and permit prices obtained from MPPDC to check solution
    # transition_check = cases.run_primal_to_check_mppdc_solution(output_directory, final_model, scenarios_per_year,
    #                                                             'transition')
    #
    # # Max price difference
    # transition_diff = cases.compare_absolute_values(transition_check['PRICES'], transition['stage_3_mppdc']['lamb'])
    # print(f'Max price difference for scheme transition case: {transition_diff[1]}')
