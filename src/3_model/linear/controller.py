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
    final_year = 2040
    scenarios_per_year = 5
    permit_prices = {y: float(40) for y in range(start_year, final_year + 1)}

    # Business-as-usual case
    bau_res = cases.run_bau_case(output_directory, final_year, scenarios_per_year, mode='primal')

    # Carbon tax
    carbon_tax_res = cases.run_carbon_tax_case(output_directory, final_year, scenarios_per_year, permit_prices)
