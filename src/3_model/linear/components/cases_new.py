"""Generation expansion planning cases"""

import os
import pickle
import logging

from pyomo.environ import *

from gep import MPPDCModel, Primal, Dual


class ModelCases:
    def __init__(self, output_dir, log_name):
        logging.basicConfig(filename=os.path.join(output_dir, f'{log_name}.log'), filemode='w',
                            format='%(asctime)s %(name)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

    @staticmethod
    def algorithm_logger(function, message, print_message=False):
        """Write message to logfile and optionally print output"""

        # Get logging object
        logger = logging.getLogger(__name__)

        # Message to write to logfile
        log_message = f'{function} - {message}'

        if print_message:
            print(log_message)
            logger.info(log_message)
        else:
            logger.info(log_message)

    @staticmethod
    def extract_result(m, component_name):
        """Extract values associated with model components"""

        model_component = m.__getattribute__(component_name)

        if type(model_component) == pyomo.core.base.expression.IndexedExpression:
            return {k: model_component[k].expr() for k in model_component.keys()}

        elif type(model_component) == pyomo.core.base.expression.SimpleExpression:
            return model_component.expr()

        elif type(model_component) == pyomo.core.base.var.IndexedVar:
            return model_component.get_values()

        elif type(model_component) == pyomo.core.base.param.IndexedParam:
            return {k: v for k, v in model_component.items()}

        else:
            raise Exception(f'Unexpected model component: {component_name}')

    @staticmethod
    def save_results(results, output_dir, filename):
        """Save model results"""

        with open(os.path.join(output_dir, filename), 'wb') as f:
            pickle.dump(results, f)

    @staticmethod
    def run_fixed_policy(final_year, scenarios_per_year, permit_prices, baselines):
        """Run primal model with fixed policy parameters"""

        # Initialise object and model used to run primal model
        primal = Primal(final_year, scenarios_per_year)
        m = primal.construct_model()

        # Fix permit prices and baselines to specified levels
        for y in m.Y:
            m.permit_price[y].fix(permit_prices[y])
            m.baseline[y].fix(baselines[y])

        # Solve primal model with fixed policy parameters
        m, status = primal.solve_model(m)

        # Fix investment decisions - re-solve to obtain correct prices
        m.x_c.fix()
        m, status = primal.solve_model(m)

        # Unfix investment decisions (in case following solve required)
        m.x_c.unfix()

        return m, status

    def run_bau_case(self, output_dir, first_year, final_year, scenarios_per_year):
        """Run business-as-usual case"""

        # Permit prices and emissions intensity baselines for BAU case (all 0)
        permit_prices = {y: float(0) for y in range(first_year, final_year + 1)}
        baselines = {y: float(0) for y in range(first_year, final_year + 1)}

        # Run model
        m, status = self.run_fixed_policy(final_year, scenarios_per_year, permit_prices, baselines)

        # Results to extract
        result_keys = ['x_c', 'p', 'p_V', 'p_in', 'p_out', 'p_L', 'baseline', 'permit_price', 'YEAR_EMISSIONS',
                       'YEAR_EMISSIONS_INTENSITY', 'YEAR_SCHEME_REVENUE', 'TOTAL_SCHEME_REVENUE', 'C_MC', 'ETA',
                       'DELTA', 'RHO', 'EMISSIONS_RATE']

        # Model results
        results = {k: self.extract_result(m, k) for k in result_keys}

        # Add dual variable from power balance constraint
        results['PRICES'] = {k: m.dual[m.POWER_BALANCE[k]] for k in m.POWER_BALANCE.keys()}

        # Save results
        filename = 'bau_case.pickle'
        self.save_results(results, output_dir, filename)

        return results

    def run_carbon_tax_case(self, output_dir, first_year, final_year, scenarios_per_year, permit_prices):
        """Run carbon tax scenario"""

        # Permit prices and emissions intensity baselines for BAU case (all 0)
        baselines = {y: float(0) for y in range(first_year, final_year + 1)}

        # Run model
        m, status = self.run_fixed_policy(final_year, scenarios_per_year, permit_prices, baselines)

        # Results to extract
        result_keys = ['x_c', 'p', 'p_V', 'p_in', 'p_out', 'p_L', 'baseline', 'permit_price', 'YEAR_EMISSIONS',
                       'YEAR_EMISSIONS_INTENSITY', 'YEAR_SCHEME_REVENUE', 'TOTAL_SCHEME_REVENUE', 'C_MC', 'ETA',
                       'DELTA', 'RHO', 'EMISSIONS_RATE']

        # Model results
        results = {k: self.extract_result(m, k) for k in result_keys}

        # Add dual variable from power balance constraint
        results['PRICES'] = {k: m.dual[m.POWER_BALANCE[k]] for k in m.POWER_BALANCE.keys()}

        # Save results
        filename = 'carbon_tax_case.pickle'
        self.save_results(results, output_dir, filename)

        return results

    def run_rep_case(self, output_dir, first_year, final_year, scenarios_per_year, permit_prices):
        """Run carbon tax scenario"""

        # Results to extract
        result_keys = ['x_c', 'p', 'p_V', 'p_in', 'p_out', 'p_L', 'baseline', 'permit_price', 'YEAR_EMISSIONS',
                       'YEAR_EMISSIONS_INTENSITY', 'YEAR_SCHEME_REVENUE', 'TOTAL_SCHEME_REVENUE', 'C_MC', 'ETA',
                       'DELTA', 'RHO', 'EMISSIONS_RATE', 'YEAR_SCHEME_EMISSIONS_INTENSITY']

        # First run carbon tax case
        baselines = {y: float(0) for y in range(first_year, final_year + 1)}

        # Run model (carbon tax case)
        m, status = self.run_fixed_policy(final_year, scenarios_per_year, permit_prices, baselines)

        # Model results
        carbon_tax_results = {k: self.extract_result(m, k) for k in result_keys}

        # Add dual variable from power balance constraint
        carbon_tax_results['PRICES'] = {k: m.dual[m.POWER_BALANCE[k]] for k in m.POWER_BALANCE.keys()}

        # Update baselines so they = emissions intensity of output from participating generators
        rep_baselines = carbon_tax_results['YEAR_SCHEME_EMISSIONS_INTENSITY']

        # Re-run model with new baselines
        m, status = self.run_fixed_policy(final_year, scenarios_per_year, permit_prices, rep_baselines)

        # Model results
        rep_results = {k: self.extract_result(m, k) for k in result_keys}

        # Add dual variable from power balance constraint
        rep_results['PRICES'] = {k: m.dual[m.POWER_BALANCE[k]] for k in m.POWER_BALANCE.keys()}

        # Combine results into single dictionary
        results = {'stage_1_carbon_tax': carbon_tax_results, 'stage_2_rep': rep_results}

        # Save results
        filename = 'rep_case.pickle'
        self.save_results(results, output_dir, filename)

        return results


if __name__ == '__main__':
    log_file_name = 'case_logger'
    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, 'output', 'local')

    # Object used to run model cases
    cases = ModelCases(output_directory, log_file_name)

    # Common model parameters
    start_year = 2016
    end_year = 2040
    scenarios = 5

    # Run business-as-usual case
    # r_bau = cases.run_bau_case(output_directory, start_year, end_year, scenarios)

    # Permit prices for carbon pricing scenarios
    permit_prices_model = {y: float(40) for y in range(start_year, end_year + 1)}

    # Run carbon tax case
    # r_ct = cases.run_carbon_tax_case(output_directory, start_year, end_year, scenarios, permit_prices_model)

    # Run Refunded Emissions Payment Scheme
    r_rep = cases.run_rep_case(output_directory, start_year, end_year, scenarios, permit_prices_model)
