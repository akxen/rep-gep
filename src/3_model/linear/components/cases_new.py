"""Generation expansion planning cases"""

import os
import sys
import copy
import pickle
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, '4_analysis'))

from pyomo.environ import *

from gep import MPPDCModel, Primal, Dual
from auxiliary import BaselineUpdater
from analysis import AnalyseResults


class ModelCases:
    def __init__(self, output_dir, log_name):
        logging.basicConfig(filename=os.path.join(output_dir, f'{log_name}.log'), filemode='w',
                            format='%(asctime)s %(name)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

        self.analysis = AnalyseResults()

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
        result_keys = ['x_c', 'p', 'p_V', 'p_in', 'p_out', 'q', 'p_L', 'baseline', 'permit_price', 'YEAR_EMISSIONS',
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

    def run_price_smoothing_heuristic_case(self, output_dir, rep_filename, transition_year):
        """Smooth prices over entire model horizon using approximated price functions"""

        # Results to extract
        result_keys = ['x_c', 'p', 'p_V', 'p_in', 'p_out', 'p_L', 'baseline', 'permit_price', 'YEAR_EMISSIONS',
                       'YEAR_EMISSIONS_INTENSITY', 'YEAR_SCHEME_REVENUE', 'TOTAL_SCHEME_REVENUE', 'C_MC', 'ETA',
                       'DELTA', 'RHO', 'EMISSIONS_RATE', 'YEAR_SCHEME_EMISSIONS_INTENSITY']

        # Load REP results - used as starting point for heuristic and MPPDC solution methods
        with open(os.path.join(output_directory, rep_filename), 'rb') as f:
            rep_results = pickle.load(f)

        # Carbon tax and REP results
        carbon_tax_results, rep_results = rep_results['stage_1_carbon_tax'], rep_results['stage_2_rep']

        # Extract permit prices
        permit_prices = rep_results['permit_price']

        # Final year in model horizon
        first_year, final_year = min(permit_prices.keys()), max(permit_prices.keys())

        # Extract scenarios per year
        scenarios_per_year = len([s for y, s in rep_results['RHO'].keys() if y == final_year])

        # Get BAU initial price
        initial_price = self.get_bau_initial_price(output_dir, first_year)

        # Baseline updater
        baseline = BaselineUpdater(final_year, scenarios_per_year)

        # Container for iteration results
        iteration_results = {}

        stop_flag = False

        # Iteration counter
        i = 1

        # Results to input into price targeting model. Set to REP results for first iteration.
        result_input = rep_results

        while not stop_flag:
            # Identify price setting generators
            psg = baseline.prices.get_price_setting_generators_from_model_results(result_input)

            # Construct model
            baseline_model = baseline.construct_model(psg)

            # Update parameters
            baseline_model = baseline.update_parameters(baseline_model, result_input)

            # Update initial price (set to BAU price in first year)
            baseline_model.INITIAL_AVERAGE_PRICE = float(initial_price)

            # Activate constraints corresponding to different cases
            # baseline_model.REVENUE_NEUTRAL_HORIZON.activate()
            baseline_model.REVENUE_NEUTRAL_TRANSITION.activate()
            baseline_model.TRANSITION_YEAR = transition_year
            baseline_model.reconstruct()

            for y in range(baseline_model.TRANSITION_YEAR.value, final_year + 1):
                baseline_model.REVENUE_NEUTRAL_YEAR[y].activate()

            # Solve model
            baseline_model, status = baseline.solve_model(baseline_model)

            # Price targeting baselines
            pt_baselines = baseline_model.baseline.get_values()

            # Re-solve primal program with updated baselines
            m, status = self.run_fixed_policy(final_year, scenarios_per_year, permit_prices, pt_baselines)

            # Get results
            pt_results = copy.deepcopy({k: self.extract_result(m, k) for k in result_keys})

            # Add dual variable from power balance constraint
            pt_results['PRICES'] = copy.deepcopy({k: m.dual[m.POWER_BALANCE[k]] for k in m.POWER_BALANCE.keys()})

            # Add results to iteration results container
            iteration_results[i] = pt_results

            # Check if stop criterion satisfied
            cap_diff = max([abs(result_input['x_c'][k] - pt_results['x_c'][k]) for k in result_input['x_c'].keys()])
            print(f'{i}: Maximum capacity difference = {cap_diff} MW')
            if cap_diff < 5:
                stop_flag = True

            i += 1

        # Combine results into single dictionary
        results = {'stage_1_carbon_tax': carbon_tax_results, 'stage_2_rep': rep_results,
                   'stage_3_price_targeting': iteration_results}

        # Rename file
        filename = f'price_targeting_heuristic_case.pickle'
        self.save_results(results, output_dir, filename)

        return results

    def run_price_smoothing_mppdc_case(self, output_dir, rep_filename, transition_year):
        """Run case to smooth prices over model horizon, subject to total revenue constraint"""

        # Results to extract
        result_keys = ['x_c', 'p', 'p_V', 'p_in', 'p_out', 'p_L', 'q', 'baseline', 'permit_price', 'lamb',
                       'YEAR_EMISSIONS', 'YEAR_EMISSIONS_INTENSITY', 'YEAR_SCHEME_EMISSIONS_INTENSITY',
                       'YEAR_SCHEME_REVENUE', 'TOTAL_SCHEME_REVENUE']

        # Load REP results
        with open(os.path.join(output_dir, rep_filename), 'rb') as f:
            results = pickle.load(f)

        # Carbon tax and REP results
        carbon_tax_results, rep_results = results['stage_1_carbon_tax'], results['stage_2_rep']

        # Extract permit prices
        permit_prices = rep_results['permit_price']

        # First and final year in model horizon
        first_year, final_year = min(permit_prices.keys()), max(permit_prices.keys())

        # Get BAU initial price
        initial_price = self.get_bau_initial_price(output_dir, first_year)

        # Extract scenarios per year
        scenarios_per_year = len([s for y, s in rep_results['RHO'].keys() if y == final_year])

        # Container for iteration results
        iteration_results = {}

        stop_flag = False

        # Iteration counter
        i = 1

        # Construct MPPDC
        mppdc = MPPDCModel(final_year, scenarios_per_year)
        mppdc_model = mppdc.construct_model(include_primal_constraints=False)

        # Set average price in year prior to model start (assume same first year average price in BAU case)
        mppdc_model.YEAR_AVERAGE_PRICE_0 = float(initial_price)

        # Primal variables to fix. Set equal to results obtained from REP case in first iteration.
        primal_variables = ['x_c', 'p', 'p_in', 'p_out', 'q', 'p_V', 'p_L', 'permit_price']
        fixed_variables = {k: rep_results[k] for k in primal_variables}

        while not stop_flag:
            # Fix MPPDC variables
            mppdc_model = mppdc.fix_variables(mppdc_model, fixed_variables)

            # Set the transition year
            mppdc_model.TRANSITION_YEAR = transition_year

            # Activate year revenue neutrality requirement for all years
            mppdc_model.YEAR_NET_SCHEME_REVENUE_NEUTRAL_CONS.activate()

            # Deactivate year revenue neutrality requirement for years before transition year
            for y in range(2016, transition_year):
                mppdc_model.YEAR_NET_SCHEME_REVENUE_NEUTRAL_CONS[y].deactivate()

            # Reconstruct and enforce revenue neutrality over the transition period
            mppdc_model.TRANSITION_NET_SCHEME_REVENUE_NEUTRAL_CONS.reconstruct()
            mppdc_model.TRANSITION_NET_SCHEME_REVENUE_NEUTRAL_CONS.activate()

            # Solve model
            mppdc_model, solve_status = mppdc.solve_model(mppdc_model)

            # Get results
            pt_results = copy.deepcopy({k: self.extract_result(mppdc_model, k) for k in result_keys})

            # Extract model results
            iteration_results[i] = pt_results

            # Check if stop criterion satisfied
            cap_diff = max([abs(fixed_variables['x_c'][k] - pt_results['x_c'][k]) for k in fixed_variables['x_c'].keys()])
            print(f'{i}: Maximum capacity difference = {cap_diff} MW')
            if cap_diff < 5:
                stop_flag = True

            # Update dictionary of fixed variables to be used in next iteration
            fixed_variables = {k: pt_results[k] for k in primal_variables}

        # Combine results into a single dictionary
        combined_results = {**results, 'stage_3_price_targeting': iteration_results}

        # Save results
        self.save_results(combined_results, output_dir, f'price_targeting_mppdc_case.pickle')

        return combined_results

    def get_bau_initial_price(self, output_dir, first_year):
        """Get BAU price in first year"""

        # Load BAU results
        with open(os.path.join(output_dir, 'bau_case.pickle'), 'rb') as f:
            bau_results = pickle.load(f)

        # Get BAU average price in first year
        bau_prices = self.analysis.get_year_average_price(bau_results['PRICES'], factor=-1)
        initial_price = bau_prices.loc[first_year, 'average_price_real']

        return initial_price


if __name__ == '__main__':
    log_file_name = 'case_logger'
    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, 'output', 'local')

    # Object used to run model cases
    cases = ModelCases(output_directory, log_file_name)

    # Common model parameters
    start_year = 2016
    end_year = 2026
    scenarios = 5
    scheme_transition_year = 2022

    # Run business-as-usual case
    r_bau = cases.run_bau_case(output_directory, start_year, end_year, scenarios)

    # Permit prices for carbon pricing scenarios
    permit_prices_model = {y: float(40) for y in range(start_year, end_year + 1)}

    # Run carbon tax case
    # r_ct = cases.run_carbon_tax_case(output_directory, start_year, end_year, scenarios, permit_prices_model)

    # Run Refunded Emissions Payment Scheme
    r_rep = cases.run_rep_case(output_directory, start_year, end_year, scenarios, permit_prices_model)

    # Price targeting heuristic case over whole model horizon
    r_pth = cases.run_price_smoothing_heuristic_case(output_directory, 'rep_case.pickle', scheme_transition_year)

    # Price targeting MPPDC model
    r_ptm = cases.run_price_smoothing_mppdc_case(output_directory, 'rep_case.pickle', scheme_transition_year)

    # # Load results
    # with open(os.path.join(output_directory, 'price_targeting_heuristic_case.pickle'), 'rb') as f:
    #     r_pth = pickle.load(f)
    #
    # # Load results
    # with open(os.path.join(output_directory, 'price_targeting_mppdc_case.pickle'), 'rb') as f:
    #     r_ptm = pickle.load(f)
