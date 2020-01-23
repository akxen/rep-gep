"""Generation expansion planning cases"""

import os
import sys
import time
import copy
import pickle
import hashlib
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, '4_analysis'))

from pyomo.environ import *
import matplotlib.pyplot as plt
from pyomo.util.infeasible import log_infeasible_constraints

from targets import Targets
from gep import MPPDCModel, Primal, Dual
from auxiliary import BaselineUpdater
from analysis import AnalyseResults


class REP:
    def __init__(self, output_dir, params):
        # Timer
        t_start = time.time()
        print(f'Constructing class: {t_start}')

        # Output directory
        self.output_dir = output_dir

        # Placeholder for object used to construct and run models
        self.primal = Primal(params['start'], params['end'], params['scenarios'])

        # Key model parameters
        self.params = params

        # Placeholder for model object
        self.m = self.primal.construct_model()

        print(f'Initialised model: {time.time() - t_start}s')

    @staticmethod
    def extract_results(m, component_name):
        """Extract values associated with model components"""

        model_component = m.__getattribute__(component_name)

        if type(model_component) == pyomo.core.base.expression.IndexedExpression:
            return {k: model_component[k].expr() for k in model_component.keys()}

        elif type(model_component) == pyomo.core.base.expression.SimpleExpression:
            return model_component.expr()

        elif type(model_component) == pyomo.core.base.var.SimpleVar:
            return model_component.value

        elif type(model_component) == pyomo.core.base.var.IndexedVar:
            return model_component.get_values()

        elif type(model_component) == pyomo.core.base.param.IndexedParam:
            try:
                return {k: v.value for k, v in model_component.items()}
            except AttributeError:
                return {k: v for k, v in model_component.items()}

        elif type(model_component) == pyomo.core.base.param.SimpleParam:
            return model_component.value

        elif type(model_component) == pyomo.core.base.objective.SimpleObjective:
            return model_component.expr()

        else:
            raise Exception(f'Unexpected model component: {component_name}')

    @staticmethod
    def get_hash(params):
        """Get hash string of model parameters. Used to identify cases in log file."""

        return hashlib.sha224(str(params).encode('utf-8', 'ignore')).hexdigest()[:10]

    def save_hash(self, case_id, params):
        """Save case ID and associated parameters to file"""

        # Include case ID in dictionary
        params['case_id'] = case_id

        # Save case IDs and all associated params to file
        with open(os.path.join(self.output_dir, 'case_ids.txt'), 'a+') as f:
            f.write(str(params) + '\n')

    def save_results(self, results, filename):
        """Save model results"""

        with open(os.path.join(self.output_dir, filename), 'wb') as f:
            pickle.dump(results, f)

    def save_solution_summary(self, summary):
        """Save solution summary"""

        # Save summary of total solution time + number of iterations (if specified)
        with open(os.path.join(self.output_dir, 'solution_summary.txt'), 'a+') as f:
            f.write(str(summary) + '\n')

    def update_baseline(self, baselines):
        """Update emissions intensity baseline"""

        for y in self.m.Y:
            self.m.baseline[y].fix(baselines[y])

    def update_permit_price(self, permit_price):
        """Update permit price"""

        for y in self.m.Y:
            self.m.permit_price[y].fix(permit_price)

    @staticmethod
    def get_iteration_difference(i_input, i_output, key):
        """Get max absolute difference between successive iterations for a particular model component"""

        return max([abs(i_input[key][k] - i_output[key][k]) for k in i_input[key].keys()])

    def get_results(self):
        """Extract results"""

        # Results to extract
        keys = ['x_c', 'p', 'p_V', 'p_in', 'p_out', 'q', 'p_L', 'baseline', 'permit_price', 'YEAR_EMISSIONS',
                'YEAR_EMISSIONS_INTENSITY', 'YEAR_SCHEME_REVENUE', 'TOTAL_SCHEME_REVENUE', 'C_MC', 'ETA',
                'DELTA', 'RHO', 'EMISSIONS_RATE', 'YEAR_SCHEME_EMISSIONS_INTENSITY',
                'YEAR_CUMULATIVE_SCHEME_REVENUE', 'OBJECTIVE']

        # Model results
        results = {k: self.extract_results(self.m, k) for k in keys}

        # Add dual variable from power balance constraint
        results['PRICES'] = {k: self.m.dual[self.m.POWER_BALANCE[k]] for k in self.m.POWER_BALANCE.keys()}

        return results

    def run_model(self, baselines, permit_price):
        """Run carbon tax case"""

        # Start timer
        t_start = time.time()

        # Fix baseline and permit price
        self.update_baseline(baselines)
        print(f'Fixed baselines: {time.time() - t_start}')

        self.update_permit_price(permit_price)
        print(f'Fixed permit prices: {time.time() - t_start}')

        # Solve model
        self.m, status = self.primal.solve_model(self.m)
        print(f'Solved model: {time.time() - t_start}')

        # Get results
        results = self.get_results()
        print(f'Extracted results model: {time.time() - t_start}')

        return results, status

    def run_case(self, carbon_price, overwrite=False):
        """Run REP case"""

        # Filename for REP case
        filename = f'rep_cp-{carbon_price:.0f}.pickle'

        # Check if model has already been solved
        if (not overwrite) and (filename in os.listdir(self.output_dir)):
            print(f'Already solved: {filename}')
            return

        # Construct hash for case ID and save case hash with associated parameters
        case_id = self.get_hash(self.params)
        self.save_hash(case_id, self.params)

        # Start time
        t_start = time.time()

        # First run carbon tax case
        baselines = {y: 0 for y in self.m.Y}
        carbon_tax_results, carbon_tax_status = self.run_model(baselines, carbon_price)

        # Container for iteration results
        i_results = dict()

        # Iteration input
        i_input = carbon_tax_results

        # Initialise iteration counter
        counter = 1

        while True:
            # Re-run model with new baselines
            i_output, i_status = self.run_model(baselines, carbon_price)
            log_infeasible_constraints(self.m)

            # Add results to iteration results container
            i_results[counter] = copy.deepcopy(i_output)

            # Check max absolute capacity difference between successive iterations
            max_capacity_difference = self.get_iteration_difference(i_input, i_output, 'x_c')
            print(f'{counter}: Maximum capacity difference = {max_capacity_difference} MW')

            # Max absolute baseline difference between successive iterations
            max_baseline_difference = self.get_iteration_difference(i_input, i_output, 'baseline')
            print(f'{counter}: Maximum baseline difference = {max_baseline_difference} tCO2/MWh')

            # If max absolute difference between successive iterations is sufficiently small stop iterating
            if max_baseline_difference < 0.05:
                break

            # If iteration limit exceeded
            elif counter > 9:
                print('Max iterations exceeded. Exiting loop.')
                break

            # Update iteration inputs (used to check stopping criterion in next iteration)
            i_input = copy.deepcopy(i_output)

            # Update baselines to be used in next iteration
            baselines = i_output['YEAR_SCHEME_EMISSIONS_INTENSITY']

            # Update iteration counter
            counter += 1

        # Combine results into single dictionary
        results = {'stage_1_carbon_tax': carbon_tax_results, 'stage_2_rep': i_results}

        # Save results
        self.save_results(results, filename)

        # Dictionary to be returned by method
        output = {'results': results, 'model': self.m, 'status': i_status}

        # Total number of iterations processed
        total_iterations = max(i_results.keys())

        # Save summary of the solution time
        solution_summary = {'case_id': case_id, 'mode': 'rep', 'carbon_price': carbon_price,
                            'total_solution_time': time.time() - t_start, 'total_iterations': total_iterations,
                            'max_capacity_difference': max_capacity_difference,
                            'max_baseline_difference': max_baseline_difference}

        self.save_solution_summary(solution_summary)
        message = 'Finished REP case: ' + str(solution_summary)

        return output

    def run_cases(self, carbon_prices):
        """Run REP model for a range of carbon prices"""

        for c in carbon_prices:
            self.run_case(c)


    # def run_case_old(self, params, output_dir, overwrite=False):
    #     """Run case for a given permit price (same for all years in model horizon)"""
    #
    #     # Extract case parameters
    #     start, end, scenarios = params['start'], params['end'], params['scenarios']
    #     permit_prices = params['permit_prices']
    #
    #     # First run carbon tax case
    #     baselines = {y: float(0) for y in range(start, end + 1)}
    #
    #     # Check that carbon tax is same for all years in model horizon
    #     assert len(set(permit_prices.values())) == 1, f'Permit price trajectory is not flat: {permit_prices}'
    #
    #     # Extract carbon price in first year (same as all other used). To be used in filename.
    #     carbon_price = permit_prices[start]
    #
    #     # Filename for REP case
    #     filename = f'rep_cp-{carbon_price:.0f}.pickle'
    #
    #     # Check if model has already been solved
    #     if (not overwrite) and (filename in os.listdir(output_dir)):
    #         print(f'Already solved: {filename}')
    #         return
    #
    #     # Construct hash for case ID
    #     case_id = self.get_hash(params)
    #
    #     # Save hash and associated parameters
    #     self.save_hash(case_id, params, output_dir)
    #
    #     # Start timer for model run
    #     t_start = time.time()
    #
    #     self.algorithm_logger('run_rep_case', 'Starting case with params: ' + str(params))
    #
    #     # Results to extract
    #     result_keys = ['x_c', 'p', 'p_V', 'p_in', 'p_out', 'q', 'p_L', 'baseline', 'permit_price', 'YEAR_EMISSIONS',
    #                    'YEAR_EMISSIONS_INTENSITY', 'YEAR_SCHEME_REVENUE', 'TOTAL_SCHEME_REVENUE', 'C_MC', 'ETA',
    #                    'DELTA', 'RHO', 'EMISSIONS_RATE', 'YEAR_SCHEME_EMISSIONS_INTENSITY',
    #                    'YEAR_CUMULATIVE_SCHEME_REVENUE', 'OBJECTIVE']
    #
    #     # Run carbon tax case
    #     self.algorithm_logger('run_rep_case', 'Starting carbon tax case solve')
    #     m, status = self.run_primal_fixed_policy(start, end, scenarios, permit_prices, baselines)
    #     log_infeasible_constraints(m)
    #     self.algorithm_logger('run_rep_case', 'Finished carbon tax case solve')
    #
    #     # Model results
    #     carbon_tax_results = {k: self.extract_result(m, k) for k in result_keys}
    #
    #     # Add dual variable from power balance constraint
    #     carbon_tax_results['PRICES'] = {k: m.dual[m.POWER_BALANCE[k]] for k in m.POWER_BALANCE.keys()}
    #
    #     # Update baselines so they = emissions intensity of output from participating generators
    #     baselines = carbon_tax_results['YEAR_SCHEME_EMISSIONS_INTENSITY']
    #
    #     # Container for iteration results
    #     i_results = dict()
    #
    #     # Iteration counter
    #     counter = 1
    #
    #     # Initialise iteration input to carbon tax scenario results (used to check stopping criterion)
    #     i_input = carbon_tax_results
    #
    #     while True:
    #         # Re-run model with new baselines
    #         self.algorithm_logger('run_rep_case', f'Starting solve for REP iteration={counter}')
    #         m, status = self.run_primal_fixed_policy(start, end, scenarios, permit_prices, baselines)
    #         log_infeasible_constraints(m)
    #         self.algorithm_logger('run_rep_case', f'Finished solved for REP iteration={counter}')
    #
    #         # Model results
    #         i_output = {k: self.extract_result(m, k) for k in result_keys}
    #
    #         # Get dual variable values from power balance constraint
    #         i_output['PRICES'] = {k: m.dual[m.POWER_BALANCE[k]] for k in m.POWER_BALANCE.keys()}
    #
    #         # Add results to iteration results container
    #         i_results[counter] = copy.deepcopy(i_output)
    #
    #         # Check max absolute capacity difference between successive iterations
    #         max_capacity_difference = self.get_successive_iteration_difference(i_input, i_output, 'x_c')
    #         print(f'{counter}: Maximum capacity difference = {max_capacity_difference} MW')
    #
    #         # Max absolute baseline difference between successive iterations
    #         max_baseline_difference = self.get_successive_iteration_difference(i_input, i_output, 'baseline')
    #         print(f'{counter}: Maximum baseline difference = {max_baseline_difference} tCO2/MWh')
    #
    #         # If max absolute difference between successive iterations is sufficiently small stop iterating
    #         if max_baseline_difference < 0.05:
    #             break
    #
    #         # If iteration limit exceeded
    #         elif counter > 9:
    #             message = f'Max iterations exceeded. Exiting loop.'
    #             print(message)
    #             self.algorithm_logger('run_rep_case', message)
    #             break
    #
    #         # Update iteration inputs (used to check stopping criterion in next iteration)
    #         i_input = copy.deepcopy(i_output)
    #
    #         # Update baselines to be used in next iteration
    #         baselines = i_output['YEAR_SCHEME_EMISSIONS_INTENSITY']
    #
    #         # Update iteration counter
    #         counter += 1
    #
    #     # Combine results into single dictionary
    #     results = {'stage_1_carbon_tax': carbon_tax_results, 'stage_2_rep': i_results}
    #
    #     # Save results
    #     self.save_results(results, output_dir, filename)
    #
    #     # Dictionary to be returned by method
    #     output = {'results': results, 'model': m, 'status': status}
    #     self.algorithm_logger('run_rep_case', f'Finished REP case')
    #
    #     # Total number of iterations processed
    #     total_iterations = max(i_results.keys())
    #
    #     # Save summary of the solution time
    #     solution_summary = {'case_id': case_id, 'mode': params['mode'], 'carbon_price': carbon_price,
    #                         'total_solution_time': time.time() - t_start, 'total_iterations': total_iterations,
    #                         'max_capacity_difference': max_capacity_difference,
    #                         'max_baseline_difference': max_baseline_difference}
    #
    #     self.save_solution_summary(solution_summary, output_dir)
    #
    #     message = 'Finished REP case: ' + str(solution_summary)
    #     self.algorithm_logger('run_rep_case', message)
    #
    #     return output


class ModelCases:
    def __init__(self, output_dir, log_name):
        logging.basicConfig(filename=os.path.join(output_dir, f'{log_name}.log'), filemode='a',
                            format='%(asctime)s %(name)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

        # Used to parse prices and analyse results
        self.analysis = AnalyseResults()

        # Get scheme targets
        self.targets = Targets()

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

        elif type(model_component) == pyomo.core.base.var.SimpleVar:
            return model_component.value

        elif type(model_component) == pyomo.core.base.var.IndexedVar:
            return model_component.get_values()

        elif type(model_component) == pyomo.core.base.param.IndexedParam:
            try:
                return {k: v.value for k, v in model_component.items()}
            except AttributeError:
                return {k: v for k, v in model_component.items()}

        elif type(model_component) == pyomo.core.base.param.SimpleParam:
            return model_component.value

        elif type(model_component) == pyomo.core.base.objective.SimpleObjective:
            return model_component.expr()

        else:
            raise Exception(f'Unexpected model component: {component_name}')

    @staticmethod
    def save_results(results, output_dir, filename):
        """Save model results"""

        with open(os.path.join(output_dir, filename), 'wb') as f:
            pickle.dump(results, f)

    @staticmethod
    def get_hash(params):
        """Get hash string of model parameters. Used to identify cases in log file."""

        return hashlib.sha224(str(params).encode('utf-8', 'ignore')).hexdigest()[:10]

    @staticmethod
    def save_hash(case_id, params, output_dir):
        """Save case ID and associated parameters to file"""

        # Include case ID in dictionary
        params['case_id'] = case_id

        # Save case IDs and all associated params to file
        with open(os.path.join(output_dir, 'case_ids.txt'), 'a+') as f:
            f.write(str(params) + '\n')

    @staticmethod
    def save_solution_summary(summary, output_dir):
        """Save solution summary"""

        # Save summary of total solution time + number of iterations (if specified)
        with open(os.path.join(output_dir, 'solution_summary.txt'), 'a+') as f:
            f.write(str(summary) + '\n')

    def get_bau_initial_price(self, output_dir, first_year):
        """Get BAU price in first year"""

        # Load BAU results
        with open(os.path.join(output_dir, 'bau_case.pickle'), 'rb') as f:
            results = pickle.load(f)

        # Get BAU average price in first year
        prices = self.analysis.get_year_average_price(results['PRICES'], factor=-1)
        initial_price = prices.loc[first_year, 'average_price_real']

        return initial_price

    @staticmethod
    def get_successive_iteration_difference(i_input, i_output, key):
        """Get max absolute difference between successive iterations for a particular model component"""

        return max([abs(i_input[key][k] - i_output[key][k]) for k in i_input[key].keys()])

    @staticmethod
    def run_mppdc_fixed_policy(final_year, scenarios_per_year, permit_prices, baselines,
                               include_primal_constraints=True):
        """Run MPPDC model with fixed policy parameters"""

        # Initialise object and model used to run MPPDC model
        mppdc = MPPDCModel(final_year, scenarios_per_year)
        m = mppdc.construct_model(include_primal_constraints=include_primal_constraints)

        # Fix permit prices and baselines
        for y in m.Y:
            m.permit_price[y].fix(permit_prices[y])
            m.baseline[y].fix(baselines[y])

        # Solve MPPDC model with fixed policy parameters
        m, status = mppdc.solve_model(m)

        return m, status

    @staticmethod
    def run_primal_fixed_policy(start_year, final_year, scenarios_per_year, permit_prices, baselines):
        """Run primal model with fixed policy parameters"""

        # Initialise object and model used to run primal model
        primal = Primal(start_year, final_year, scenarios_per_year)
        m = primal.construct_model()

        # Fix permit prices and baselines to specified levels
        for y in m.Y:
            m.permit_price[y].fix(permit_prices[y])
            m.baseline[y].fix(baselines[y])

        # Solve primal model with fixed policy parameters
        m, status = primal.solve_model(m)

        return m, status

    def run_bau_case(self, params, output_dir, overwrite=False):
        """Run business-as-usual case"""

        # Case filename
        filename = 'bau_case.pickle'

        # Check if case exists
        if (not overwrite) and (filename in os.listdir(output_dir)):
            print(f'Already solved: {filename}')
            return

        # Construct hash for case
        case_id = self.get_hash(params)

        # Save case params and associated hash
        self.save_hash(case_id, params, output_dir)

        # Extract case parameters for model
        start, end, scenarios = params['start'], params['end'], params['scenarios']

        # Start timer for case run
        t_start = time.time()

        message = f"""Starting case: first_year={start}, final_year={end}, scenarios_per_year={scenarios}"""
        self.algorithm_logger('run_bau_case', message)

        # Permit prices and emissions intensity baselines for BAU case (all 0)
        permit_prices = {y: float(0) for y in range(start, end + 1)}
        baselines = {y: float(0) for y in range(start, end + 1)}

        # Run model
        self.algorithm_logger('run_bau_case', 'Starting solve')
        m, status = self.run_primal_fixed_policy(start, end, scenarios, permit_prices, baselines)
        log_infeasible_constraints(m)
        self.algorithm_logger('run_bau_case', 'Finished solve')

        # Results to extract
        result_keys = ['x_c', 'p', 'p_V', 'p_in', 'p_out', 'p_L', 'baseline', 'permit_price', 'YEAR_EMISSIONS',
                       'YEAR_EMISSIONS_INTENSITY', 'YEAR_SCHEME_REVENUE', 'TOTAL_SCHEME_REVENUE', 'C_MC', 'ETA',
                       'DELTA', 'RHO', 'EMISSIONS_RATE', 'OBJECTIVE']

        # Model results
        results = {k: self.extract_result(m, k) for k in result_keys}

        # Add dual variable from power balance constraint
        results['PRICES'] = {k: m.dual[m.POWER_BALANCE[k]] for k in m.POWER_BALANCE.keys()}

        # Save results
        self.save_results(results, output_dir, filename)

        # Combine output in dictionary. To be returned by method.
        output = {'results': results, 'model': m, 'status': status}

        # Solution summary
        solution_summary = {'case_id': case_id, 'mode': params['mode'], 'total_solution_time': time.time() - t_start}
        self.save_solution_summary(solution_summary, output_dir)

        self.algorithm_logger('run_bau_case', f'Finished BAU case: case_id={case_id}, total_solution_time={time.time() - t_start}s')

        return output

    def run_rep_case(self, params, output_dir, overwrite=False):
        """Run carbon tax scenario"""

        # Extract case parameters
        start, end, scenarios = params['start'], params['end'], params['scenarios']
        permit_prices = params['permit_prices']

        # First run carbon tax case
        baselines = {y: float(0) for y in range(start, end + 1)}

        # Check that carbon tax is same for all years in model horizon
        assert len(set(permit_prices.values())) == 1, f'Permit price trajectory is not flat: {permit_prices}'

        # Extract carbon price in first year (same as all other used). To be used in filename.
        carbon_price = permit_prices[start]

        # Filename for REP case
        filename = f'rep_cp-{carbon_price:.0f}.pickle'

        # Check if model has already been solved
        if (not overwrite) and (filename in os.listdir(output_dir)):
            print(f'Already solved: {filename}')
            return

        # Construct hash for case ID
        case_id = self.get_hash(params)

        # Save hash and associated parameters
        self.save_hash(case_id, params, output_dir)

        # Start timer for model run
        t_start = time.time()

        self.algorithm_logger('run_rep_case', 'Starting case with params: ' + str(params))

        # Results to extract
        result_keys = ['x_c', 'p', 'p_V', 'p_in', 'p_out', 'q', 'p_L', 'baseline', 'permit_price', 'YEAR_EMISSIONS',
                       'YEAR_EMISSIONS_INTENSITY', 'YEAR_SCHEME_REVENUE', 'TOTAL_SCHEME_REVENUE', 'C_MC', 'ETA',
                       'DELTA', 'RHO', 'EMISSIONS_RATE', 'YEAR_SCHEME_EMISSIONS_INTENSITY',
                       'YEAR_CUMULATIVE_SCHEME_REVENUE', 'OBJECTIVE']

        # Run carbon tax case
        self.algorithm_logger('run_rep_case', 'Starting carbon tax case solve')
        m, status = self.run_primal_fixed_policy(start, end, scenarios, permit_prices, baselines)
        log_infeasible_constraints(m)
        self.algorithm_logger('run_rep_case', 'Finished carbon tax case solve')

        # Model results
        carbon_tax_results = {k: self.extract_result(m, k) for k in result_keys}

        # Add dual variable from power balance constraint
        carbon_tax_results['PRICES'] = {k: m.dual[m.POWER_BALANCE[k]] for k in m.POWER_BALANCE.keys()}

        # Update baselines so they = emissions intensity of output from participating generators
        baselines = carbon_tax_results['YEAR_SCHEME_EMISSIONS_INTENSITY']

        # Container for iteration results
        i_results = dict()

        # Iteration counter
        counter = 1

        # Initialise iteration input to carbon tax scenario results (used to check stopping criterion)
        i_input = carbon_tax_results

        while True:
            # Re-run model with new baselines
            self.algorithm_logger('run_rep_case', f'Starting solve for REP iteration={counter}')
            m, status = self.run_primal_fixed_policy(start, end, scenarios, permit_prices, baselines)
            log_infeasible_constraints(m)
            self.algorithm_logger('run_rep_case', f'Finished solved for REP iteration={counter}')

            # Model results
            i_output = {k: self.extract_result(m, k) for k in result_keys}

            # Get dual variable values from power balance constraint
            i_output['PRICES'] = {k: m.dual[m.POWER_BALANCE[k]] for k in m.POWER_BALANCE.keys()}

            # Add results to iteration results container
            i_results[counter] = copy.deepcopy(i_output)

            # Check max absolute capacity difference between successive iterations
            max_capacity_difference = self.get_successive_iteration_difference(i_input, i_output, 'x_c')
            print(f'{counter}: Maximum capacity difference = {max_capacity_difference} MW')

            # Max absolute baseline difference between successive iterations
            max_baseline_difference = self.get_successive_iteration_difference(i_input, i_output, 'baseline')
            print(f'{counter}: Maximum baseline difference = {max_baseline_difference} tCO2/MWh')

            # If max absolute difference between successive iterations is sufficiently small stop iterating
            if max_baseline_difference < 0.05:
                break

            # If iteration limit exceeded
            elif counter > 9:
                message = f'Max iterations exceeded. Exiting loop.'
                print(message)
                self.algorithm_logger('run_rep_case', message)
                break

            # Update iteration inputs (used to check stopping criterion in next iteration)
            i_input = copy.deepcopy(i_output)

            # Update baselines to be used in next iteration
            baselines = i_output['YEAR_SCHEME_EMISSIONS_INTENSITY']

            # Update iteration counter
            counter += 1

        # Combine results into single dictionary
        results = {'stage_1_carbon_tax': carbon_tax_results, 'stage_2_rep': i_results}

        # Save results
        self.save_results(results, output_dir, filename)

        # Dictionary to be returned by method
        output = {'results': results, 'model': m, 'status': status}
        self.algorithm_logger('run_rep_case', f'Finished REP case')

        # Total number of iterations processed
        total_iterations = max(i_results.keys())

        # Save summary of the solution time
        solution_summary = {'case_id': case_id, 'mode': params['mode'], 'carbon_price': carbon_price,
                            'total_solution_time': time.time() - t_start, 'total_iterations': total_iterations,
                            'max_capacity_difference': max_capacity_difference,
                            'max_baseline_difference': max_baseline_difference}

        self.save_solution_summary(solution_summary, output_dir)

        message = 'Finished REP case: ' + str(solution_summary)
        self.algorithm_logger('run_rep_case', message)

        return output

    def run_price_smoothing_heuristic_case(self, params, output_dir, overwrite=False):
        """Smooth prices over entire model horizon using approximated price functions"""

        # Get filename based on run mode
        if params['mode'] == 'bau_deviation_minimisation':
            filename = f"heuristic_baudev_ty-{params['transition_year']}_cp-{params['carbon_price']}.pickle"

        elif params['mode'] == 'price_change_minimisation':
            filename = f"heuristic_pdev_ty-{params['transition_year']}_cp-{params['carbon_price']}.pickle"

        elif params['mode'] == 'price_target':
            filename = f"heuristic_ptar_ty-{params['transition_year']}_cp-{params['carbon_price']}.pickle"

        else:
            raise Exception(f"Unexpected run mode: {params['mode']}")

        # Check if case already solved
        if (not overwrite) and (filename in os.listdir(output_dir)):
            print(f'Already solved: {filename}')
            return

        # Get hash for case
        case_id = self.get_hash(params)

        # Save case ID and associated model parameters
        self.save_hash(case_id, params, output_dir)

        # Start timer for model run
        t_start = time.time()

        self.algorithm_logger('run_price_smoothing_heuristic_case', 'Starting case with params: ' + str(params))

        # Load REP results
        with open(os.path.join(output_dir, params['rep_filename']), 'rb') as f:
            rep_results = pickle.load(f)

        # Get results corresponding to last iteration of REP solution
        rep_iteration = rep_results['stage_2_rep'][max(rep_results['stage_2_rep'].keys())]

        # Model parameters used to initialise classes that construct and run models
        start, end, scenarios = params['start'], params['end'], params['scenarios']
        bau_initial_price = self.get_bau_initial_price(output_dir, start)

        # Classes used to construct and run primal and MPPDC programs
        primal = Primal(start, end, scenarios)
        baseline = BaselineUpdater(start, end, scenarios, params['transition_year'])

        # Construct primal program
        m_p = primal.construct_model()

        # Results to extract from primal program
        primal_keys = ['x_c', 'p', 'p_V', 'p_in', 'p_out', 'p_L', 'baseline', 'permit_price', 'YEAR_EMISSIONS',
                       'YEAR_EMISSIONS_INTENSITY', 'YEAR_SCHEME_REVENUE', 'TOTAL_SCHEME_REVENUE', 'C_MC', 'ETA',
                       'DELTA', 'RHO', 'EMISSIONS_RATE', 'YEAR_CUMULATIVE_SCHEME_REVENUE',
                       'YEAR_SCHEME_EMISSIONS_INTENSITY', 'OBJECTIVE']

        # Results to extract from baseline targeting model
        baseline_keys = ['YEAR_AVERAGE_PRICE', 'YEAR_AVERAGE_PRICE_0', 'YEAR_ABSOLUTE_PRICE_DIFFERENCE',
                         'TOTAL_ABSOLUTE_PRICE_DIFFERENCE', 'PRICE_WEIGHTS', 'YEAR_SCHEME_REVENUE',
                         'YEAR_CUMULATIVE_SCHEME_REVENUE', 'baseline']

        # Container for iteration results
        i_results = dict()

        # Initialise price setting generator input as results obtained from final REP iteration
        psg_input = rep_iteration

        # Initialise iteration counter
        counter = 1

        while True:
            self.algorithm_logger('run_price_smoothing_heuristic_case', f'Starting iteration={counter}')

            # Identify price setting generators
            psg = baseline.prices.get_price_setting_generators_from_model_results(psg_input)

            # Construct model used to calibrate baseline
            m_b = baseline.construct_model(psg)

            # Update parameters
            m_b = baseline.update_parameters(m_b, psg_input)
            m_b.YEAR_AVERAGE_PRICE_0 = float(bau_initial_price)
            m_b.PRICE_WEIGHTS.store_values(params['price_weights'])

            # Activate constraints and objectives depending on case being run
            m_b.NON_NEGATIVE_TRANSITION_REVENUE_CONS.activate()

            if params['mode'] == 'bau_deviation_minimisation':
                # Set the price target to be BAU price
                bau_price_target = {y: bau_initial_price for y in m_b.Y}
                m_b.YEAR_AVERAGE_PRICE_TARGET.store_values(bau_price_target)

                # Activate price targeting constraints and objective
                m_b.PRICE_TARGET_DEVIATION_1.activate()
                m_b.PRICE_TARGET_DEVIATION_2.activate()
                m_b.OBJECTIVE_PRICE_TARGET_DIFFERENCE.activate()

                # Append name of objective so objective value can be extracted, and create filename for case
                baseline_keys.append('OBJECTIVE_PRICE_TARGET_DIFFERENCE')

            elif params['mode'] == 'price_change_minimisation':
                # Activate constraints penalised price deviations over successive years
                m_b.PRICE_CHANGE_DEVIATION_1.activate()
                m_b.PRICE_CHANGE_DEVIATION_2.activate()
                m_b.OBJECTIVE_PRICE_DEVIATION.activate()

                # Append name of objective so objective value can be extracted, and create filename for case
                baseline_keys.append('OBJECTIVE_PRICE_DEVIATION')

            elif params['mode'] == 'price_target':
                # Set target price trajectory to prices obtained from BAU model over same period
                m_b.YEAR_AVERAGE_PRICE_TARGET.store_values(params['price_target'])

                # Activate price targeting constraints and objective function
                m_b.PRICE_TARGET_DEVIATION_1.activate()
                m_b.PRICE_TARGET_DEVIATION_2.activate()
                m_b.OBJECTIVE_PRICE_TARGET_DIFFERENCE.activate()

                # Append name of objective so objective value can be extracted, and create filename for case
                baseline_keys.append('OBJECTIVE_PRICE_TARGET_DIFFERENCE')

            else:
                raise Exception(f"Unexpected run mode: {params['mode']}")

            for y in m_b.Y:
                if y >= params['transition_year']:
                    m_b.YEAR_NET_SCHEME_REVENUE_NEUTRAL_CONS[y].activate()

            # Solve model
            m_b, m_b_status = baseline.solve_model(m_b)
            r_b = copy.deepcopy({k: self.extract_result(m_b, k) for k in baseline_keys})

            # Update baselines and permit prices in primal model
            for y in m_p.Y:
                m_p.baseline[y].fix(m_b.baseline[y].value)
                m_p.permit_price[y].fix(m_b.PERMIT_PRICE[y].value)

            # Solve primal program
            m_p, m_p_status = primal.solve_model(m_p)

            # Log all infeasible constraints
            log_infeasible_constraints(m_p)

            # Get results
            r_p = copy.deepcopy({v: self.extract_result(m_p, v) for v in primal_keys})
            r_p['PRICES'] = copy.deepcopy({k: m_p.dual[m_p.POWER_BALANCE[k]] for k in m_p.POWER_BALANCE.keys()})
            i_results[counter] = {'primal': r_p, 'auxiliary': r_b}

            # Max difference in capacity sizing decision between iterations
            max_capacity_difference = self.get_successive_iteration_difference(psg_input, r_p, 'x_c')
            print(f'Max capacity difference: {max_capacity_difference} MW')

            # Max absolute baseline difference between successive iterations
            max_baseline_difference = self.get_successive_iteration_difference(psg_input, r_p, 'baseline')
            print(f'{counter}: Maximum baseline difference = {max_baseline_difference} tCO2/MWh')

            self.algorithm_logger('run_price_smoothing_heuristic_case', f'Finished iteration={counter}')

            # If baseline difference between successive iterations is sufficiently small then stop
            if max_baseline_difference < 0.05:
                break

            # Stop iterating if max iteration limit exceeded
            elif counter > 9:
                message = f'Max iterations exceeded. Exiting loop.'
                print(message)
                self.algorithm_logger('run_price_smoothing_heuristic_case', message)
                break

            else:
                # Update dictionary of price setting generator program inputs
                psg_input = r_p

            # Update iteration counter
            counter += 1

        self.algorithm_logger('run_price_smoothing_heuristic_case', f'Finished solving model')

        # Combine results into a single dictionary
        results = {**rep_results, 'stage_3_price_targeting': i_results, 'parameters': params}

        # Save results
        self.save_results(results, output_dir, filename)

        # Combine output for method (can be used for debugging)
        output = {'auxiliary_model': m_b, 'auxiliary_status': m_b_status, 'primal_model': m_p,
                  'primal_status': m_p_status, 'results': results}

        # Total iterations
        total_iterations = max(i_results.keys())

        # Save summary of the solution time
        solution_summary = {'case_id': case_id, 'mode': params['mode'], 'carbon_price': params['carbon_price'],
                            'transition_year': params['transition_year'],
                            'total_solution_time': time.time() - t_start, 'total_iterations': total_iterations,
                            'max_capacity_difference': max_capacity_difference,
                            'max_baseline_difference': max_baseline_difference}
        self.save_solution_summary(solution_summary, output_dir)

        message = f"Finished heuristic case: " + str(solution_summary)
        self.algorithm_logger('run_price_smoothing_heuristic_case', message)

        return output

    def run_price_smoothing_mppdc_case(self, params, output_dir, overwrite=False):
        """Run case to smooth prices over model horizon, subject to total revenue constraint"""

        # Get case filename based on run mode
        if params['mode'] == 'bau_deviation_minimisation':
            filename = f"mppdc_baudev_ty-{params['transition_year']}_cp-{params['carbon_price']}.pickle"

        elif params['mode'] == 'price_change_minimisation':
            filename = f"mppdc_pdev_ty-{params['transition_year']}_cp-{params['carbon_price']}.pickle"

        elif params['mode'] == 'price_target':
            filename = f"mppdc_ptar_ty-{params['transition_year']}_cp-{params['carbon_price']}.pickle"

        else:
            raise Exception(f"Unexpected run mode: {params['mode']}")

        # Check if case already solved
        if (not overwrite) and (filename in os.listdir(output_dir)):
            print(f'Already solved: {filename}')
            return

        # Construct hash for case ID
        case_id = self.get_hash(params)

        # Save hash and associated parameters
        self.save_hash(case_id, params, output_dir)

        # Start timer for model run
        t_start = time.time()

        self.algorithm_logger('run_price_smoothing_mppdc_case', 'Starting MPPDC case with params: ' + str(params))

        # Load REP results
        with open(os.path.join(output_dir, params['rep_filename']), 'rb') as f:
            rep_results = pickle.load(f)

        # Get results corresponding to last iteration of REP solution
        rep_iteration = rep_results['stage_2_rep'][max(rep_results['stage_2_rep'].keys())]

        # Extract parameters from last iteration of REP program results
        start, end, scenarios = params['start'], params['end'], params['scenarios']
        bau_initial_price = self.get_bau_initial_price(output_dir, start)

        # Classes used to construct and run primal and MPPDC programs
        mppdc = MPPDCModel(start, end, scenarios, params['transition_year'])
        primal = Primal(start, end, scenarios)

        # Construct MPPDC
        m_m = mppdc.construct_model(include_primal_constraints=True)

        # Construct primal model
        m_p = primal.construct_model()

        # Update MPPDC model parameters
        m_m.YEAR_AVERAGE_PRICE_0 = float(bau_initial_price)
        m_m.PRICE_WEIGHTS.store_values(params['price_weights'])

        # Activate necessary constraints depending on run mode
        m_m.NON_NEGATIVE_TRANSITION_REVENUE_CONS.activate()

        if params['mode'] == 'bau_deviation_minimisation':
            m_m.PRICE_BAU_DEVIATION_1.activate()
            m_m.PRICE_BAU_DEVIATION_2.activate()

        elif params['mode'] == 'price_change_minimisation':
            m_m.PRICE_CHANGE_DEVIATION_1.activate()
            m_m.PRICE_CHANGE_DEVIATION_2.activate()

        elif params['mode'] == 'price_target':
            m_m.YEAR_AVERAGE_PRICE_TARGET.store_values(params['price_target'])
            m_m.PRICE_TARGET_DEVIATION_1.activate()
            m_m.PRICE_TARGET_DEVIATION_2.activate()

        else:
            raise Exception(f"Unexpected run mode: {params['mode']}")

        for y in m_m.Y:
            if y >= params['transition_year']:
                m_m.YEAR_NET_SCHEME_REVENUE_NEUTRAL_CONS[y].activate()

        # Primal variables
        primal_vars = ['x_c', 'p', 'p_in', 'p_out', 'q', 'p_V', 'p_L', 'permit_price']
        fixed_vars = {v: rep_iteration[v] for v in primal_vars}

        # Results to extract from MPPDC model
        mppdc_keys = ['x_c', 'p', 'p_V', 'p_in', 'p_out', 'p_L', 'q', 'baseline', 'permit_price', 'lamb',
                      'YEAR_EMISSIONS', 'YEAR_EMISSIONS_INTENSITY', 'YEAR_SCHEME_EMISSIONS_INTENSITY',
                      'YEAR_SCHEME_REVENUE', 'YEAR_CUMULATIVE_SCHEME_REVENUE', 'TOTAL_SCHEME_REVENUE',
                      'YEAR_ABSOLUTE_PRICE_DIFFERENCE', 'YEAR_AVERAGE_PRICE_0', 'YEAR_AVERAGE_PRICE',
                      'YEAR_SUM_CUMULATIVE_PRICE_DIFFERENCE_WEIGHTED', 'OBJECTIVE',
                      'YEAR_ABSOLUTE_PRICE_DIFFERENCE_WEIGHTED', 'TOTAL_ABSOLUTE_PRICE_DIFFERENCE_WEIGHTED',
                      'YEAR_CUMULATIVE_PRICE_DIFFERENCE_WEIGHTED', 'sd_1', 'sd_2', 'STRONG_DUALITY_VIOLATION_COST',
                      'TRANSITION_YEAR', 'PRICE_WEIGHTS']

        # Container for iteration results
        i_results = {}

        # Initialise iteration counter
        counter = 1

        # Placeholder for max difference variables
        max_baseline_difference = None
        max_capacity_difference = None

        while True:
            self.algorithm_logger('run_price_smoothing_mppdc_case', f'Starting iteration={counter}')

            # Fix MPPDC variables
            m_m = mppdc.fix_variables(m_m, fixed_vars)

            # Solve MPPDC
            m_m, m_m_status = mppdc.solve_model(m_m)

            # Model timeout will cause sub-optimal termination condition
            if m_m_status.solver.termination_condition != TerminationCondition.optimal:
                i_results[counter] = None
                self.algorithm_logger('run_price_smoothing_mppdc_case', f'Sub-optimal solution')
                self.algorithm_logger('run_price_smoothing_mppdc_case', f'User time: {m_m_status.solver.user_time}s')

                # No primal model solved
                m_p, m_p_status = None, None
                break

            # Log infeasible constraints
            log_infeasible_constraints(m_m)

            # Results from MPPDC program
            r_m = copy.deepcopy({v: self.extract_result(m_m, v) for v in mppdc_keys})
            i_results[counter] = r_m

            # Update primal program with baselines and permit prices obtained from MPPDC model
            for y in m_p.Y:
                m_p.baseline[y].fix(m_m.baseline[y].value)
                m_p.permit_price[y].fix(m_m.permit_price[y].value)

            # Solve primal model
            m_p, m_p_status = primal.solve_model(m_p)
            log_infeasible_constraints(m_p)

            # Results from primal program
            p_r = copy.deepcopy({v: self.extract_result(m_p, v) for v in primal_vars})
            p_r['PRICES'] = copy.deepcopy({k: m_p.dual[m_p.POWER_BALANCE[k]] for k in m_p.POWER_BALANCE.keys()})
            i_results[counter]['primal'] = p_r

            # Max absolute capacity difference between MPPDC and primal program
            max_capacity_difference = max(abs(m_m.x_c[k].value - m_p.x_c[k].value) for k in m_m.x_c.keys())
            print(f'Max capacity difference: {max_capacity_difference} MW')

            # Max absolute baseline difference between MPPDC and primal program
            max_baseline_difference = max(abs(m_m.baseline[k].value - m_p.baseline[k].value) for k in m_m.baseline.keys())
            print(f'Max baseline difference: {max_baseline_difference} tCO2/MWh')

            # Check if capacity variables have changed
            if max_baseline_difference < 0.05:
                break

            # Check if max iterations exceeded
            elif counter > 9:
                message = f'Max iterations exceeded. Exiting loop.'
                print(message)
                self.algorithm_logger('run_price_smoothing_mppdc_case', message)
                break

            else:
                # Update dictionary of fixed variables to be used in next iteration
                fixed_vars = {v: p_r[v] for v in primal_vars}

            self.algorithm_logger('run_price_smoothing_mppdc_case', f'Finished iteration={counter}')
            counter += 1

        self.algorithm_logger('run_price_smoothing_mppdc_case', f'Finished solving model')

        # Combine results into a single dictionary
        results = {**rep_results, 'stage_3_price_targeting': i_results, 'parameters': params}

        # Save results
        self.save_results(results, output_dir, filename)

        # Method output
        output = {'mppdc_model': m_m, 'mppdc_status': m_m_status, 'primal_model': m_p, 'primal_status': m_p_status,
                  'results': results}

        self.algorithm_logger('run_price_smoothing_mppdc_case', 'Finished MPPDC case')

        # Total iterations
        total_iterations = max(i_results.keys())

        # Save summary of the solution time
        solution_summary = {'case_id': case_id, 'mode': params['mode'], 'carbon_price': params['carbon_price'],
                            'transition_year': params['transition_year'],
                            'total_solution_time': time.time() - t_start, 'total_iterations': total_iterations,
                            'max_capacity_difference': max_capacity_difference,
                            'max_baseline_difference': max_baseline_difference}
        self.save_solution_summary(solution_summary, output_dir)

        message = f"Finished MPPDC case: " + str(solution_summary)
        self.algorithm_logger('run_price_smoothing_mppdc_case', message)

        return output


if __name__ == '__main__':
    log_file_name = 'case_logger'
    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, 'output', 'local2')

    # Object used to run model cases
    cases = ModelCases(output_directory, log_file_name)

    p = {'start': 2016, 'end': 2030, 'scenarios': 5}
    rep = REP(output_directory, p)

    b = {y: 0 for y in range(p['start'], p['end'] + 1)}
    # r = rep.run_model(b, 20)

    rep.run_cases(range(5, 11, 5))