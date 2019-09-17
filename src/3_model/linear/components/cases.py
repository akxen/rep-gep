"""Generation expansion planning cases"""

import os
import sys
import copy
import pickle
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, '4_analysis'))

from pyomo.environ import *
import matplotlib.pyplot as plt

from targets import Targets
from gep import MPPDCModel, Primal, Dual
from auxiliary import BaselineUpdater
from analysis import AnalyseResults


class ModelCases:
    def __init__(self, output_dir, log_name):
        logging.basicConfig(filename=os.path.join(output_dir, f'{log_name}.log'), filemode='w',
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

        elif type(model_component) == pyomo.core.base.var.IndexedVar:
            return model_component.get_values()

        elif type(model_component) == pyomo.core.base.param.IndexedParam:
            return {k: v for k, v in model_component.items()}

        elif type(model_component) == pyomo.core.base.param.SimpleParam:
            return model_component.value

        else:
            raise Exception(f'Unexpected model component: {component_name}')

    @staticmethod
    def save_results(results, output_dir, filename):
        """Save model results"""

        with open(os.path.join(output_dir, filename), 'wb') as f:
            pickle.dump(results, f)

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

        # # Fix capacity variables
        # m.x_c.fix()
        #
        # # Re-solve to obtain correct prices
        # m, status = mppdc.solve_model(m)
        #
        # # Unfix capacity variables. Ready for next solve.
        # m.x_c.unfix()

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

        # # Fix investment decisions - re-solve to obtain correct prices
        # m.x_c.fix()
        # m, status = primal.solve_model(m)
        #
        # # Unfix investment decisions (in case following solve required)
        # m.x_c.unfix()

        return m, status

    def run_bau_case(self, first_year, final_year, scenarios_per_year, output_dir):
        """Run business-as-usual case"""

        # Permit prices and emissions intensity baselines for BAU case (all 0)
        permit_prices = {y: float(0) for y in range(first_year, final_year + 1)}
        baselines = {y: float(0) for y in range(first_year, final_year + 1)}

        # Run model
        m, status = self.run_primal_fixed_policy(first_year, final_year, scenarios_per_year, permit_prices, baselines)

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

        return m, status

    def run_carbon_tax_case(self, output_dir, first_year, final_year, scenarios_per_year, permit_prices):
        """Run carbon tax scenario"""

        # Permit prices and emissions intensity baselines for BAU case (all 0)
        baselines = {y: float(0) for y in range(first_year, final_year + 1)}

        # Run model
        m, status = self.run_fixed_policy(first_year, final_year, scenarios_per_year, permit_prices, baselines)

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

    def run_rep_case(self, first_year, final_year, scenarios_per_year, permit_prices, output_dir):
        """Run carbon tax scenario"""

        # Results to extract
        result_keys = ['x_c', 'p', 'p_V', 'p_in', 'p_out', 'q', 'p_L', 'baseline', 'permit_price', 'YEAR_EMISSIONS',
                       'YEAR_EMISSIONS_INTENSITY', 'YEAR_SCHEME_REVENUE', 'TOTAL_SCHEME_REVENUE', 'C_MC', 'ETA',
                       'DELTA', 'RHO', 'EMISSIONS_RATE', 'YEAR_SCHEME_EMISSIONS_INTENSITY',
                       'YEAR_CUMULATIVE_SCHEME_REVENUE']

        # First run carbon tax case
        baselines = {y: float(0) for y in range(first_year, final_year + 1)}

        # Run model (carbon tax case)
        m, status = self.run_primal_fixed_policy(first_year, final_year, scenarios_per_year, permit_prices, baselines)

        # Model results
        carbon_tax_results = {k: self.extract_result(m, k) for k in result_keys}

        # Add dual variable from power balance constraint
        carbon_tax_results['PRICES'] = {k: m.dual[m.POWER_BALANCE[k]] for k in m.POWER_BALANCE.keys()}

        # Update baselines so they = emissions intensity of output from participating generators
        rep_baselines = carbon_tax_results['YEAR_SCHEME_EMISSIONS_INTENSITY']

        # Container for iteration results
        iteration_results = {}

        # Flag used to terminate loop if stopping criterion met
        stop_flag = False

        # Iteration counter
        i = 1

        # Initialise result input to carbon tax scenario for first iteration (used to check stopping criterion)
        result_input = carbon_tax_results

        while not stop_flag:

            # Re-run model with new baselines
            m, status = self.run_primal_fixed_policy(first_year, final_year, scenarios_per_year, permit_prices,
                                                     rep_baselines)

            # Model results
            rep_results = {k: self.extract_result(m, k) for k in result_keys}

            # Add dual variable from power balance constraint
            rep_results['PRICES'] = {k: m.dual[m.POWER_BALANCE[k]] for k in m.POWER_BALANCE.keys()}

            # Add results to iteration results container
            iteration_results[i] = copy.deepcopy(rep_results)

            # Check if stop criterion satisfied
            cap_diff = max([abs(result_input['x_c'][k] - rep_results['x_c'][k]) for k in result_input['x_c'].keys()])
            print(f'{i}: Maximum capacity difference = {cap_diff} MW')
            if cap_diff < 5:
                stop_flag = True

            # Update input results (used to check stopping criterion in next iteration)
            result_input = copy.deepcopy(rep_results)

            # Update baselines to be used in next iteration
            rep_baselines = result_input['YEAR_SCHEME_EMISSIONS_INTENSITY']

            i += 1

        # Combine results into single dictionary
        results = {'stage_1_carbon_tax': carbon_tax_results, 'stage_2_rep': iteration_results}

        # Save results
        filename = 'rep_case.pickle'
        self.save_results(results, output_dir, filename)

        return m, status, results

    def run_price_smoothing_heuristic_case(self, params, output_dir):
        """Smooth prices over entire model horizon using approximated price functions"""

        # Model parameters
        rep_filename = params['rep_filename']

        # Load REP results
        with open(os.path.join(output_dir, rep_filename), 'rb') as f:
            rep_results = pickle.load(f)

        # Get results corresponding to last iteration of REP solution
        rep_iteration = rep_results['stage_2_rep'][max(rep_results['stage_2_rep'].keys())]

        # Extract parameters from last iteration of REP program results
        permit_prices = rep_iteration['permit_price']
        first_year = min(permit_prices.keys())
        final_year = max(permit_prices.keys())
        bau_initial_price = self.get_bau_initial_price(output_dir, first_year)
        scenarios_per_year = len([s for y, s in rep_iteration['RHO'].keys() if y == final_year])

        # Classes used to construct and run primal and MPPDC programs
        primal = Primal(first_year, final_year, scenarios_per_year)
        baseline = BaselineUpdater(first_year, final_year, scenarios_per_year)

        # Construct primal program
        m_p = primal.construct_model()

        # Results to extract from primal program
        primal_keys = ['x_c', 'p', 'p_V', 'p_in', 'p_out', 'p_L', 'baseline', 'permit_price', 'YEAR_EMISSIONS',
                       'YEAR_EMISSIONS_INTENSITY', 'YEAR_SCHEME_REVENUE', 'TOTAL_SCHEME_REVENUE', 'C_MC', 'ETA',
                       'DELTA', 'RHO', 'EMISSIONS_RATE', 'YEAR_CUMULATIVE_SCHEME_REVENUE',
                       'YEAR_SCHEME_EMISSIONS_INTENSITY']

        # Results to extract from baseline targeting model
        baseline_keys = ['YEAR_AVERAGE_PRICE', 'YEAR_AVERAGE_PRICE_0', 'YEAR_ABSOLUTE_PRICE_DIFFERENCE']

        iteration_results = {}
        stop_flag = False
        counter = 1

        # Initialise price setting generator input as results obtained from final REP iteration
        psg_input = rep_iteration

        while not stop_flag:
            # Identify price setting generators
            psg = baseline.prices.get_price_setting_generators_from_model_results(psg_input)

            # Construct model
            m_b = baseline.construct_model(psg)

            # Update parameters
            m_b = baseline.update_parameters(m_b, psg_input)
            m_b.YEAR_AVERAGE_PRICE_0 = float(bau_initial_price)
            m_b.SCHEME_REVENUE_ENVELOPE_LO.store_values(params['revenue_envelope_lo'])
            m_b.PRICE_WEIGHTS.store_values(params['price_weights'])

            # Activate constraints
            m_b.SCHEME_REVENUE_ENVELOPE_LO_CONS.activate()

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

            # Get results
            r_p = copy.deepcopy({v: self.extract_result(m_p, v) for v in primal_keys})
            r_p['PRICES'] = copy.deepcopy({k: m_p.dual[m_p.POWER_BALANCE[k]] for k in m_p.POWER_BALANCE.keys()})
            iteration_results[counter] = {'primal': r_p, 'auxiliary': r_b}

            # Max difference in capacity sizing decision between iterations
            max_cap_difference = max(abs(psg_input['x_c'][k] - m_p.x_c[k].value) for k in m_p.x_c.keys())
            print(f'Max capacity difference: {max_cap_difference} MW')

            # Check if capacity variables have changed
            if max_cap_difference < 5:
                stop_flag = True

            else:
                # Update dictionary of price setting generator program inputs
                psg_input = r_p

            counter += 1

            # Combine results into a single dictionary
        combined_results = {**rep_results, 'stage_3_price_targeting': iteration_results}

        # Save results
        self.save_results(combined_results, output_dir, f'price_targeting_heuristic_case.pickle')

        return m_b, m_b_status, m_p, m_p_status, combined_results

    def run_price_smoothing_mppdc_case(self, params, output_dir):
        """Run case to smooth prices over model horizon, subject to total revenue constraint"""

        # Model parameters
        rep_filename = params['rep_filename']

        # Load REP results
        with open(os.path.join(output_dir, rep_filename), 'rb') as f:
            rep_results = pickle.load(f)

        # Get results corresponding to last iteration of REP solution
        rep_iteration = rep_results['stage_2_rep'][max(rep_results['stage_2_rep'].keys())]

        # Extract parameters from last iteration of REP program results
        permit_prices = rep_iteration['permit_price']
        first_year = min(permit_prices.keys())
        final_year = max(permit_prices.keys())
        bau_initial_price = self.get_bau_initial_price(output_dir, first_year)
        scenarios_per_year = len([s for y, s in rep_iteration['RHO'].keys() if y == final_year])

        # Classes used to construct and run primal and MPPDC programs
        mppdc = MPPDCModel(first_year, final_year, scenarios_per_year)
        primal = Primal(first_year, final_year, scenarios_per_year)

        # Construct MPPDC
        m_m = mppdc.construct_model(include_primal_constraints=True)

        # Construct primal model
        m_p = primal.construct_model()

        # Update MPPDC model parameters
        m_m.YEAR_AVERAGE_PRICE_0 = float(bau_initial_price)
        m_m.SCHEME_REVENUE_ENVELOPE_LO.store_values(params['revenue_envelope_lo'])
        m_m.PRICE_WEIGHTS.store_values(params['price_weights'])

        # Activate necessary constraints
        m_m.SCHEME_REVENUE_ENVELOPE_LO_CONS.activate()

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
                      'YEAR_ABSOLUTE_PRICE_DIFFERENCE', 'YEAR_AVERAGE_PRICE_0', 'YEAR_AVERAGE_PRICE']

        # Stop flag and iteration counter
        stop_flag = False
        counter = 1

        # Container for iteration results
        iteration_results = {}

        while not stop_flag:
            # Fix MPPDC variables
            m_m = mppdc.fix_variables(m_m, fixed_vars)

            # Solve MPPDC
            m_m, m_m_status = mppdc.solve_model(m_m)

            # Results from MPPDC program
            r_m = copy.deepcopy({v: self.extract_result(m_m, v) for v in mppdc_keys})
            iteration_results[counter] = r_m

            # Update primal program with baselines and permit prices obtained from MPPDC model
            for y in m_p.Y:
                m_p.baseline[y].fix(m_m.baseline[y].value)
                m_p.permit_price[y].fix(m_m.permit_price[y].value)

            # Solve primal model
            m_p, m_p_status = primal.solve_model(m_p)

            # Results from primal program
            p_r = copy.deepcopy({v: self.extract_result(m_p, v) for v in primal_vars})
            p_r['PRICES'] = copy.deepcopy({k: m_p.dual[m_p.POWER_BALANCE[k]] for k in m_p.POWER_BALANCE.keys()})
            iteration_results[counter]['primal'] = p_r

            # Max difference in capacity sizing decision between MPPDC and primal program
            max_cap_difference = max(abs(m_m.x_c[k].value - m_p.x_c[k].value) for k in m_m.x_c.keys())
            print(f'Max capacity difference: {max_cap_difference} MW')

            # Check if capacity variables have changed
            if max_cap_difference < 5:
                stop_flag = True

            else:
                # Update dictionary of fixed variables to be used in next iteration
                fixed_vars = {v: p_r[v] for v in primal_vars}

            counter += 1

            # Combine results into a single dictionary
        combined_results = {**rep_results, 'stage_3_price_targeting': iteration_results}

        # Save results
        self.save_results(combined_results, output_dir, f'price_targeting_mppdc_case.pickle')

        return m_m, m_m_status, m_p, m_p_status, combined_results

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

    # Object used to assist in calculation of target trajectories (e.g. price weights)
    targets = Targets()

    # Common model parameters
    start, end, scenarios = 2016, 2026, 10

    # Year when scheme transitions to a Refunded Emissions Payment (REP) scheme
    transition_year = 2021

    # Permit prices for carbon pricing scenarios
    permit_prices_model = {y: float(40) for y in range(start, end + 1)}
    scheme_revenue_envelope_lo = {y: targets.get_envelope(-100e6, 4, start, y) if y < transition_year else float(0)
                                  for y in range(start, end + 1)}
    scheme_price_weights = {y: targets.get_envelope(1, 4, start, y) for y in range(start, end + 1)}

    # Define case parameters and run model
    case_params = {'rep_filename': 'rep_case.pickle',
                   'revenue_envelope_lo': scheme_revenue_envelope_lo,
                   'price_weights': scheme_price_weights,
                   'transition_year': transition_year}

    # Run BAU case
    mo_bau, mo_bau_status = cases.run_bau_case(start, end, scenarios, output_directory)

    # Run REP case
    mo_rep, mo_rep_status, r_rep = cases.run_rep_case(start, end, scenarios, permit_prices_model, output_directory)

    # Run price case targeting model using MPPDC model
    mo_m, mo_m_status, mo_p, mo_p_status, r_ptm = cases.run_price_smoothing_mppdc_case(case_params, output_directory)

    # Run price case targeting model using auxiliary model
    mo_b, mo_b_status, mo_ph, mo_ph_status, r_pth = cases.run_price_smoothing_heuristic_case(case_params,
                                                                                             output_directory)

    # Check prices
    p_m = mo_m.lamb.get_values()
    p_p = {k: mo_p.dual[mo_p.POWER_BALANCE[k]] for k in mo_p.POWER_BALANCE.keys()}

    # Absolute price difference, and max difference
    p_diff = {k: abs(abs(p_m[k]) - abs(p_p[k])) for k in p_m.keys()}
    p_diff_max = max(p_diff.values())

    # Check baselines from both plots. Include lower scheme revenue envelope
    fig, ax = plt.subplots()
    ax.plot(list(mo_m.baseline.get_values().values()))
    ax.plot(list(mo_b.baseline.get_values().values()))
    plt.show()
