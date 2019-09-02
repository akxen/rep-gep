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

        else:
            raise Exception(f'Unexpected model component: {component_name}')

    @staticmethod
    def transfer_primal_solution_to_mppdc(m_p, m_m, vars_to_fix):
        """Transfer solution from primal model to MPPDC model"""

        for v in vars_to_fix:
            # Get values obtained from primal model
            primal_values = m_p.__getattribute__(v).get_values()

            # Fix values in MPPDC model
            m_m.__getattribute__(v).set_values(primal_values)

            # Fix MPPDC values
            m_m.__getattribute__(v).fix()

        return m_m

    @staticmethod
    def save_results(results, output_dir, filename):
        """Save model results"""

        with open(os.path.join(output_dir, filename), 'wb') as f:
            pickle.dump(results, f)

    def run_mppdc_fixed_policy(self, baselines, permit_prices, final_year, scenarios_per_year, result_keys):
        """Run BAU case"""

        # Initialise object and model used to run MPPDC model
        mppdc = MPPDCModel(final_year, scenarios_per_year)
        m = mppdc.construct_model()

        # Fix permit prices and baselines
        m.permit_price.store_values(permit_prices)
        m.permit_price.fix()

        m.baseline.store_values(baselines)
        m.baseline.fix()

        # Solve MPPDC model with fixed policy parameters
        m, status = mppdc.solve_model(m)

        # Model results
        results = {k: self.extract_result(m, k) for k in result_keys}

        return results

    def run_primal_fixed_policy(self, baselines, permit_prices, final_year, scenarios_per_year, result_keys):
        """Run primal model with fixed policy parameters"""

        # Initialise object and model used to run primal model
        primal = Primal(final_year, scenarios_per_year)
        m = primal.construct_model()

        # Fix permit prices and baselines
        for y in m.Y:
            m.permit_price[y].fix(permit_prices[y])
            m.baseline[y].fix(baselines[y])

        # Solve primal model with fixed policy parameters
        m, status = primal.solve_model(m)

        # Model results
        results = {k: self.extract_result(m, k) for k in result_keys}

        # Add dual variable from power balance constraint
        results['PRICES'] = {k: m.dual[m.POWER_BALANCE[k]] for k in m.POWER_BALANCE.keys()}

        return results

    def run_primal_cumulative_emissions_cap(self, final_year, scenarios_per_year, emissions_cap, result_keys):
        """Run primal emissions cap scenario"""

        # Initialise object and model used to run primal model
        primal = Primal(final_year, scenarios_per_year)
        m = primal.construct_model()

        # Fix permit prices and baselines
        m.permit_price.fix(0)
        m.baseline.fix(0)

        # Update cumulative emissions cap parameter
        m.CUMULATIVE_EMISSIONS_CAP = float(emissions_cap)

        # Add cumulative emissions cap constraint
        m.CUMULATIVE_EMISSIONS_CAP_CONS.activate()

        # Solve primal model with fixed policy parameters
        m, status = primal.solve_model(m)

        # Model results
        results = {k: self.extract_result(m, k) for k in result_keys}

        # Add dual variable from power balance constraint
        results['PRICES'] = {k: m.dual[m.POWER_BALANCE[k]] for k in m.POWER_BALANCE.keys()}

        # Add dual variable associated with cumulative emissions cap constraint
        results['CUMULATIVE_EMISSIONS_CAP_CONS_DUAL'] = m.dual[m.CUMULATIVE_EMISSIONS_CAP_CONS]

        return results

    def run_primal_interim_emissions_cap(self, final_year, scenarios_per_year, interim_emissions_cap, result_keys):
        """Run primal emissions cap scenario"""

        # Initialise object and model used to run primal model
        primal = Primal(final_year, scenarios_per_year)
        m = primal.construct_model()

        # Fix permit prices and baselines
        m.permit_price.fix(0)
        m.baseline.fix(0)

        # Update cumulative emissions cap parameter
        for y in m.Y:
            m.INTERIM_EMISSIONS_CAP[y] = float(interim_emissions_cap[y])

        # Add interim emissions cap constraints
        m.INTERIM_EMISSIONS_CAP_CONS.activate()

        # Solve primal model with fixed policy parameters
        m, status = primal.solve_model(m)

        # Model results
        results = {k: self.extract_result(m, k) for k in result_keys}

        # Add dual variable from power balance constraint
        results['PRICES'] = {k: m.dual[m.POWER_BALANCE[k]] for k in m.POWER_BALANCE.keys()}

        # Add dual variable associated with interim emissions cap constraints
        results['INTERIM_EMISSIONS_CAP_CONS_DUAL'] = {y: m.dual[m.INTERIM_EMISSIONS_CAP_CONS[y]] for y in m.Y}

        return results

    def run_bau_case(self, output_dir, final_year, scenarios_per_year, mode='primal'):
        """Run business-as-usual case"""

        # Initialise object and model used to run primal model
        primal = Primal(final_year, scenarios_per_year)
        primal_dummy_model = primal.construct_model()

        # Baselines and permit prices = 0 for all years in model horizon
        baselines = {y: float(0) for y in primal_dummy_model.Y}
        permit_prices = {y: float(0) for y in primal_dummy_model.Y}

        # Results to extract
        result_keys = ['x_c', 'p', 'p_V', 'p_in', 'p_out', 'p_L', 'baseline', 'permit_price', 'YEAR_EMISSIONS',
                       'YEAR_EMISSIONS_INTENSITY', 'YEAR_SCHEME_REVENUE', 'TOTAL_SCHEME_REVENUE']

        if mode == 'primal':
            # Run primal BAU case
            results = self.run_primal_fixed_policy(baselines, permit_prices, final_year, scenarios_per_year,
                                                   result_keys)
            filename = 'primal_bau_case.pickle'

        elif mode == 'mppdc':
            # Include power balance constraint dual variable in results to extract
            result_keys += ['lamb']

            # Run MPPDC BAU case
            results = self.run_mppdc_fixed_policy(baselines, permit_prices, final_year, scenarios_per_year, result_keys)
            filename = 'mppdc_bau_case.pickle'

        else:
            raise Exception(f'Unexpected run mode: {mode}')

        # Save results
        with open(os.path.join(output_dir, filename), 'wb') as f:
            pickle.dump(results, f)

        return results

    def run_cumulative_emissions_cap_case(self, output_dir, final_year, scenarios_per_year, emissions_cap):
        """Run case with a cumulative emissions cap"""

        # Results to extract
        result_keys = ['x_c', 'p', 'p_V', 'p_in', 'p_out', 'p_L', 'baseline', 'permit_price', 'YEAR_EMISSIONS',
                       'YEAR_EMISSIONS_INTENSITY', 'YEAR_SCHEME_REVENUE', 'TOTAL_SCHEME_REVENUE']

        # Run primal model with cumulative emissions cap
        results = self.run_primal_cumulative_emissions_cap(final_year, scenarios_per_year, emissions_cap, result_keys)

        # Save results
        with open(os.path.join(output_dir, 'cumulative_emissions_cap_results.pickle'), 'wb') as f:
            pickle.dump(results, f)

        return results

    def run_interim_emissions_cap_case(self, output_dir, final_year, scenarios_per_year, interim_emissions_cap):
        """Run case with interim emissions cap - cap defined for each year of model horizon"""

        # Results to extract
        result_keys = ['x_c', 'p', 'p_V', 'p_in', 'p_out', 'p_L', 'baseline', 'permit_price', 'YEAR_EMISSIONS',
                       'YEAR_EMISSIONS_INTENSITY', 'YEAR_SCHEME_REVENUE', 'TOTAL_SCHEME_REVENUE']

        # Run primal model with interim emissions cap
        results = self.run_primal_interim_emissions_cap(final_year, scenarios_per_year, interim_emissions_cap,
                                                        result_keys)

        # Save results
        with open(os.path.join(output_dir, 'interim_emissions_cap_results.pickle'), 'wb') as f:
            pickle.dump(results, f)

        return results

    def run_carbon_tax_case(self, output_dir, final_year, scenarios_per_year, permit_prices):
        """Run carbon tax case (baseline = 0 for all years in model horizon)"""

        # Results to extract
        result_keys = ['x_c', 'p', 'p_V', 'p_in', 'p_out', 'p_L', 'baseline', 'permit_price', 'YEAR_EMISSIONS',
                       'YEAR_EMISSIONS_INTENSITY', 'YEAR_SCHEME_REVENUE', 'TOTAL_SCHEME_REVENUE']

        # Baselines = 0 for all years in model horizon
        baselines = {y: float(0) for y in range(2016, final_year + 1)}

        # Run policy with fixed baselines and permit prices
        results = self.run_primal_fixed_policy(baselines, permit_prices, final_year, scenarios_per_year, result_keys)

        with open(os.path.join(output_dir, 'carbon_tax_case.pickle'), 'wb') as f:
            pickle.dump(results, f)

        return results

    def run_rep_case(self, output_dir, final_year, scenarios_per_year, permit_prices):
        """Run refunded emissions payment scheme algorithm"""
        # Results to extract
        result_keys = ['x_c', 'p', 'p_V', 'p_in', 'p_out', 'p_L', 'q', 'baseline', 'permit_price', 'YEAR_EMISSIONS',
                       'YEAR_EMISSIONS_INTENSITY', 'YEAR_SCHEME_EMISSIONS_INTENSITY', 'YEAR_SCHEME_REVENUE',
                       'TOTAL_SCHEME_REVENUE']

        # Run carbon tax case
        carbon_tax_baselines = {y: float(0) for y in range(2016, final_year + 1)}
        carbon_tax_results = self.run_primal_fixed_policy(carbon_tax_baselines, permit_prices, final_year,
                                                          scenarios_per_year, result_keys)

        # Update baselines so they = emissions intensity of output from participating generators
        rep_baselines = carbon_tax_results['YEAR_SCHEME_EMISSIONS_INTENSITY']
        rep_results = self.run_primal_fixed_policy(rep_baselines, permit_prices, final_year, scenarios_per_year,
                                                   result_keys)

        # Combine results into single dictionary
        results = {'stage_1_carbon_tax': carbon_tax_results, 'stage_2_rep': rep_results}

        # Save results
        self.save_results(results, output_dir, 'rep_case.pickle')

        return results

    def run_price_smoothing_mppdc_case(self, output_dir, final_year, scenarios_per_year, permit_prices, mode):
        """Run case to smooth prices over model horizon, subject to total revenue constraint"""

        # Try to load REP results from previous run
        try:
            with open(os.path.join(output_dir, 'rep_case.pickle'), 'rb') as f:
                rep_results = pickle.load(f)
        except Exception as e:
            print(f'ModelCases.run_price_smoothing_mppdc_case: ' + str(e))
            rep_results = self.run_rep_case(output_dir, final_year, scenarios_per_year, permit_prices)

        # Construct MPPDC
        mppdc = MPPDCModel(final_year, scenarios_per_year)
        mppdc_model = mppdc.construct_model(include_primal_constraints=False)

        # Fix primal variables to results obtained from REP case (including permit price)
        primal_variables = ['x_c', 'p', 'p_in', 'p_out', 'q', 'p_V', 'p_L', 'permit_price']
        fixed_variables = {k: rep_results['stage_2_rep'][k] for k in primal_variables}
        mppdc_model = mppdc.fix_variables(mppdc_model, fixed_variables)

        if mode == 'non_negative_revenue':
            # Activate non-negative revenue constraint
            mppdc_model.TOTAL_SCHEME_REVENUE_NON_NEGATIVE_CONS.activate()

        elif mode == 'neutral_revenue':
            # Activate revenue neutrality constraint
            mppdc_model.TOTAL_NET_SCHEME_REVENUE_NEUTRAL_CONS.activate()

        elif mode == 'neutral_revenue_lower_bound':
            # Activate revenue neutrality constraint + lower scheme revenue constraint
            mppdc_model.TOTAL_NET_SCHEME_REVENUE_NEUTRAL_CONS.activate()
            mppdc_model.CUMULATIVE_NET_SCHEME_REVENUE_LB_CONS.activate()

        else:
            raise Exception(f'Unexpected mode: {mode}')

        # Solve model
        mppdc_model, solve_status = mppdc.solve_model(mppdc_model)

        # Results to extract
        result_keys = ['x_c', 'p', 'p_V', 'p_in', 'p_out', 'p_L', 'q', 'baseline', 'permit_price', 'lamb',
                       'YEAR_EMISSIONS', 'YEAR_EMISSIONS_INTENSITY', 'YEAR_SCHEME_EMISSIONS_INTENSITY',
                       'YEAR_SCHEME_REVENUE', 'TOTAL_SCHEME_REVENUE']

        # Extract model results
        mppdc_results = {'stage_3_mppdc': {k: self.extract_result(mppdc_model, k) for k in result_keys}}

        # Combine results into a single dictionary
        combined_results = {**rep_results, **mppdc_results}

        # Save results
        self.save_results(combined_results, output_directory, f'mppdc_{mode}_case.pickle')

        return combined_results

    def run_primal_to_check_mppdc_solution(self, output_dir, final_year, scenarios_per_year, mode):
        """Load permit prices and baselines obtained from MPPDC solution"""
        # TODO:Check that solutions match when running primal model with same baseline and permit prices

        with open(os.path.join(output_dir, f'mppdc_{mode}_case.pickle'), 'rb') as f:
            mppdc_results = pickle.load(f)

        # Load baselines and permit prices from corresponding MPPDC solution
        baselines = mppdc_results['stage_3_mppdc']['baseline']
        permit_prices = mppdc_results['stage_3_mppdc']['permit_prices']

        # Results to extract from primal model
        result_keys = ['x_c', 'p', 'p_V', 'p_in', 'p_out', 'p_L', 'q', 'baseline', 'permit_price',
                       'YEAR_EMISSIONS', 'YEAR_EMISSIONS_INTENSITY', 'YEAR_SCHEME_EMISSIONS_INTENSITY',
                       'YEAR_SCHEME_REVENUE', 'TOTAL_SCHEME_REVENUE']

        # Primal results
        results = self.run_primal_fixed_policy(baselines, permit_prices, final_year, scenarios_per_year, result_keys)

        # Save results
        self.save_results(results, output_directory, f'primal_{mode}_case_check.pickle')

        return results

    def get_permit_price_trajectory(self, primal, model, target_emissions_trajectory, baselines, initial_permit_prices,
                                    permit_price_tol, permit_price_cap):
        """Run algorithm to identify sequence of permit prices that achieves a given emissions intensity trajectory"""

        # Name of function - used by logger
        function_name = 'get_permit_price_trajectory'

        # Initialise results container
        results = {'target_emissions_trajectory': target_emissions_trajectory, 'iteration_results': {}}

        # Initialise permit prices for each year in model horizon
        permit_prices = initial_permit_prices

        # Update emissions intensity baseline. Set baseline = target emissions intensity trajectory
        for y in model.Y:
            model.baseline[y].fix(baselines[y])

        # Iteration counter
        i = 1

        while True:
            self.algorithm_logger(function_name, f'Running iteration {i}', print_message=True)

            # Fix permit prices
            for y in model.Y:
                model.permit_price[y].fix(permit_prices[y])

            self.algorithm_logger(function_name, f'Model permit prices: {model.permit_price.get_values()}',
                                  print_message=True)

            # Run model
            model, solver_status = primal.solve_model(model)
            self.algorithm_logger(function_name, 'Solved model')

            # Latest emissions intensities
            latest_emissions_intensities = self.extract_result(model, 'YEAR_EMISSIONS_INTENSITY')
            self.algorithm_logger(function_name, f'Emissions intensities: {latest_emissions_intensities}',
                                  print_message=True)

            # Compute difference between actual emissions intensity and target
            emissions_intensity_difference = {y: latest_emissions_intensities[y] - target_emissions_trajectory[y]
                                              for y in model.Y}
            self.algorithm_logger(function_name,
                                  f'Emissions intensity target difference: {emissions_intensity_difference}',
                                  print_message=True)

            # Container for new permit prices
            new_permit_prices = {}

            # Update permit prices
            for y in model.Y:
                new_permit_prices[y] = permit_prices[y] + (emissions_intensity_difference[y] * 200)

                # Set equal to zero if permit price is less than 0
                if new_permit_prices[y] < 0:
                    new_permit_prices[y] = 0

                # Set equal to price cap is permit price exceeds cap
                elif new_permit_prices[y] > permit_price_cap:
                    new_permit_prices[y] = permit_price_cap

            self.algorithm_logger(function_name, f'Updated permit prices: {new_permit_prices}', print_message=True)

            # Compute max difference between new and old permit prices
            max_permit_price_difference = primal.utilities.get_max_absolute_difference(new_permit_prices, permit_prices)

            # Copying results from model object into a dictionary
            result_keys = ['baseline', 'permit_price', 'YEAR_EMISSIONS', 'YEAR_EMISSIONS_INTENSITY',
                           'YEAR_SCHEME_REVENUE', 'TOTAL_SCHEME_REVENUE']

            # Store selected results
            results['iteration_results'][i] = {k: self.extract_result(model, k) for k in result_keys}

            # Check if the max difference is less than the tolerance
            if max_permit_price_difference < permit_price_tol:
                message = f"""
                Max permit price difference: {max_permit_price_difference}
                Tolerance: {permit_price_tol}
                Exiting algorithm
                """
                self.algorithm_logger(function_name, message, print_message=True)

                return model, results

            else:
                # update permit prices to use in next iteration
                permit_prices = new_permit_prices

                # Update iteration counter
                i += 1

    def run_carbon_tax_permit_price_trajectory_case(self, output_dir, final_year, scenarios_per_year,
                                                    target_emissions_trajectory, permit_price_tol, permit_price_cap):
        """Run carbon tax case (no refunding)"""

        # Initialise object and model used to run primal model
        primal = Primal(final_year, scenarios_per_year)
        primal_model = primal.construct_model()

        # Baselines = 0 for all years in model horizon
        baselines = {y: float(0) for y in primal_model.Y}

        # Initial permit prices (assume = 0 for all years)
        initial_permit_prices = {y: float(0) for y in primal_model.Y}

        # Run algorithm to identify permit price trajectory that achieve emissions trajectory target
        primal_model, permit_price_results = self.get_permit_price_trajectory(primal, primal_model,
                                                                              target_emissions_trajectory,
                                                                              baselines, initial_permit_prices,
                                                                              permit_price_tol, permit_price_cap)

        # Combine results into single dictionary
        results = {'permit_price_trajectory': permit_price_results}

        # Extract results from model
        result_keys = ['x_c', 'p', 'p_V', 'p_in', 'p_out', 'p_L', 'baseline', 'permit_price', 'YEAR_EMISSIONS',
                       'YEAR_EMISSIONS_INTENSITY', 'YEAR_SCHEME_REVENUE', 'TOTAL_SCHEME_REVENUE']

        # Model results
        results['final_iteration'] = {k: self.extract_result(primal_model, k) for k in result_keys}

        # Save results
        filename = 'carbon_tax_results.pickle'
        self.save_results(results, output_dir, filename)

        return results

    def run_algorithm(self, output_dir, final_year, scenarios_per_year, target_emissions_trajectory, baseline_tol,
                      permit_price_tol, permit_price_cap, case):
        """Run price smoothing case where non-negative revenue is enforced"""

        # Name of function
        function_name = 'run_algorithm'

        # Model parameters
        parameters = {'final_year': final_year, 'scenarios_per_year': scenarios_per_year,
                      'target_emissions_trajectory': target_emissions_trajectory, 'baseline_tol': baseline_tol,
                      'permit_price_tol': permit_price_tol, 'permit_price_cap': permit_price_cap, 'case': case}

        self.algorithm_logger(function_name, f'Running algorithm with parameters: {parameters}', print_message=True)

        # Container for model results
        results = {'parameters': parameters, 'baseline_iteration': {}}

        # Construct primal model
        primal = Primal(final_year, scenarios_per_year)
        primal_model = primal.construct_model()
        self.algorithm_logger(function_name, f'Constructed primal model', print_message=True)

        # Construct MPPDC
        mppdc = MPPDCModel(final_year, scenarios_per_year)
        mppdc_model = mppdc.construct_model()
        self.algorithm_logger(function_name, f'Constructed MPPDC model', print_message=True)

        # Activate constraints required for case
        if case == 'price_smoothing_non_negative_revenue':
            mppdc_model.TOTAL_SCHEME_REVENUE_NON_NEGATIVE_CONS.activate()

        elif case == 'price_smoothing_neutral_revenue':
            mppdc_model.TOTAL_NET_SCHEME_REVENUE_NEUTRAL_CONS.activate()

        elif case == 'price_smoothing_neutral_revenue_lower_bound':
            mppdc_model.TOTAL_NET_SCHEME_REVENUE_NEUTRAL_CONS.activate()
            mppdc_model.CUMULATIVE_NET_SCHEME_REVENUE_LB_CONS.activate()

        elif case == 'rep':
            mppdc_model.YEAR_NET_SCHEME_REVENUE_NEUTRAL_CONS.activate()

        elif case == 'transition':
            # Set the transition year
            transition_year = 2017
            mppdc_model.TRANSITION_YEAR = 2017

            # Record transition year in model parameters dictionary
            results['parameters']['transition_year'] = transition_year

            # Activate year revenue neutrality requirement for all years
            mppdc_model.YEAR_NET_SCHEME_REVENUE_NEUTRAL_CONS.activate()

            # Deactivate year revenue neutrality requirement for years before transition year
            for y in range(2016, transition_year + 1):
                mppdc_model.YEAR_NET_SCHEME_REVENUE_NEUTRAL_CONS[y].deactivate()

            # Reconstruct and enforce revenue neutrality over the transition period
            mppdc_model.TRANSITION_NET_SCHEME_REVENUE_NEUTRAL_CONS.reconstruct()
            mppdc_model.TRANSITION_NET_SCHEME_REVENUE_NEUTRAL_CONS.activate()

        else:
            self.algorithm_logger(function_name, f'Unexpected case: {case}')
            raise Exception(f'Unexpected case: {case}')

        # Initialise baselines = target emissions intensity trajectory
        baselines = target_emissions_trajectory

        # Initialise permit prices = 0 for first iteration
        initial_permit_prices = {y: float(0) for y in primal_model.Y}

        # Iteration counter
        i = 1

        while True:
            self.algorithm_logger(function_name, f'Performing iteration {i}', print_message=True)

            # Initialise container for iteration results
            baseline_iteration = {'baselines': baselines}

            self.algorithm_logger(function_name, f'Fixing baselines {baselines}')
            for k, v in baselines.items():
                primal_model.baseline[k].fix(v)

            # Implement permit price trajectory algorithm
            self.algorithm_logger(function_name, f'Solving for permit price trajectory')
            primal_model, permit_price_results = self.get_permit_price_trajectory(primal, primal_model,
                                                                                  target_emissions_trajectory,
                                                                                  baselines, initial_permit_prices,
                                                                                  permit_price_tol, permit_price_cap)

            self.algorithm_logger(function_name, f'Solved permit price trajectory: {permit_price_results}')

            # Save permit price trajectory results to dictionary
            baseline_iteration['permit_price_results'] = permit_price_results

            # Fix p, x_c, and baseline, and permit_price based on primal model solution
            mppdc_model = self.transfer_primal_solution_to_mppdc(primal_model, mppdc_model,
                                                                 vars_to_fix=['p', 'x_c', 'permit_price', 'baseline'])
            self.algorithm_logger(function_name, f'Fixed primal variables')

            # Unfixing baseline
            mppdc_model.baseline.unfix()

            # Solving MPPDC
            mppdc_model, model_status = mppdc.solve_model(mppdc_model)
            self.algorithm_logger(function_name, f'Solved MPPDC model: {model_status}')

            # New baseline trajectory
            latest_baselines = mppdc_model.baseline.get_values()
            self.algorithm_logger(function_name, f'New baseline trajectory: {latest_baselines}')

            # Check max difference between baselines from this iteration and last
            max_baseline_difference = primal.utilities.get_max_absolute_difference(latest_baselines, baselines)
            self.algorithm_logger(function_name,
                                  f'Max baseline difference between successive iterations: {max_baseline_difference}')

            # Store baseline iteration results
            results['baseline_iteration'][i] = baseline_iteration

            # If max difference < tol stop; else update baseline and repeat procedure
            if max_baseline_difference < baseline_tol:

                message = f"""
                Max baseline difference between successive iterations: {max_baseline_difference}
                Difference less than tolerance: {baseline_tol}
                Terminating algorithm"""
                self.algorithm_logger(function_name, message, print_message=True)

                break

            else:
                # Update baselines
                baselines = latest_baselines

                # Update initial permit prices to be used in next iteration for permit price trajectory algorithm
                max_iteration = max(permit_price_results['iteration_results'].keys())
                initial_permit_prices = permit_price_results['iteration_results'][max_iteration]['permit_price']

                message = f"""
                Max baseline difference between successive iterations: {max_baseline_difference}
                Difference greater than tolerance: {baseline_tol}
                Updated baselines: {baselines}"""
                self.algorithm_logger(function_name, message, print_message=True)

                # Update iteration counter
                i += 1

        # Extract results from final MPPDC model
        result_keys = ['x_c', 'p', 'p_V', 'p_in', 'p_out', 'p_L', 'baseline', 'permit_price', 'lamb', 'YEAR_EMISSIONS',
                       'YEAR_EMISSIONS_INTENSITY', 'YEAR_SCHEME_REVENUE', 'TOTAL_SCHEME_REVENUE', 'YEAR_AVERAGE_PRICE']

        # Model results
        results['final_iteration'] = {k: self.extract_result(mppdc_model, k) for k in result_keys}

        # Filename
        filename = f'{case}_results.pickle'

        # Save results
        with open(os.path.join(output_dir, filename), 'wb') as f:
            pickle.dump(results, f)

        return results


if __name__ == '__main__':
    output_directory = '.'
    log_file_name = 'case_logger'
    cases = ModelCases(output_directory, log_file_name)

    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, 'output', 'local')
    start_model_year = 2016
    final_year_model = 2020
    scenarios_per_year_model = 2
    permit_prices_model = {y: float(40) for y in range(start_model_year, final_year_model + 1)}

    # Run REP case algorithm
    rep_res = cases.run_rep_case(output_directory, final_year_model, scenarios_per_year_model, permit_prices_model)

    # Run MPPDC price smoothing algorithm - non-negative revenue constraint
    non_neg_rev = cases.run_price_smoothing_mppdc_case(output_directory, final_year_model, scenarios_per_year_model,
                                                       permit_prices_model, 'non_negative_revenue')

    non_neg_rev_check = cases.run_primal_to_check_mppdc_solution(output_directory, final_year_model,
                                                                 scenarios_per_year_model, 'non_negative_revenue')

    # Run MPPDC price smoothing algorithm - neutral revenue constraint
    neutral_rev = cases.run_price_smoothing_mppdc_case(output_directory, final_year_model, scenarios_per_year_model,
                                                       permit_prices_model, 'neutral_revenue')

    neutral_rev_check = cases.run_primal_to_check_mppdc_solution(output_directory, final_year_model,
                                                                 scenarios_per_year_model, 'neutral_revenue')

    # Run MPPDC price smoothing algorithm - neutral revenue constraint with lower bound constraint on cumulative revenue
    neutral_rev_lb = cases.run_price_smoothing_mppdc_case(output_directory, final_year_model, scenarios_per_year_model,
                                                          permit_prices_model, 'neutral_revenue_lower_bound')

    neutral_rev_lb_check = cases.run_primal_to_check_mppdc_solution(output_directory, final_year_model,
                                                                    scenarios_per_year_model,
                                                                    'neutral_revenue_lower_bound')

