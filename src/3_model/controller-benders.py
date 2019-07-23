"""Run algorithm to find least-cost investment plan"""

import os
import sys
import copy
import pickle
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.master import InvestmentPlan
from utils.subproblem import UnitCommitment
from utils.calculations import Calculations


class BendersAlgorithmController:
    def __init__(self,
                 master_solution_dir=os.path.join(os.path.dirname(__file__), 'output', 'investment_plan'),
                 subproblem_solution_dir=os.path.join(os.path.dirname(__file__), 'output', 'dispatch_plan')):

        # Setup logger
        logging.basicConfig(filename='controller-benders.log', filemode='w',
                            format='%(asctime)s %(name)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.DEBUG)

        self.logger = logging.getLogger('benders')

        # Object used to control master problem
        self.master = InvestmentPlan()

        # Object used to control unit commitment (UC) subproblems
        self.subproblem = UnitCommitment()

        # Object used to compute upper and lower bounds based on model solutions
        self.calculations = Calculations()

        # Directory containing master problem solutions
        self.master_solution_dir = master_solution_dir

        # Directory containing subproblem (UC) output files
        self.subproblem_solution_dir = subproblem_solution_dir

        # Containers for model objects
        self.master_models = {}
        self.subproblem_models = {}

    def print_log(self, message):
        """Print message to console and create entry in log file"""

        print(message)
        self.logger.info(message)

    def get_solution_upper_bound(self, m, iteration):
        """Get solution upper-bound"""

        # Upper bound for total cost
        upper_bound = self.calculations.get_upper_bound(m, iteration, self.master_solution_dir,
                                                        self.subproblem_solution_dir)

        return upper_bound

    def get_solution_lower_bound(self, iteration):
        """Get solution lower-bound"""

        # Load investment plan solution
        with open(os.path.join(self.master_solution_dir, f'investment-results_{iteration}.pickle'), 'rb') as f:
            investment_solution = pickle.load(f)

        # Total cost including Benders auxiliary variable
        lower_bound = investment_solution['OBJECTIVE']

        return lower_bound

    def convergence_check(self, m, iteration, relative_bound_tolerance=0.05):
        """Check if model has converged"""

        upper = self.get_solution_upper_bound(m, iteration)
        self.print_log(f'Solution upper bound: {upper}')

        lower = self.get_solution_lower_bound(iteration)
        self.print_log(f'Solution lower bound: {lower}')

        # Difference between upper and lower bound
        bound_difference = upper - lower
        self.print_log('Absolute difference between upper and lower bounds: {bound_difference}')

        bound_relative_difference = abs(bound_difference / lower)
        self.print_log(f'Relative difference ((upper - lower)/ lower): {bound_relative_difference}')

        if bound_relative_difference < relative_bound_tolerance:
            self.print_log(f'Model converged')
            return True

        else:
            return False

    def run_benders(self):
        """Run Benders decomposition algorithm"""

        # Remove pickle files in results directories
        self.cleanup_results(self.subproblem_solution_dir)
        self.cleanup_results(self.master_solution_dir)

        logging.info("Running Benders decomposition algorithm\nUC - constructing model")
        model_uc = self.subproblem.construct_model()

        logging.info('INV - constructing model')
        model_inv = self.master.construct_model()

        for i in range(1, 100):
            self.print_log(f"Performing iteration {i}\n{''.join(['-'] * 70)}")

            if i == 1:
                logging.info('INIT - Initialising feasible investment plan (set all candidate capacity = 0)')
                self.master.initialise_investment_plan(model_inv, self.master_solution_dir)
            else:
                logging.info('INV - Solving investment plan problem')
                self.master.solve_model(model_inv)
                logging.info(f'Candidate capacity solution (total): {model_inv.a.get_values()}')
                logging.info(f'Candidate capacity solution (year investment): {model_inv.x_c.get_values()}')
                self.master.save_solution(model_inv, i, self.master_solution_dir)

            # Store model object containing solution for given iteration
            self.master_models[i] = copy.deepcopy(model_inv)

            # Solve subproblems
            for y in model_uc.Y:
                self.print_log(f"\nSolving UC subproblems for year: {y}\n{''.join(['-'] * 70)}")

                # Update parameters for a given year
                year_parameters = self.subproblem.get_year_parameters(model_uc, y, i, self.master_solution_dir)
                model_uc = self.subproblem.update_parameters(model_uc, year_parameters)
                fixed_cap = {i.index(): i.value for i in model_uc.CAPACITY_FIXED.values()}
                logging.info(f'Fixed candidate capacity for subproblems: {fixed_cap}')

                for s in model_uc.S:
                    self.print_log(f'Solving UC subproblem: year {y}, scenario {s}')

                    # Update parameters for a given scenario
                    scenario_parameters = self.subproblem.get_scenario_parameters(model_uc, y, s)
                    model_uc = self.subproblem.update_parameters(model_uc, scenario_parameters)

                    # Solve subproblem (MILP)
                    model_uc = self.subproblem.solve_model(model_uc)

                    # Fix binary variables
                    model_uc = self.subproblem.fix_binary_variables(model_uc)

                    # Re-solve to obtain dual variables
                    model_uc = self.subproblem.solve_model(model_uc)

                    # Save model output
                    self.subproblem.save_solution(model_uc, i, y, s, self.subproblem_solution_dir)

                    # Unfix binary variables
                    model_uc = self.subproblem.unfix_binary_variables(model_uc)

                    # Store model object containing solution for given year and scenario
                    self.subproblem_models[(i, y, s)] = copy.deepcopy(model_uc)

            # Check for convergence
            converged = self.convergence_check(model_inv, i)

            if converged:
                break

            # Add Benders cut
            self.print_log('Adding Benders cut')
            model_inv = self.master.add_benders_cut(model_inv, i, self.master_solution_dir,
                                                    self.subproblem_solution_dir)

    @staticmethod
    def cleanup_results(directory):
        """Remove all pickle files from a given directory"""

        # All pickle files in a directory
        files = [f for f in os.listdir(directory) if '.pickle' in f]

        # Remove files
        for f in files:
            os.remove(os.path.join(directory, f))


if __name__ == '__main__':
    # Object used to run Benders decomposition algorithm
    benders = BendersAlgorithmController()

    model_inv = benders.master.construct_model()
    # benders.calculations.get_end_of_horizon_operating_cost(model_inv, 1, benders.subproblem_solution_dir)
    # self = benders
    # model_inv = self.master.add_benders_cut(model_inv, 1, self.master_solution_dir, self.subproblem_solution_dir)

    # Run algorithm
    benders.run_benders()
