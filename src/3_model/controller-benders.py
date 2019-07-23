"""Run algorithm to find least-cost investment plan"""

import os
import sys
import pickle
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.uc_subproblem import UnitCommitment
from utils.benders_master import InvestmentPlan


class BendersAlgorithmController:
    def __init__(self,
                 master_solution_dir=os.path.join(os.path.dirname(__file__), 'output', 'investment_plan'),
                 subproblem_solution_dir=os.path.join(os.path.dirname(__file__), 'output', 'operational_plan')):

        # Setup logger
        logging.basicConfig(filename='controller-benders.log', filemode='w',
                            format='%(asctime)s %(name)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.DEBUG)

        # Object used to control master problem
        self.master = InvestmentPlan()

        # Object used to control unit commitment (UC) subproblems
        self.subproblem = UnitCommitment()

        # Directory containing master problem solutions
        self.master_solution_dir = master_solution_dir

        # Directory containing subproblem (UC) output files
        self.subproblem_solution_dir = subproblem_solution_dir

    def get_solution_upper_bound(self, m, iteration):
        """Get solution upper-bound"""

        # Upper bound for total cost
        upper_bound = self.master.get_cost_upper_bound(m, iteration,
                                                       self.master_solution_dir,
                                                       self.subproblem_solution_dir)

        return upper_bound

    def get_solution_lower_bound(self, iteration):
        """Get solution lower-bound"""

        # Load investment plan solution
        with open(os.path.join(self.master_solution_dir, f'investment-results_{iteration}.pickle'), 'rb') as f:
            investment_solution = pickle.load(f)

        # Total cost including Benders auxiliary variable
        lower_bound = investment_solution['OBJECTIVE']
        print(f'LOWER BOUND: {lower_bound}')

        return lower_bound

    def convergence_check(self, m, iteration, relative_bound_tolerance=0.05):
        """Check if model has converged"""

        upper = self.get_solution_upper_bound(m, iteration)
        print(f'Solution upper bound: {upper}')
        logging.info(f'Solution upper bound: {upper}')

        lower = self.get_solution_lower_bound(iteration)
        print(f'Solution lower bound: {lower}')
        logging.info(f'Solution lower bound: {lower}')

        # Difference between upper and lower bound
        bound_difference = upper - lower
        print(f'Absolute difference between upper and lower bounds: {bound_difference}')
        logging.info(f'Absolute difference between upper and lower bounds: {bound_difference}')

        bound_relative_difference = abs((upper - lower) / lower)
        print(f'Relative difference ((upper - lower)/ lower): {bound_relative_difference}')
        logging.info(f'Relative difference between ((upper - lower)/ lower): {bound_relative_difference}')

        if bound_relative_difference < relative_bound_tolerance:
            print('Model converged')
            logging.info(f'Model converged')
            return True
        else:
            return False

    def run_benders(self):
        """Run Benders decomposition algorithm"""

        # Remove pickle files in results directories
        self.cleanup_results(self.subproblem_solution_dir)
        self.cleanup_results(self.master_solution_dir)

        logging.info("Running Benders decomposition algorithm")
        logging.info('UC - constructing model')
        model_uc = self.subproblem.construct_model()

        logging.info('INV - constructing model')
        model_inv = self.master.construct_model()

        for i in range(1, 100):
            print(f"Performing iteration {i}\n{''.join(['-'] * 70)}")
            logging.info(f"Performing iteration {i}\n{''.join(['-'] * 70)}")

            if i == 1:
                logging.info('INIT - Initialising feasible investment plan (set all candidate capacity = 0)')
                self.master.initialise_investment_plan(model_inv, self.master_solution_dir)
            else:
                logging.info('INV - Solving investment plan problem')
                self.master.solve_model(model_inv)
                print(model_inv.alpha.value)
                self.master.save_solution(model_inv, i, self.master_solution_dir)

            # Solve subproblems
            for y in model_uc.Y:
                print(f"Solving UC subproblems for year: {y}\n{''.join(['-'] * 70)}")
                logging.info(f"\nSolving UC subproblems for year: {y}\n{''.join(['-'] * 70)}")

                # Update parameters for a given year
                year_parameters = self.subproblem.get_year_parameters(model_uc, y, self.master_solution_dir)
                model_uc = self.subproblem.update_parameters(model_uc, year_parameters)
                fixed_cap = {i.index(): i.value for i in model_uc.CAPACITY_FIXED.values()}
                logging.info(f'Fixed candidate capacity for subproblems: {fixed_cap}')

                for s in model_uc.S:
                    print(f'Solving UC subproblem: year {y}, scenario {s}')
                    logging.info(f'Solving UC subproblem: year {y}, scenario {s}')
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

            # Check for convergence
            converged = self.convergence_check(model_inv, i)

            if converged:
                break

            # Add Benders cut
            print('Adding Benders cut')
            logging.info('Adding Benders cut')
            model_inv = self.master.add_benders_cut(model_inv, i, self.master_solution_dir, self.subproblem_solution_dir)

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

    # Run algorithm
    benders.run_benders()
