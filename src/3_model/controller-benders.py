"""Run algorithm to find least-cost investment plan"""

import os
import sys
import copy
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
        logging.basicConfig(filename='controller-benders.log', filemode='a',
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

    @staticmethod
    def cleanup_results(directory):
        """Remove all pickle files from a given directory"""

        # All pickle files in a directory
        files = [f for f in os.listdir(directory) if '.pickle' in f]

        # Remove files
        for f in files:
            os.remove(os.path.join(directory, f))

    def print_log(self, message):
        """Print message to console and create entry in log file"""

        print(message), self.logger.info(message)

    def convergence_check(self, m, iteration, relative_bound_tolerance=0.05):
        """Check if model has converged"""

        upper = self.calculations.get_upper_bound(m, iteration, self.master_solution_dir, self.subproblem_solution_dir)
        self.print_log(f'Solution upper bound: {upper}')

        lower = self.calculations.get_lower_bound(iteration, self.master_solution_dir)
        self.print_log(f'Solution lower bound: {lower}')

        # Difference between upper and lower bound
        bound_difference = upper - lower
        self.print_log(f'Absolute difference between upper and lower bounds: {bound_difference}')

        bound_relative_difference = abs(bound_difference / lower)
        self.print_log(f'Relative difference ((upper - lower)/ lower): {bound_relative_difference}')

        if bound_relative_difference < relative_bound_tolerance:
            self.print_log(f'Model converged')
            return True

        else:
            return False

    def run_benders(self):
        """Run Benders decomposition algorithm"""

        self.logger.info("Running Benders decomposition algorithm")

        # Remove pickle files in results directories
        # self.cleanup_results(self.master_solution_dir)
        # self.cleanup_results(self.subproblem_solution_dir)

        # Construct model objects
        self.logger.info("UC - constructing model")
        m_uc = self.subproblem.construct_model()

        self.logger.info('INV - constructing model')
        m_in = self.master.construct_model()

        # Index for first iteration
        start_iteration = 10

        for i in range(start_iteration, 100):
            self.print_log(f"Performing iteration {i}\n{''.join(['-'] * 70)}")

            # Hot-start - create cuts from previous iterations and add to master problem
            if start_iteration != 1:

                for c in range(1, start_iteration):
                    self.print_log(f'Adding Benders cut for iteration {c}')
                    m_in = self.master.add_benders_cut(m_in, c, self.master_solution_dir, self.subproblem_solution_dir)

            if i == 1:
                logging.info('INIT - Initialising feasible investment plan (set all candidate capacity = 0)')
                self.master.initialise_investment_plan(m_in, i, self.master_solution_dir)

            else:
                self.print_log('INV - Solving investment plan problem')
                self.master.solve_model(m_in)
                self.print_log(f'INV - Candidate capacity solution: {m_in.a.get_values()}')
                self.master.save_solution(m_in, i, self.master_solution_dir)

            # Store model object containing solution for given iteration
            # self.master_models[i] = copy.deepcopy(m_in)

            # Solve subproblems
            for y in m_uc.Y:
            # for y in [2020]:
                self.print_log(f"\nSolving UC subproblems for year: {y}\n{''.join(['-'] * 70)}")

                # Update parameters for a given year
                year_parameters = self.subproblem.get_year_parameters(m_uc, y, i, self.master_solution_dir)
                m_uc = self.subproblem.update_parameters(m_uc, year_parameters)

                fixed_cap = {i.index(): i.value for i in m_uc.CAPACITY_FIXED.values()}
                self.logger.info(f'Fixed candidate capacity for subproblems: {fixed_cap}')

                for s in m_uc.S:
                # for s in [7]:
                    self.print_log(f'Solving UC subproblem: year {y}, scenario {s}')

                    # Update parameters for a given scenario
                    scenario_parameters = self.subproblem.get_scenario_parameters(m_uc, y, s)
                    m_uc = self.subproblem.update_parameters(m_uc, scenario_parameters)

                    # Solve subproblem (MILP)
                    m_uc, solve_status = self.subproblem.solve_model(m_uc)
                    self.logger.info(f'Solved MILP: {solve_status}')

                    # Fix binary variables
                    m_uc = self.subproblem.fix_binary_variables(m_uc)

                    # Re-solve to obtain dual variables
                    # self.subproblem.solver_options = {'simplex.tolerances.feasibility': 0.1}
                    m_uc, solve_status = self.subproblem.solve_model(m_uc)
                    self.logger.info(f'Solved LP: {solve_status}')

                    # Save model output
                    self.subproblem.save_solution(m_uc, solve_status, i, y, s, self.subproblem_solution_dir)

                    # Unfix binary variables
                    m_uc = self.subproblem.unfix_binary_variables(m_uc)

                    # Store model object containing solution for given year and scenario
                    # self.subproblem_models[(i, y, s)] = copy.deepcopy(m_uc)

            # Check convergence
            converged = self.convergence_check(m_in, i, relative_bound_tolerance=0.01)

            if converged:
                break

            # Add Benders cut
            self.print_log('Adding Benders cut')
            m_in = self.master.add_benders_cut(m_in, i, self.master_solution_dir, self.subproblem_solution_dir)

            # Log all benders cuts now associated with model
            for index, constraint in enumerate(m_in.BENDERS_CUTS.values()):
                self.logger.info(f'Benders cut {index}: {str(constraint.expr)}')


if __name__ == '__main__':
    # Object used to run Benders decomposition algorithm
    benders = BendersAlgorithmController()

    # model_inv = benders.master.construct_model()
    # benders.master.solve_model(model_inv)

    # Run algorithm
    benders.run_benders()
