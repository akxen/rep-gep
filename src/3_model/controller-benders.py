"""Run algorithm to find least-cost investment plan"""

import os
import sys
import pickle
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.uc_subproblem import UnitCommitment
from utils.benders_master import InvestmentPlan


def convergence_check(m, iteration, uc_solutions_dir):
    """Check convergence in investment subproblem - same output will yield same solutions in UC some problems"""

    print(f'Checking convergence')
    logging.info(f'Checking convergence')

    # All UC subproblem results
    files = [f for f in os.listdir(uc_solutions_dir) if ('.pickle' in f)]

    # Current iteration files
    current_iteration = [f for f in files if f'uc-results_{iteration}' in f]
    
    # Value for total objective
    total_objective = 0
    
    for f in current_iteration:
        # Get iteration, year, and scenario from filename
        year, scenario = int(f.split('_')[-2]), int(f.split('_')[-1].replace('.pickle', ''))

        with open(os.path.join(uc_solutions_dir, f), 'rb') as g:
            # Load UC solution
            uc_solution = pickle.load(g)

            # Days for representative scenario
            rho = investment._get_scenario_duration_days(year, scenario)
            
            # Objective function value
            total_objective += rho * uc_solution['OBJECTIVE']

    # Difference between master problem parametrisation solution and total subproblem objective value
    objective_difference = total_objective - m.alpha.value

    logging.info(f'Subproblem total objective value: {total_objective}')
    logging.info(f'Master problem lower subproblem parametrisation value: {m.alpha.value}')
    logging.info(f'Difference between upper and lower bound: {objective_difference}')

    print(f'Subproblem total objective value: {total_objective}')
    print(f'Master problem lower subproblem parametrisation value: {m.alpha.value}')
    print(f'Difference between upper and lower bound: {objective_difference}')


if __name__ == '__main__':

    # Setup logger
    logging.basicConfig(filename='controller-benders.log', filemode='w',
                        format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)

    logging.info("Running subproblem")
    # logger = logging.getLogger('Controller')

    # Directory for iterative solutions
    investment_solutions_directory = os.path.join(os.path.dirname(__file__), 'output', 'investment_plan')
    uc_solutions_directory = os.path.join(os.path.dirname(__file__), 'output', 'operational_plan')

    # Object used to construct and run unit commitment model
    uc = UnitCommitment()

    # Object used to construct and run investment plan subproblem
    investment = InvestmentPlan()

    # Construct UC and investment planning model
    logging.info('UC - constructing model')
    model_uc = uc.construct_model()

    logging.info('INV - constructing model')
    model_inv = investment.construct_model()
    logging.info('INV - capacity dual variables (fixed): {0}'.format({i: j.value for i, j in model_inv.PSI_FIXED._data.items()}))
    logging.info('INV - fixed subproblem capacity: {0}\n'.format({i: j.value for i, j in model_inv.CANDIDATE_CAPACITY_FIXED._data.items()}))

    for i in range(1, 10):
        print(f"Performing iteration {i}\n{''.join(['-']*70)}")
        logging.info(f"Performing iteration {i}\n{''.join(['-']*70)}")

        # Solve master problem
        logging.info(f"Solving investment (master) problem")
        model_inv = investment.solve_model(model_inv)
        print(f"Master problem solution (objective lower bound): {model_inv.OBJECTIVE.expr()}")
        logging.info(f"Master problem solution (objective lower bound): {model_inv.OBJECTIVE.expr()}")
        investment.save_solution(model_inv, i, investment_solutions_directory)
        logging.info(f"Investment capacity solution: {model_inv.a.get_values()}")

        # Solve subproblems
        for y in model_uc.Y:
            print(f"\nSolving UC subproblems for year: {y}\n{''.join(['-'] * 70)}")
            logging.info(f"\nSolving UC subproblems for year: {y}\n{''.join(['-']*70)}")

            # Update parameters for a given year
            year_parameters = uc.get_year_parameters(model_uc, y, investment_solutions_directory)
            model_uc = uc.update_parameters(model_uc, year_parameters)

            for s in model_uc.S:
                print(f'Solving UC subproblem: year {y}, scenario {s}')
                logging.info(f'Solving UC subproblem: year {y}, scenario {s}')
                # Update parameters for a given scenario
                scenario_parameters = uc.get_scenario_parameters(model_uc, y, s)
                model_uc = uc.update_parameters(model_uc, scenario_parameters)

                # Solve subproblem (MILP)
                model_uc = uc.solve_model(model_uc)

                # Fix binary variables
                model_uc = uc.fix_binary_variables(model_uc)

                # Re-solve to obtain dual variables
                model_uc = uc.solve_model(model_uc)

                # Save model output
                uc.save_solution(model_uc, i, y, s, uc_solutions_directory)

                # Unfix binary variables
                model_uc = uc.unfix_binary_variables(model_uc)

        # Check for convergence
        convergence_check(model_inv, i, uc_solutions_directory)

        # Add Benders cut
        print('Adding Benders cut')
        logging.info('Adding Benders cut')
        model_inv = investment.add_benders_cut(model_inv, i, uc_solutions_directory)
