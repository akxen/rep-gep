"""Run algorithm to find least-cost investment plan"""

import os
import sys
import pickle
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.uc_subproblem import UnitCommitment
from utils.investment_subproblem import InvestmentPlan


def convergence_check(model, investment_solutions_dir):
    """Check convergence in investment subproblem - same output will yield same solutions in UC some problems"""

    with open(os.path.join(investment_solutions_dir, 'investment-results.pickle'), 'rb') as f:
        previous_solution = pickle.load(f)

    print(previous_solution['CAPACITY_FIXED'])
    print(model.a.get_values())

    # Difference between installed capacity solution in successive iterations
    capacity_difference = {k: abs(model.a[k].value - previous_solution['CAPACITY_FIXED'][k]) for k in model.a.keys()}

    # Max capacity difference
    max_capacity_difference = max(capacity_difference.values())

    # Difference in emissions constraint dual variable
    lambda_difference = abs(model.dual[model.EMISSIONS_CONSTRAINT] - previous_solution['LAMBDA_FIXED'])

    # Message to print when running convergence check
    message = f"""
    Convergence check
    -----------------
    Max capacity difference: {max_capacity_difference} 
    Lambda difference: {lambda_difference}
    """
    print(message)

    if (max_capacity_difference < 1) and (lambda_difference < 1):
        return True
    else:
        return False


if __name__ == '__main__':

    # Directory for iterative solutions
    investment_solutions_directory = os.path.join(os.path.dirname(__file__), 'output', 'investment_plan')
    uc_solutions_directory = os.path.join(os.path.dirname(__file__), 'output', 'operational_plan')

    # Object used to construct and run unit commitment model
    uc = UnitCommitment()

    # Object used to construct and run investment plan subproblem
    investment = InvestmentPlan()

    # Construct UC and investment planning model
    model_uc = uc.construct_model()
    model_inv = investment.construct_model()

    # Solve investment plan
    model_inv = investment.solve_model(model_inv)

    for i in range(1, 10):
        print(f'Solving investment problem: iteration {i}')
        if i != 1:
            # Update parameters obtained from UC subproblem solutions
            model_inv = investment.update_parameters(model_inv, uc_solutions_directory)

        # Solve model
        model_inv = investment.solve_model(model_inv)

        # Fix all binary variables
        model_inv = investment.fix_binary_variables(model_inv)

        # Re-solve to obtain dual variables
        model_inv = investment.solve_model(model_inv)

        if i != 1:
            # Check convergence
            converged = convergence_check(model_inv, investment_solutions_directory)

            if converged:
                print('Model has converged. Terminating solution algorithm.')
                break

        # Save solution
        investment.save_solution(model_inv, investment_solutions_directory)

        # Unfix all binary variables
        model_inv = investment.unfix_binary_variables(model_inv)

        # Get iteration parameters - UC problem
        iteration_parameters = uc.get_iteration_parameters(investment_solutions_directory)

        # Update iteration parameters common to all UC subproblems
        model_uc = uc.update_parameters(model_uc, iteration_parameters)

        # Solve UC subproblem for each year and operating scenario
        for y in model_uc.Y:

            # Get parameters that must be updated for each year
            year_parameters = uc.get_year_parameters(model_uc, y, investment_solutions_directory)

            # Update parameters for a given year
            model_uc = uc.update_parameters(model_uc, year_parameters)

            for s in model_uc.S:

                # Get parameters that must be updated for a given scenario
                scenario_parameters = uc.get_scenario_parameters(model_uc, y, s)

                # Update parameters for a given scenario
                model_uc = uc.update_parameters(model_uc, scenario_parameters)

                print(f'Solving UC subproblem: year {y}, scenario {s}')
                # Solve model
                model_uc = uc.solve_model(model_uc)

                # Fix all binary variables
                model_uc = uc.fix_binary_variables(model_uc)

                # Re-solve to obtain dual variables
                model_uc = uc.solve_model(model_uc)

                # Save model output
                uc.save_solution(model_uc, y, s, uc_solutions_directory)

                # Unfix binary variables
                model_uc = uc.unfix_binary_variables(model_uc)
