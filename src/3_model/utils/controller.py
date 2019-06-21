"""Run model Benders decomposition to solve investment planning problem"""

from master import MasterProblem
from subproblem import Subproblem


def check_bounds(master_results, subproblem_results):
    """Compare upper and lower bounds - return in absolute and % terms (% difference relative to lower bound)"""
    gap, gap_pct = None, None
    return gap, gap_pct


def run_model():
    """Run Benders decomposition to solve model"""

    # Instantiate master problem object
    master = MasterProblem()
    master_model = master.construct_model()

    # Instantiate subproblem object
    subproblem = Subproblem()
    subproblem_model = subproblem.construct_model()

    # Initialise gap to arbitrarily large number
    gap_pct = 1000

    while gap_pct > 0.1:

        # Solve master problem
        master_results = master.solve_model(master_model)

        # Update subproblem fixed variables
        subproblem_model = subproblem.update_parameters(subproblem_model, master_results)

        # Solve subproblem
        subproblem_results = subproblem.solve_model(subproblem_model)

        # Check objective bounds
        gap, gap_pct = check_bounds(master_results, subproblem_results)

        # Check if tolerance satisfied
        if gap_pct < 0.1:
            print(f'Tolerance satisfied')
            break

        # Add benders cuts
        master_model = master.add_benders_cuts(subproblem_results)
