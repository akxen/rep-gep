import os
import pickle
import logging

from .MasterProblem import Master
from .Subproblem import UnitCommitment


# Class to implement Benders decomposition algorithm. Optimal investment
# decisions and the permit price are taken as complicating variables. Probably
# need to treat emissions constraint as a complicating constraint.

# Load the Master and SubProblem classes, and iteratively solve these
# until convergence occurs.

# Output should be the optimal operation plan, investment decisions, and
# permit price trajectory over the planning horizon.

# Could also place this in a class, and class the solution protocol from within
# main.py

class RunBenders:
    def __init__(self, output_dir):
        """
        Run Benders decomposition algorithm. Input is model objects specifying
        master and sub-problem structure. Output is the optimal investment
        portfolio, and trajectory of the permit price / emissions intensity
        baseline.
        """

        # Directory for output files
        self.output_dir = output_dir

        pass

    def save_results(self):
        """
        Save results for selected variables and expressions
        """
        pass

    def run_benders(self):
        """
        Implement benders decomposition algorithm

        Returns
        -------
        results : dict
            Optimal operation plan (power output for each generator for each
            subproblem), prices for each NEM zone, optimal investment
            decisions (binary variables), and permit price and baseline
            trajectories over the planning horizon.
        """

        # Initialise iteration counter = 0
        iter = 0

        # Initialise tolerance
        tol = 1

        # Set a maximum iteration limit
        iter_lim = 100

        # While difference between upper and lower-bound < stopping tol
        # and the iteration limit has not been exceeded.

            # Increment iteration counter by 1
            # Log this


            # Solve the master problem
            # Log if solved successfully

            # Container for sub-problem solutions

            # For each sub-problem

                # Update sub-problem parameters
                # Log this

                # Solve the sub-problem
                # Log this

                # if sub-problem is infeasible
                    # Log this
                    # Switch to formulation including artificial variables
                    # Log this
                    # Re-solve
                    # Log this

                # Store sub-problem results in a dictionary
                # Log this

                # Append sub-problem results to container
                # Log this

            # If difference between upper and lower bounds for the objective
            # function value are sufficiently small, stop, else continue
            # Log gap

            # Add cuts to the master problem using sub-problem results
            # Log this

        # Save results for final operations + investment plan


if __name__ == '__main__':
    # Directory for output files
    output_dir = os.path.join(os.path.curdir, 'output')

    # Initialise object used to run algorithm
    Benders = RunBenders(output_dir)

    # Run Benders Decomposition algorithm
    Benders.run_benders()
