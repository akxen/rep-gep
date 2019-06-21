"""Classes used to construct investment planning subproblem"""

from pyomo.environ import *

from data import ModelData
from components import CommonComponents


class Subproblem:
    # Pre-processed model data
    data = ModelData()

    def __init__(self):
        # Solver options
        pass

    def construct_model(self):
        """Construct subproblem model components"""

        # Used to define sets and parameters common to both master and subproblem
        common_components = CommonComponents()

        # Initialise base model object
        m = ConcreteModel()

        # Define sets - common to both master and subproblem
        m = common_components.define_sets(m)

        # Define parameters - common to both master and subproblem
        m = common_components.define_parameters(m)

        # Define variables - common to both master and subproblem
        m = common_components.define_variables(m)

        # Define expressions
        m = self.define_expressions(m)

        # Define constraints
        m = self.define_constraints(m)

        # Define objective
        m = self.define_objective(m)

        return m

    @staticmethod
    def define_expressions(m):
        """Define subproblem expressions"""
        return m

    @staticmethod
    def define_constraints(m):
        """Define subproblem constraints"""
        return m

    @staticmethod
    def define_objective(m):
        """Define subproblem objective"""
        return m

    @staticmethod
    def update_parameters(m, parameters):
        """Update parameter values"""
        return m

    @staticmethod
    def _solve_for_output(m):
        """Solve for output"""
        result = None
        return m, result

    @staticmethod
    def _solve_for_prices(m):
        """Solve for marginal prices"""
        results = None
        return m, results

    @staticmethod
    def _unfix_variables(m):
        """Unfix subproblem variables - return model variable fixed state to what it was prior to first solve"""
        return m

    @staticmethod
    def _save_results(results):
        """
        Save (potentially intermediary) results

        Parameters
        ----------
        results : dict
            Sub-problem results stored in a dictionary
        """

        # Save results

        pass

    def solve_model(self, m):
        """Solve model"""

        # Container for model results
        results = {}

        # Solve for output
        m, output_results = self._solve_for_output(m)

        # Solve for prices
        m, output_prices = self._solve_for_output(m)

        # Store results in dictionary
        results['output'] = output_results
        results['prices'] = output_prices

        # Unfix variables - returning model to state prior to first solve
        m = self._unfix_variables(m)

        return m, results
