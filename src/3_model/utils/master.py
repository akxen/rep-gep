"""Class used to construct investment planning master problem"""

from pyomo.environ import *

from data import ModelData
from components import CommonComponents


class MasterProblem:
    # Pre-processed model data
    data = ModelData()

    def __init__(self):
        # Solver options
        pass

    def construct_model(self):
        """Construct master problem model components"""

        # Used to define sets and parameters common to both master and master problem
        common_components = CommonComponents()

        # Initialise base model object
        m = ConcreteModel()

        # Define sets - common to both master and master problem
        m = common_components.define_sets(m)

        # Define parameters - common to both master and master problem
        m = common_components.define_parameters(m)

        # Define variables - common to both master and master problem
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
        """Define master problem expressions"""
        return m

    @staticmethod
    def define_constraints(m):
        """Define master problem constraints"""
        return m

    @staticmethod
    def define_objective(m):
        """Define master problem objective"""
        return m

    @staticmethod
    def add_cuts(m, subproblem_results):
        """
        Add Benders cuts to model using data obtained from subproblem solution

        Parameters
        ----------
        m : pyomo model object
            Master problem model object

        subproblem_results : dict
            Results obtained from subproblem solution

        Returns
        -------
        m : pyomo model object
            Master problem model object with additional constraints (Benders cuts) added
        """

        return m

    def solve_model(self, m):
        """Solve model"""

        # Container for model results
        results = {}

        # Solve model
        self.opt.solve(m, keepfiles=False)

        return m, results
