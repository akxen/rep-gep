from pyomo.environ import *


class MasterProblem:
    def __init__(self):

        # Candidate generators

        #

        pass

    def define_variables(self, m):
        """Master problem variables"""

        # Investment decisions - binary for thermal, continuous for other plant types

        pass

    def define_parameters(self, m):
        """Master problem parameters"""
        # Amortisation rate

        # Investment cost

        # Build limits (solar, wind, storage)

        # Sub-problem upper-bound

        pass

    def define_expressions(self, m):
        """Master problem expressions"""

        # Fixed operation and maintenance cost

        # Revenue constraints

        #

        pass

    def define_constraints(self, m):
        """Master problem constraints"""
        # Thermal unit sizing

        # Thermal unit size selection

        # Solar investment

        # Wind investment

        # Storage investment

        # Investment budget constraint

        # Emissions constraint

        pass

    def objective_function_rule(self, m):
        """Master problem objective function"""

        pass
