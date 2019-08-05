"""Classes used to define mathematical program with primal and dual constraints for generation expansion planning"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'base'))

from pyomo.environ import *

from base.components import CommonComponents


class Primal:
    def __init__(self):
        self.components = CommonComponents()

    def define_variables(self, m):
        """Primal problem variables"""

        # Investment capacity in each year
        m.x_c = Var(m.G_C, m.Y, initialize=0)

        # Cumulative installed capacity
        m.a = Var(m.G_C, m.Y, initialize=0)

        # Power output
        m.p = Var(m.G.difference(m.G_STORAGE), m.Y, m.S, m.T, initialize=0)

        return m

    def define_expressions(self, m):
        """Primal problem expressions"""
        return m

    def define_constraints(self, m):
        """Primal problem constraints"""

        def min_power_rule(m, g, y, s, t):
            """Minimum power output"""

            return - m.p[g, y, s, t] + m.P_MIN[g] <= 0

        # Minimum power output
        m.MIN_POWER = Constraint(m.G.difference(m.G_STORAGE), m.Y, m.S, m.T, rule=min_power_rule)

        return m

    def define_objective(self, m):
        """Primal program objective"""
        return m

    def construct_model(self):
        """Construct primal program"""

        # Initialise model
        m = ConcreteModel()

        # Add common components
        m = self.components.define_sets(m)
        m = self.components.define_parameters(m)

        # Primal problem variables
        m = self.define_variables(m)

        # Primal problem expressions
        m = self.define_expressions(m)

        # Primal problem constraints
        m = self.define_constraints(m)

        # Primal problem objective
        m = self.define_objective(m)

        return m


if __name__ == '__main__':
    primal = Primal()

    mo = primal.construct_model()
