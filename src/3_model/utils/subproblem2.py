"""Classes used to construct investment planning subproblem"""

import time
import pickle
import logging
from collections import OrderedDict

from pyomo.environ import *
from pyomo.core.expr import current as EXPR
from pyomo.util.infeasible import log_infeasible_constraints

from data import ModelData
from components import CommonComponents


class Subproblem:
    # Pre-processed model data
    data = ModelData()

    def __init__(self):
        # Solver options
        self.keepfiles = False
        self.solver_options = {'Method': 1}  # 'MIPGap': 0.0005
        self.opt = SolverFactory('gurobi', solver_io='lp')

        # Setup logger
        logging.basicConfig(filename='subproblem.log', filemode='a',
                            format='%(asctime)s %(name)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.DEBUG)

        logging.info("Running subproblem")
        self.logger = logging.getLogger('Subproblem')

    def define_parameters(self, m):
        """Define parameters"""
        m.ZERO = Param(initialize=0)

        return m

    def define_variables(self, m):
        """Define variables used in model"""
        return m

    def define_expressions(self, m):
        """Define expressions to be used in model"""

        def discrete_investment_capacity_lhs_rule(m, g, y):
            """Discrete investment capacity expression"""

            return m.x_c[g, y] - sum(m.d[g, y, n] * m.X_CT[g, n] for n in m.G_C_THERM_SIZE_OPTIONS)

        # LHS for constraint enforcing candidate thermal units
        m.DISCRETE_CAPACITY_LHS = Expression(m.G_C_THERM, m.Y, rule=discrete_investment_capacity_lhs_rule)

        def discrete_investment_binary_variable_lhs_rule(m, g, y):
            """Discrete investment expression enforcing single selection per investment period and technology type"""

            return sum(m.d[g, y, n] for n in m.G_C_THERM_SIZE_OPTIONS) - 1

        # LHS for constraint enforcing single option selection for candidate capacity
        m.DISCRETE_CAPACITY_BINARY_LHS = Expression(m.G_C_THERM, m.Y, rule=discrete_investment_binary_variable_lhs_rule)

        def non_negative_installed_capacity_lhs_rule(m, g, y):
            """Candidate installed capacity is non-negative"""

            return -m.x_c[g, y]

        # LHS for constraint defining non-negative candidate capacity
        m.NON_NEGATIVE_CAPACITY_LHS = Expression(m.G_C_SOLAR.union(m.G_C_WIND).union(m.G_C_STORAGE), m.Y,
                                                 rule=non_negative_installed_capacity_lhs_rule)

        def solar_build_limits_lhs_rule(m, z, y):
            """LHS for solar build limit constraint"""

            return (sum(m.x_c[g, j] for g in m.G_C_SOLAR for j in m.Y if j <= y)
                    - float(self.data.candidate_unit_build_limits_dict[z]['SOLAR']))

        # LHS for constraint enforcing solar zone build limits
        m.SOLAR_BUILD_LIMITS_LHS = Expression(m.Z, m.Y, rule=solar_build_limits_lhs_rule)

        def wind_build_limits_lhs_rule(m, z, y):
            """LHS for wind build limit constraint"""

            return (sum(m.x_c[g, j] for g in m.G_C_WIND for j in m.Y if j <= y)
                    - float(self.data.candidate_unit_build_limits_dict[z]['WIND']))

        # LHS for constraint enforcing solar zone build limits
        m.WIND_BUILD_LIMITS_LHS = Expression(m.Z, m.Y, rule=wind_build_limits_lhs_rule)

        def storage_build_limits_lhs_rule(m, z, y):
            """LHS for storage build limit constraint"""

            return (sum(m.x_c[g, j] for g in m.G_C_WIND for j in m.Y if j <= y)
                    - float(self.data.candidate_unit_build_limits_dict[z]['STORAGE']))

        # LHS for constraint enforcing solar zone build limits
        m.STORAGE_BUILD_LIMITS_LHS = Expression(m.Z, m.Y, rule=storage_build_limits_lhs_rule)

        return m

    def define_investment_constraints(self, m):
        """Define investment constraints"""

        def discrete_investment_capacity_rule(m, g, y):
            """Discrete capacity size"""
            return m.DISCRETE_CAPACITY_LHS[g, y] == float(0)

        # Discrete investment capacity constraint
        m.DISCRETE_CAPACITY = Constraint(m.G_C_THERM, m.Y, rule=discrete_investment_capacity_rule)

        def discrete_capacity_binary_variable_rule(m, g, y):
            """Define discrete capacity decision enforcing single selection"""
            return m.DISCRETE_CAPACITY_BINARY_LHS[g, y] == float(0)

        # Discrete capacity binary variable
        m.DISCRETE_CAPACITY_BINARY = Constraint(m.G_C_THERM, m.Y, rule=discrete_capacity_binary_variable_rule)

        def non_negative_installed_capacity_rule(m, g, y):
            """Enforce non-negative installed capacity"""
            return m.NON_NEGATIVE_CAPACITY_LHS[g, y] <= float(0)

        # Non-negative installed capacities for candidate wind, solar, and storage units
        m.NON_NEGATIVE_CAPACITY = Constraint(m.G_C_SOLAR.union(m.G_C_WIND).union(m.G_C_STORAGE), m.Y,
                                             rule=non_negative_installed_capacity_rule)

        def solar_build_limits_rule(m, z, y):
            """Build limits for candidate solar generators in each zone"""
            return m.SOLAR_BUILD_LIMITS_LHS[z, y] <= float(0)

        # Solar build limits for each zone
        m.SOLAR_BUILD_LIMITS = Constraint(m.Z, m.Y, rule=solar_build_limits_rule)

        def wind_build_limits_rule(m, z, y):
            """Build limits for candidate wind generators in each zone"""
            return m.SOLAR_BUILD_LIMITS_LHS[z, y] <= float(0)

        # Solar build limits for each zone
        m.WIND_BUILD_LIMITS = Constraint(m.Z, m.Y, rule=wind_build_limits_rule)

        def storage_build_limits_rule(m, z, y):
            """Build limits for candidate wind generators in each zone"""
            return m.STORAGE_BUILD_LIMITS_LHS[z, y] <= float(0)

        # Solar build limits for each zone
        m.STORAGE_BUILD_LIMITS = Constraint(m.Z, m.Y, rule=storage_build_limits_rule)

        return m

    def construct_model(self):
        """Construct subproblem model components"""

        # Used to define sets and parameters common to both master and subproblem
        common_components = CommonComponents()

        # Initialise base model object
        m = ConcreteModel()

        # Define sets - common to both master and subproblem
        m = common_components.define_sets(m)

        # Define parameters
        m = common_components.define_parameters(m)

        # Define variables
        m = common_components.define_variables(m)

        # Define parameters
        m = self.define_parameters(m)

        # Define expressions
        m = self.define_expressions(m)

        # Define investment constraints
        m = self.define_investment_constraints(m)

        return m


if __name__ == '__main__':
    # Start timer
    start_timer = time.time()

    # Define object used to construct subproblem model
    subproblem = Subproblem()

    # Construct subproblem model
    model = subproblem.construct_model()

    # Prepare to read suffix values (dual values)
    model.dual = Suffix(direction=Suffix.IMPORT)
    print(f'Constructed model in: {time.time() - start_timer}s')
