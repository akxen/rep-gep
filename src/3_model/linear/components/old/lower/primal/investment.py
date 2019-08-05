"""Block of investment constraints"""

import os
import sys

from pyomo.environ import *

path = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'common')
sys.path.append(path)

from data import ModelData
from components import CommonComponents


class PrimalInvestmentBlocks:
    def __init__(self):
        self.data = ModelData()
        self.components = CommonComponents()

    def define_year_block_parameters(self, m, b, y):
        """Parameters related to investment blocks"""

        def candidate_unit_build_costs_rule(b, g):
            """
            Candidate unit build costs [$/MW]

            Note: build cost in $/MW. May need to scale if numerical conditioning problems.
            """

            if g in m.G_C_STORAGE:
                return float(self.data.battery_build_costs_dict[y][g] * 1000)

            else:
                return float(self.data.candidate_units_dict[('BUILD_COST', y)][g] * 1000)

        # Candidate unit build cost
        b.I_C = Param(m.G_C, rule=candidate_unit_build_costs_rule)

        def amortisation_rate(b, g):
            """Amortisation rate for a given generator"""

            # Numerator
            num = m.INTEREST_RATE * ((1 + m.INTEREST_RATE) ** m.ASSET_LIFE[g])

            # Denominator
            den = ((1 + m.INTEREST_RATE) ** m.ASSET_LIFE[g]) - 1

            # Amortisation rate
            rate = float(num / den)

            return rate

        # Amortisation rate for a given investment
        b.GAMMA = Param(m.G_C, rule=amortisation_rate)

        def retirement_indicator_rule(_m, g):
            """Indicates if unit is retired in a given year. 1 - retired, 0 - still in service"""
            # TODO: Use announced retirement data from NTNDP. Assume no closures for now.
            return float(0)

        # Retirement indicator: 1 - unit is retired, 0 - unit is still in service
        b.F = Param(m.G_E, rule=retirement_indicator_rule)

        # Discount rate
        b.DELTA = Param(initialize=float(1 / ((1 + m.INTEREST_RATE) ** y)))

        return b

    @staticmethod
    def define_year_block_primal_variables(m, b):
        """Variables related to investment blocks"""

        # Investment decisions relating to a particular year
        b.x_c = Var(m.G_C)

        # Cumulative installed capacity for each candidate generator
        b.a = Var(m.G_C)

        return b

    @staticmethod
    def define_year_block_expressions(m, b, y):
        """Define expressions related to yearly costs"""

        def investment_costs_rule(b):
            """Total amortised and discounted investment cost for given year of model horizon"""

            return sum(b.GAMMA[g] * b.I_C[g] * b.x_c[g] for g in m.G_C) / m.INTEREST_RATE

        # Total investment cost
        b.INV = Expression(rule=investment_costs_rule)

        def fixed_operations_and_maintenance_cost_rule(b):
            """Fixed operations and maintenance costs"""

            # Candidate FOM costs
            candidate_fom = sum(m.C_FOM[g] * b.a[g] for g in m.G_C)

            # Existing FOM costs (taking into account announced retirements)
            existing_fom = sum(m.C_FOM[g] * m.P_MAX[g] * (1 - b.F[g]) for g in m.G_E)

            return candidate_fom + existing_fom

        # Total FOM cost
        b.FOM = Expression(rule=fixed_operations_and_maintenance_cost_rule)

        return b

    @staticmethod
    def define_year_block_primal_constraints(m, b, y):
        """Constraints related to investment block"""

        def cumulative_capacity_rule(b, g):
            """Constraint used to construct cumulative candidate capacity variable"""

            return b.a[g] - sum(m.y[year].x_c[g] for year in m.Y if year <= y) == 0

        # Cumulative installed candidate capacity
        b.CUMULATIVE_CAPACITY = Constraint(m.G_C, rule=cumulative_capacity_rule)

        def solar_build_limits_rule(b, z):
            """Enforce solar build limits in each NEM zone"""

            # Solar generators belonging to zone 'z'
            gens = [g for g in m.G_C_SOLAR if g.split('-')[0] == z]

            if gens:
                return sum(b.a[g] for g in gens) - m.SOLAR_BUILD_LIMITS[z] <= 0
            else:
                return Constraint.Skip

        # Storage build limit constraint for each NEM zone
        b.SOLAR_BUILD_LIMIT_CONS = Constraint(m.Z, rule=solar_build_limits_rule)

        def wind_build_limits_rule(_m, z):
            """Enforce wind build limits in each NEM zone"""

            # Wind generators belonging to zone 'z'
            gens = [g for g in m.G_C_WIND if g.split('-')[0] == z]

            if gens:
                return sum(b.a[g] for g in gens) - m.WIND_BUILD_LIMITS[z] <= 0
            else:
                return Constraint.Skip

        # Wind build limit constraint for each NEM zone
        b.WIND_BUILD_LIMIT_CONS = Constraint(m.Z, rule=wind_build_limits_rule)

        def storage_build_limits_rule(_m, z):
            """Enforce storage build limits in each NEM zone"""

            # Storage generators belonging to zone 'z'
            gens = [g for g in m.G_C_STORAGE if g.split('-')[0] == z]

            if gens:
                return sum(b.a[g] for g in gens) - m.STORAGE_BUILD_LIMITS[z] <= 0
            else:
                return Constraint.Skip

        # Storage build limit constraint for each NEM zone
        b.STORAGE_BUILD_LIMIT_CONS = Constraint(m.Z, rule=storage_build_limits_rule)

        return b

    @staticmethod
    def define_scenario_block_parameters(m, s):
        """Scenario parameters"""
        return s

    @staticmethod
    def define_scenario_block_variables(m, s):
        """Scenario variables"""

        # Power output - excluding storage units
        s.p = Var(m.G_C.difference(m.G_STORAGE), m.T, initialize=0)

        return s

    @staticmethod
    def define_scenario_block_expressions(m, s):
        """Scenario expressions"""
        return s

    @staticmethod
    def define_scenario_block_constraints(m, s):
        """Define scenario block constraints"""

        def min_power_rule(s, g, t):
            """Minimum power output"""

            return - s.p[g, t] + m.P_MIN[g] <= float(0)

        # Minimum power output constraint
        s.MIN_POWER = Constraint(m.G.difference(m.G_STORAGE), m.T, rule=min_power_rule)

        return s

    def construct_scenario_blocks(self, m, b, y):
        """Dispatch blocks"""

        def scenario_block_rule(b, s):
            """Dispatch block rule"""

            # Dispatch block variables
            s = self.define_scenario_block_variables(m, s)

            # Dispatch block constraints
            # s = self.define_scenario_block_constraints(m, s)

            return s

        b.s = Block(m.S, rule=scenario_block_rule)

        return b

    def construct_blocks(self, m):
        """Blocks of constraints related to yearly investment decisions"""

        def investment_block_rule(b, y):
            """Rule used to construct block"""

            # Investment block parameters
            b = self.define_year_block_parameters(m, b, y)

            # Investment block variables
            b = self.define_year_block_variables(m, b)

            # Investment block expressions
            b = self.define_year_block_expressions(m, b, y)

            # Investment block constraints
            b = self.define_year_block_constraints(m, b, y)

            # Dispatch block variables
            b = self.construct_scenario_blocks(m, b, y)

            return b

        # Blocks related to investment decisions - yearly resolution
        m.y = Block(m.Y, rule=investment_block_rule)

        return m


class PrimalDispatchBlocks:
    def __init__(self):
        self.components = CommonComponents()

    def define_year_block_parameters(self, m):
        """Parameters related to dispatch decision blocks"""
        return m

    @staticmethod
    def define_year_block_variables(m, b, s):
        """Variables related to dispatch decision blocks"""

        # Power output (non-storage units)
        b.p = Var(m.G.difference(m.G_STORAGE), m.T, initialize=float(0))

        return b

    def define_year_block_expressions(self, m):
        """Expressions related to dispatch decision blocks"""
        return m

    def define_year_block_constraints(self, m):
        """Constraints related to dispatch decision blocks"""
        return m

    def construct_blocks(self, m):
        """Blocks of constraints related to dispatch decision blocks"""

        def dispatch_block_rule(b, y, s):
            """Block defining dispatch operating decisions"""

            # Define variables
            b = self.define_year_block_variables(m, b, s)

            return b

        # Blocks of constraints relating to operating scenario decisions
        m.s = Block(m.Y, m.S, rule=dispatch_block_rule)

        return m


class PrimalModel:
    def __init__(self):
        self.components = CommonComponents()
        self.dispatch = PrimalDispatchBlocks()
        self.investment = PrimalInvestmentBlocks()

    def define_constraints(self, m):
        """Constraints for primal problem"""

        return m

    def define_objective(self, m):
        """Define primal model objective"""
        return m

    def construct_model(self):
        """Construct primal program"""

        # Initialise model object
        m = ConcreteModel()

        # Add common model components (sets and common parameters)
        m = self.components.define_sets(m)
        m = self.components.define_parameters(m)

        # Construct dispatch blocks
        # m = self.dispatch.construct_blocks(m)

        # Construct investment blocks
        m = self.investment.construct_blocks(m)

        # Define additional constraints specific to primal problem
        m = self.define_constraints(m)

        # Define objective
        m = self.define_objective(m)

        return m


class DualInvestmentBlocks:
    def __init__(self):
        self.components = CommonComponents()
        self.parameters = CommonParameters()

    def define_year_block_parameters(self, m):
        """Parameters related to investment decision blocks"""
        return m

    def define_year_block_variables(self, m):
        """Variables related to investment blocks"""
        return m

    def define_year_block_expressions(self, m):
        """Expressions related to investment decision blocks"""
        return m

    def define_year_block_constraints(self, m):
        """Constraints related to investment block"""
        return m

    def construct_blocks(self, m):
        """Blocks of constraints related to yearly investment decisions"""
        return m


class DualDispatchBlocks:
    def __init__(self):
        self.components = CommonComponents()
        self.parameters = CommonParameters()

    def define_year_block_parameters(self, m):
        """Parameters related to dispatch decision blocks"""
        return m

    def define_year_block_variables(self, m):
        """Variables related to dispatch decision blocks"""
        return m

    def define_year_block_expressions(self, m):
        """Expressions related to dispatch decision blocks"""
        return m

    def define_year_block_constraints(self, m):
        """Constraints related to dispatch decision blocks"""
        return m

    def construct_blocks(self, m):
        """Blocks of constraints related to dispatch decision blocks"""
        return m


class DualModel:
    def __init__(self):
        self.dispatch = DualDispatchBlocks()
        self.investment = DualInvestmentBlocks()

    def define_objective(self, m):
        """Define primal model objective"""
        return m

    def construct_model(self):
        """Construct primal program"""

        # Initialise model object
        m = ConcreteModel()

        # Construct investment blocks
        m = self.dispatch.construct_blocks(m)

        # Construct dispatch blocks
        m = self.investment.construct_blocks(m)

        # Define objective
        m = self.define_objective(m)

        return m


class MPPDCModel:
    def __init__(self):
        self.primal = PrimalModel()
        self.dual = DualModel()

    def define_constraints(self, m):
        """Define MPPDC constraints (strong duality constraint)"""
        return m

    def define_objective(self, m):
        """Define MPPDC objective"""
        return m

    def construct_model(self):
        """Join primal and dual model components to construct MPPDC"""

        # Initialise model
        m = ConcreteModel()

        # Primal problem blocks
        m = self.primal.dispatch.construct_blocks(m)
        m = self.primal.investment.construct_blocks(m)

        # Dual problem blocks
        m = self.dual.dispatch.construct_blocks(m)
        m = self.dual.investment.construct_blocks(m)

        # Strong duality constraint
        m = self.define_constraints(m)

        # Dummy objective (only needs feasibility)
        m = self.define_objective(m)

        return m

    def solve_model(self, m):
        """Solve MPPDC"""
        return m


if __name__ == '__main__':
    primal = PrimalModel()
    model = primal.construct_model()
