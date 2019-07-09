"""Unit commitment model subproblem"""

import time

from pyomo.environ import *
from pyomo.util.infeasible import log_infeasible_constraints

from data import ModelData
from components import CommonComponents


class UnitCommitment:
    # Pre-processed data for model construction
    data = ModelData()

    # Common model components to investment plan and operating sub-problems (sets)
    components = CommonComponents()

    def __init__(self):
        # Solver options
        self.keepfiles = False
        self.solver_options = {}  # 'MIPGap': 0.0005
        self.opt = SolverFactory('gurobi', solver_io='lp')

    def define_parameters(self, m):
        """Define unit commitment problem parameters"""

        def minimum_region_up_reserve_rule(_m, r):
            """Minimum upward reserve rule"""

            return float(self.data.minimum_reserve_levels.loc[r, 'MINIMUM_RESERVE_LEVEL'])

        # Minimum upward reserve
        m.RESERVE_UP = Param(m.R, rule=minimum_region_up_reserve_rule)

        # -------------------------------------------------------------------------------------------------------------
        # Parameters to update each time model is run
        # -------------------------------------------------------------------------------------------------------------
        # Initial on-state rule - must be updated each time model is run
        m.U0 = Param(m.G_THERM, within=Binary, mutable=True, initialize=1)

        return m

    @staticmethod
    def define_variables(m):
        """Define unit commitment problem variables"""

        # Upward reserve allocation [MW]
        m.r_up = Var(m.G_THERM.union(m.G_C_STORAGE), m.T, within=NonNegativeReals, initialize=0)

        # Amount by which upward reserve is violated [MW]
        m.r_up_violation = Var(m.R, m.T, within=NonNegativeReals, initialize=0)

        # Startup state variable
        m.v = Var(m.G_THERM, m.T, within=NonNegativeReals, initialize=0)

        # On-state variable
        m.u = Var(m.G_THERM, m.T, within=NonNegativeReals, initialize=1)

        # Shutdown state variable
        m.w = Var(m.G_THERM, m.T, within=NonNegativeReals, initialize=0)

        return m

    def define_expressions(self, m):
        """Define unit commitment problem expressions"""
        pass

    def define_constraints(self, m):
        """Define unit commitment problem constraints"""

        def reserve_up_rule(_m, r, t):
            """Ensure sufficient up power reserve in each region"""

            # Existing and candidate thermal gens + candidate storage units
            gens = m.G_E_THERM.union(m.G_C_THERM).union(m.G_C_STORAGE)

            # Subset of generators with NEM region
            gens_subset = [g for g in gens if self.data.generator_zone_map[g] in self.data.nem_region_zone_map_dict[r]]

            return sum(m.r_up[g, t] for g in gens_subset) + m.r_up_violation[r, t] >= m.RESERVE_UP[r]

        # Upward power reserve rule for each NEM region
        m.RESERVE_UP_CONS = Constraint(m.R, m.T, rule=reserve_up_rule)

        def generator_state_logic_rule(_m, g, t):
            """
            Determine the operating state of the generator (startup, shutdown
            running, off)
            """

            if t == m.T.first():
                # Must use U0 if first period (otherwise index out of range)
                return m.u[g, t] - m.U0[g] == m.v[g, t] - m.w[g, t]

            else:
                # Otherwise operating state is coupled to previous period
                return m.u[g, t] - m.u[g, t-1] == m.v[g, t] - m.w[g, t]

        # Unit operating state
        m.GENERATOR_STATE_LOGIC = Constraint(m.G_THERM, m.T, rule=generator_state_logic_rule)

        def minimum_on_time_rule(_m, g, t):
            """Minimum number of hours generator must be on"""

            # Hours for existing units
            if g in m.G_E_THERM:
                hours = self.data.existing_units_dict[('PARAMETERS', 'MIN_ON_TIME')][g]

            # Hours for candidate units
            elif g in m.G_C_THERM:
                hours = self.data.candidate_units_dict[('PARAMETERS', 'MIN_ON_TIME')][g]

            else:
                raise Exception(f'Min on time hours not found for generator: {g}')

            # Time index used in summation
            time_index = [k for k in range(t - int(hours) + 1, t + 1) if k >= 0]

            # Constraint only defined over subset of timestamps
            if t >= hours:
                return sum(m.v[g, j] for j in time_index) <= m.u[g, t]
            else:
                return Constraint.Skip

        # Minimum on time constraint
        m.MINIMUM_ON_TIME = Constraint(m.G_E_THERM.union(m.G_C_THERM), m.T, rule=minimum_on_time_rule)

        return m

    def define_objective(self, m):
        """Define unit commitment problem objective function"""
        pass

    def construct_model(self):
        """Construct unit commitment model"""

        # Initialise model object
        m = ConcreteModel()

        # Define sets
        m = self.components.define_sets(m)

        # Define parameters common to unit commitment sub-problems and investment plan
        m = self.components.define_parameters(m)

        # Define parameters specific to unit commitment sub-problem
        m = self.define_parameters(m)

        # Define variables
        m = self.define_variables(m)

        # Define constraints
        m = self.define_constraints(m)

        return m

    def update_parameters(self, year, scenario):
        """Populate model object with parameters for a given operating scenario"""
        pass


if __name__ == '__main__':
    # Initialise object used to construct model
    uc = UnitCommitment()

    # Construct model object
    model = uc.construct_model()
