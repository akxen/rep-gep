import os

from pyomo.environ import *

from components import BaseComponents


class MasterProblem(BaseComponents):
    def __init__(self, raw_data_dir, data_dir, input_traces_dir):
        # Inherit model data and common model components
        super().__init__(raw_data_dir, data_dir, input_traces_dir)

    @staticmethod
    def define_variables(m):
        """Master problem variables"""

        # Investment decisions - binary for thermal, continuous for other plant types
        m.x_c = Var(m.G_C_WIND.union(m.G_C_SOLAR).union(m.G_C_STORAGE), m.I, within=NonNegativeReals)

        # Investment decisions for thermal plant
        m.d = Var(m.G_C_THERM, m.I, m.G_C_THERM_SIZE_OPTIONS, within=Binary)

        return m

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

    def construct_model(self):
        """Construct each component of the master problem"""

        m = ConcreteModel()

        # Define model sets
        m = self.define_sets(m)

        return m


if __name__ == '__main__':
    # Directory containing files from which dataset is derived
    raw_data_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'data')

    # Directory containing core data files
    data_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '1_collect_data', 'output')

    # Directory containing input traces
    input_traces_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '2_input_traces', 'output')

    # Master problem
    master = MasterProblem(raw_data_directory, data_directory, input_traces_directory)


def operating_state_logic_rule(m, g, t):
    """
    Determine the operating state of the generator (startup, shutdown
    running, off)
    """

    if t == m.T.first():
        # Must use U0 if first period (otherwise index out of range)
        return m.u[g, t] - m.U0[g] == m.v[g, t] - m.w[g, t]

    else:
        # Otherwise operating state is coupled to previous period
        return m.u[g, t] - m.u[g, t - 1] == m.v[g, t] - m.w[g, t]


# Unit operating state
m.OPERATING_STATE = Constraint(m.G_E_THERM.union(m.G_C_THERM), m.T, rule=operating_state_logic_rule)


def minimum_on_time_rule(m, g, t):
    """Minimum number of hours generator must be on"""

    # Hours for existing units
    if g in self.existing_units.index:
        hours = self.existing_units.loc[g, ('PARAMETERS', 'MIN_ON_TIME')]

    # Hours for candidate units
    elif g in self.candidate_units.index:
        hours = self.candidate_units.loc[g, ('PARAMETERS', 'MIN_ON_TIME')]

    else:
        raise Exception(f'Min on time hours not found for generator: {g}')

    # Time index used in summation
    time_index = [k for k in range(t - int(hours), t) if k >= 0]

    # Constraint only defined over subset of timestamps
    if t >= hours:
        return sum(m.v[g, j] for j in time_index) <= m.u[g, t]
    else:
        return Constraint.Skip


def minimum_on_time_rule(m, g, t):
    """Minimum number of hours generator must be on"""

    # Hours for existing units
    if g in self.existing_units.index:
        hours = self.existing_units.loc[g, ('PARAMETERS', 'MIN_ON_TIME')]

    # Hours for candidate units
    elif g in self.candidate_units.index:
        hours = self.candidate_units.loc[g, ('PARAMETERS', 'MIN_ON_TIME')]

    else:
        raise Exception(f'Min on time hours not found for generator: {g}')

    # Time index used in summation
    time_index = [k for k in range(t - int(hours), t) if k >= 0]

    # Constraint only defined over subset of timestamps
    if t >= hours:
        return sum(m.v[g, j] for j in time_index) <= m.u[g, t]
    else:
        return Constraint.Skip


# Minimum on time constraint
m.MINIMUM_ON_TIME = Constraint(m.G_E_THERM.union(m.G_C_THERM), m.T, rule=minimum_on_time_rule)


def minimum_off_time_rule(m, g, t):
    """Minimum number of hours generator must be off"""

    # Hours for existing units
    if g in self.existing_units.index:
        hours = self.existing_units.loc[g, ('PARAMETERS', 'MIN_OFF_TIME')]

    # Hours for candidate units
    elif g in self.candidate_units.index:
        hours = self.candidate_units.loc[g, ('PARAMETERS', 'MIN_OFF_TIME')]

    else:
        raise Exception(f'Min off time hours not found for generator: {g}')

    # Time index used in summation
    time_index = [k for k in range(t - int(hours) + 1, t) if k >= 0]

    # Constraint only defined over subset of timestamps
    if t >= hours:
        return sum(m.w[g, j] for j in time_index) <= 1 - m.u[g, t]
    else:
        return Constraint.Skip


# Minimum off time constraint
m.MINIMUM_OFF_TIME = Constraint(m.G_E_THERM.union(m.G_C_THERM), m.T, rule=minimum_off_time_rule)