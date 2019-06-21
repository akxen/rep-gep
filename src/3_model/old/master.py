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
