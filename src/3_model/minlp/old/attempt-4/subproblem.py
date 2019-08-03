"""Classes used to construct investment planning subproblem"""

import time
import pickle
import logging
from collections import OrderedDict

from pyomo.environ import *
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

        # Define expressions - common to both master and subproblem
        m = common_components.define_expressions(m)

        return m

    def _update_marginal_costs(self, m, g, i):
        """Marginal costs for existing and candidate generators

        Note: Marginal costs for existing and candidate thermal plant depend on fuel costs, which are time
        varying. Therefore marginal costs for thermal plant must be define for each year in model horizon.
        """

        if g in m.G_E_THERM:

            # Last year in the dataset for which fuel cost information exists
            max_year = max([year for cat, year in self.data.existing_units_dict.keys() if cat == 'FUEL_COST'])

            # If year in model horizon exceeds max year for which data are available use values for last
            # available year
            if i > max_year:
                # Use final year in dataset as max year
                i = max_year

            marginal_cost = float(self.data.existing_units_dict[('FUEL_COST', i)][g]
                                  * self.data.existing_units_dict[('PARAMETERS', 'HEAT_RATE')][g])
        elif g in m.G_C_THERM:

            # Last year in the dataset for which fuel cost information exists
            max_year = max([year for cat, year in self.data.candidate_units_dict.keys() if cat == 'FUEL_COST'])

            # If year in model horizon exceeds max year for which data are available use values for last
            # available year
            if i > max_year:
                # Use final year in dataset as max year
                i = max_year

            marginal_cost = float(self.data.candidate_units_dict[('FUEL_COST', i)][g]
                                  * self.data.candidate_units_dict[('PARAMETERS', 'HEAT_RATE')][g])

        else:
            raise Exception(f'Unexpected generator or year: {g} {i}')

        return marginal_cost

    def _update_investment_costs(self, m, g, i):
        """Update investment costs"""

        if g in m.G_C_STORAGE:
            # Build costs for batteries
            return float(self.data.battery_build_costs_dict[i][g] * 1000)
        else:
            # Build costs for other candidate units
            return float(self.data.candidate_units_dict[('BUILD_COST', i)][g] * 1000)

    def update_parameters_year(self, m, i):
        """Update model parameters for a given year and operating scenario"""

        for g in m.G_E_THERM.union(m.G_C_THERM):
            # Update marginal costs
            m.C_MC[g] = self._update_marginal_costs(m, g, i)

            # Update investment costs
            if g in m.G_C_THERM:
                m.C_INV[g] = self._update_investment_costs(m, g, i)

        return m


if __name__ == '__main__':
    # Start timer
    start_timer = time.time()

    # Define object used to construct subproblem model
    subproblem = Subproblem()

    # Construct subproblem model
    subproblem_model = subproblem.construct_model()

    # Prepare to read suffix values (dual values)
    subproblem_model.dual = Suffix(direction=Suffix.IMPORT)
    print(f'Constructed model in: {time.time() - start_timer}s')


