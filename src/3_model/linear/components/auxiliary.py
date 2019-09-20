"""Auxiliary program used to update emissions intensity baseline"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'base'))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, '4_analysis'))

import pickle

from pyomo.environ import *
from pyomo.util.infeasible import log_infeasible_constraints

from prices import PriceSetter
from analysis import AnalyseResults
from base.components import CommonComponents


class BaselineUpdater:
    def __init__(self, first_year, final_year, scenarios_per_year):
        self.common = CommonComponents(first_year, final_year, scenarios_per_year)
        self.analysis = AnalyseResults()
        self.prices = PriceSetter()

        # Solver options
        self.keepfiles = False
        self.solver_options = {}  # 'MIPGap': 0.0005
        self.opt = SolverFactory('cplex', solver_io='mps')

    @staticmethod
    def define_sets(m):
        """Define model sets"""

        # Generators subject to penalties / credits under policy
        m.G_ELIGIBLE = m.G_THERM.union(m.G_C_WIND, m.G_C_SOLAR)

        return m

    @staticmethod
    def define_parameters(m):
        """Define model parameters"""

        # Power output placeholder
        m.POWER = Param(m.G_ELIGIBLE, m.Y, m.S, m.T, initialize=0, mutable=True)

        # Permit price
        m.PERMIT_PRICE = Param(m.Y, initialize=0, mutable=True)

        # Year after which a Refunded Emissions Payment scheme is enforced
        m.TRANSITION_YEAR = Param(initialize=2020, mutable=True)

        # Lower bound for cumulative scheme revenue. Prevents cumulative scheme revenue from going below this bound.
        m.REVENUE_LOWER_BOUND = Param(initialize=float(-1e9), mutable=True)

        # Initial average price in year prior to model start
        m.YEAR_AVERAGE_PRICE_0 = Param(initialize=float(40), mutable=True)
        
        return m

    @staticmethod
    def define_variables(m):
        """Define model variables"""

        # Emissions intensity baseline
        m.baseline = Var(m.Y, initialize=0)

        # Dummy variables used to minimise price difference between years in model horizon
        m.z_1 = Var(m.Y, initialize=0, within=NonNegativeReals)
        m.z_2 = Var(m.Y, initialize=0, within=NonNegativeReals)

        return m

    @staticmethod
    def define_expressions(m, psg_results):
        """
        Define model expressions

        Parameters
        ----------
        m : pyomo model
            Pyomo model object to which expressions will be attached

        psg_results : pandas DataFrame
            Price setting generator results

        Returns
        -------
        m : pyomo model
            Pyomo model object with expressions attached
        """

        # Convert to dict for faster lookups
        price_setters_dict = psg_results.to_dict()

        def approximate_price_rule(_m, z, y, s, t):
            """Approximated price from price setting generator"""

            # DUID of the price setting generator
            g = price_setters_dict['generator'][(z, y, s, t)]

            # Under load shedding changes to the baseline do not change marginal costs
            if price_setters_dict['price_normalised'][(z, y, s, t)] >= 9000:
                return float(price_setters_dict['price_normalised'][(z, y, s, t)])

            # Price as function of the marginal unit's cost function
            elif g in m.G_ELIGIBLE:
                return m.C_MC[g, y] + (m.EMISSIONS_RATE[g] - m.baseline[y]) * m.PERMIT_PRICE[y]

            # Generator is ineligible to receive rebates
            else:
                return m.C_MC[g, y]

        # Approximate price based on price setting generator cost function
        m.PRICE_APPROX = Expression(m.Z, m.Y, m.S, m.T, rule=approximate_price_rule)

        def year_energy_revenue_rule(_m, y):
            """Total revenue from energy sales for a given year"""

            return sum(m.RHO[y, s] * m.PRICE_APPROX[z, y, s, t] * m.DEMAND[z, y, s, t]
                       for z in m.Z for s in m.S for t in m.T)

        # Scheme revenue in each year of model horizon
        m.YEAR_ENERGY_REVENUE = Expression(m.Y, rule=year_energy_revenue_rule)

        def year_energy_demand_rule(_m, y):
            """Total demand in a given year (MWh)"""

            return sum(m.RHO[y, s] * m.DEMAND[z, y, s, t] for z in m.Z for s in m.S for t in m.T)

        # Total demand (MWh) in a given year
        m.YEAR_ENERGY_DEMAND = Expression(m.Y, rule=year_energy_demand_rule)

        def year_average_price_rule(_m, y):
            """Average price in a given year"""

            return m.YEAR_ENERGY_REVENUE[y] / m.YEAR_ENERGY_DEMAND[y]

        # Average price for a given year
        m.YEAR_AVERAGE_PRICE = Expression(m.Y, rule=year_average_price_rule)

        def year_scheme_revenue_rule(_m, y):
            """Total scheme revenue for a given year"""

            return sum(m.RHO[y, s] * (m.EMISSIONS_RATE[g] - m.baseline[y]) * m.PERMIT_PRICE[y] * m.POWER[g, y, s, t]
                       for g in m.G_ELIGIBLE for s in m.S for t in m.T)

        # Total net scheme revenue collected in a given year
        m.YEAR_SCHEME_REVENUE = Expression(m.Y, rule=year_scheme_revenue_rule)

        def year_cumulative_scheme_revenue(_m, y):
            """Cumulative scheme revenue in a given year"""

            return sum(m.YEAR_SCHEME_REVENUE[j] for j in m.Y if j <= y)

        # Cumulative scheme revenue at end of a given year
        m.YEAR_CUMULATIVE_SCHEME_REVENUE = Expression(m.Y, rule=year_cumulative_scheme_revenue)

        def year_absolute_price_difference_rule(_m, y):
            """Absolute price difference between consecutive years"""

            return m.z_1[y] + m.z_2[y]

        # Absolute price difference between consecutive years
        m.YEAR_ABSOLUTE_PRICE_DIFFERENCE = Expression(m.Y, rule=year_absolute_price_difference_rule)

        # Total absolute price difference
        m.TOTAL_ABSOLUTE_PRICE_DIFFERENCE = Expression(expr=sum(m.YEAR_ABSOLUTE_PRICE_DIFFERENCE[y] for y in m.Y))

        # Weighted total absolute difference
        m.TOTAL_ABSOLUTE_PRICE_DIFFERENCE_WEIGHTED = Expression(expr=sum(m.YEAR_ABSOLUTE_PRICE_DIFFERENCE[y]
                                                                         * m.PRICE_WEIGHTS[y] for y in m.Y))

        return m

    @staticmethod
    def define_constraints(m):
        """Define model constraints"""

        def revenue_neutral_horizon_rule(_m):
            """Scheme is revenue neutral over whole model horizon"""

            return sum(m.YEAR_SCHEME_REVENUE[y] for y in m.Y) == 0

        # Enforce revenue neutrality over entire model horizon
        m.REVENUE_NEUTRAL_HORIZON = Constraint(rule=revenue_neutral_horizon_rule)
        m.REVENUE_NEUTRAL_HORIZON.deactivate()

        def revenue_neutral_transition_rule(_m):
            """Scheme is revenue neutral up until a transition year"""

            return sum(m.YEAR_SCHEME_REVENUE[j] for j in m.Y if j <= m.TRANSITION_YEAR.value) == 0

        # Enforce revenue neutrality up until transition year
        m.REVENUE_NEUTRAL_TRANSITION = Constraint(rule=revenue_neutral_transition_rule)
        m.REVENUE_NEUTRAL_TRANSITION.deactivate()

        def year_net_scheme_revenue_neutral_rule(_m, y):
            """Enforce scheme is revenue neutral for a given year"""

            return m.YEAR_SCHEME_REVENUE[y] == 0

        # Enforce revenue neutrality for a given year
        m.YEAR_NET_SCHEME_REVENUE_NEUTRAL_CONS = Constraint(m.Y, rule=year_net_scheme_revenue_neutral_rule)
        m.YEAR_NET_SCHEME_REVENUE_NEUTRAL_CONS.deactivate()

        def cumulative_revenue_lower_bound_rule(_m, y):
            """Ensure revenue never falls below a given lower bound for any year in model horizon"""

            return m.YEAR_CUMULATIVE_SCHEME_REVENUE[y] >= m.REVENUE_LOWER_BOUND

        # Ensure cumulative scheme revenue never goes below this level
        m.CUMULATIVE_REVENUE_LOWER_BOUND = Constraint(m.Y, rule=cumulative_revenue_lower_bound_rule)
        m.CUMULATIVE_REVENUE_LOWER_BOUND.deactivate()

        # def price_difference_1_rule(_m, y):
        #     """Constraints used to compute absolute difference in average prices between successive years"""
        #
        #     if y == m.Y.first():
        #         return m.z_1[y] >= m.YEAR_AVERAGE_PRICE[y] - m.YEAR_AVERAGE_PRICE_0
        #     else:
        #         return m.z_1[y] >= m.YEAR_AVERAGE_PRICE[y] - m.YEAR_AVERAGE_PRICE[y - 1]
        #
        # # Price difference dummy constraints
        # m.PRICE_DIFFERENCE_CONS_1 = Constraint(m.Y, rule=price_difference_1_rule)
        #
        # def price_difference_2_rule(_m, y):
        #     """Constraints used to compute absolute difference in average prices between successive years"""
        #
        #     if y == m.Y.first():
        #         return m.z_2[y] >= m.YEAR_AVERAGE_PRICE_0 - m.YEAR_AVERAGE_PRICE[y]
        #     else:
        #         return m.z_2[y] >= m.YEAR_AVERAGE_PRICE[y - 1] - m.YEAR_AVERAGE_PRICE[y]
        #
        # # Price difference dummy constraints
        # m.PRICE_DIFFERENCE_CONS_2 = Constraint(m.Y, rule=price_difference_2_rule)

        def price_difference_1_rule(_m, y):
            """Constraints used to compute absolute difference in average prices between successive years"""

            return m.z_1[y] >= m.YEAR_AVERAGE_PRICE[y] - m.YEAR_AVERAGE_PRICE_0

        # Price difference dummy constraints
        m.PRICE_DIFFERENCE_CONS_1 = Constraint(m.Y, rule=price_difference_1_rule)

        def price_difference_2_rule(_m, y):
            """Constraints used to compute absolute difference in average prices between successive years"""

            return m.z_2[y] >= m.YEAR_AVERAGE_PRICE_0 - m.YEAR_AVERAGE_PRICE[y]

        # Price difference dummy constraints
        m.PRICE_DIFFERENCE_CONS_2 = Constraint(m.Y, rule=price_difference_2_rule)

        def scheme_revenue_lower_envelope_rule(_m, y):
            """Ensure scheme revenue is greater than or equal to lower envelope"""

            return m.YEAR_CUMULATIVE_SCHEME_REVENUE[y] >= m.SCHEME_REVENUE_ENVELOPE_LO[y]

        # Ensure scheme revenue less than or equal to upper envelope
        m.SCHEME_REVENUE_ENVELOPE_LO_CONS = Constraint(m.Y, rule=scheme_revenue_lower_envelope_rule)
        m.SCHEME_REVENUE_ENVELOPE_LO_CONS.deactivate()

        return m

    @staticmethod
    def define_objective(m):
        """Define objective function"""

        # Minimise price difference between consecutive years
        m.OBJECTIVE = Objective(expr=m.TOTAL_ABSOLUTE_PRICE_DIFFERENCE_WEIGHTED, sense=minimize)

        return m

    def construct_model(self, psg_results):
        """
        Construct baseline updating model

        Parameters
        ----------
        psg_results : pandas DataFrame
            Price setting generator results

        Returns
        -------
        m : pyomo model
            Pyomo model object used to solve auxiliary price targeting program
        """

        # Initialise model object
        m = ConcreteModel()

        # Define common sets
        m = self.common.define_sets(m)

        # Define sets specific to baseline updating model
        m = self.define_sets(m)

        # Define common parameters
        m = self.common.define_parameters(m)

        # Define parameters specific to baseline updating model
        m = self.define_parameters(m)

        # Define model variables
        m = self.define_variables(m)

        # Define expressions - expression based on model results. Therefore need to read from output_dir
        m = self.define_expressions(m, psg_results)

        # Define constraints
        m = self.define_constraints(m)

        # Define objective function
        m = self.define_objective(m)

        return m

    @staticmethod
    def update_parameters(m, results):
        """Update model parameters based on primal results"""

        # Common keys - eligible generators only
        common_keys = set(m.POWER.keys()) & set(results['p'].keys())

        # Updated power values
        updated_values = {i: results['p'][i] for i in common_keys}

        # Power output
        m.POWER.store_values(updated_values)

        # Update permit prices
        m.PERMIT_PRICE.store_values(results['permit_price'])

        return m

    def solve_model(self, m):
        """Solve model"""

        # Solve model
        solve_status = self.opt.solve(m, tee=True, options=self.solver_options, keepfiles=self.keepfiles)

        # Log infeasible constraints if they exist
        log_infeasible_constraints(m)

        return m, solve_status


if __name__ == '__main__':
    # Model horizon and scenarios per year
    first_year_model, final_year_model, scenarios_per_year_model = 2016, 2040, 5

    # Output directory
    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, 'output', 'local')

    # Results file to load
    results_filename = 'carbon_tax_case.pickle'

    # Object used to compute baseline trajectory
    baseline = BaselineUpdater(first_year_model, final_year_model, scenarios_per_year_model)

    # Model results
    r_carbon_tax = baseline.analysis.load_results(output_directory, results_filename)

    # Get price setting generator results
    psg = baseline.prices.get_price_setting_generators_from_model_results(r_carbon_tax)

    # Construct model
    model = baseline.construct_model(psg)

    # Update parameters
    model = baseline.update_parameters(model, r_carbon_tax)

    # Solve model
    model.REVENUE_NEUTRAL_TRANSITION.activate()
    model.TRANSITION_YEAR = 2020
    for y in range(model.TRANSITION_YEAR.value, final_year_model + 1):
        model.REVENUE_NEUTRAL_YEAR[y].activate()
    model, status = baseline.solve_model(model)
