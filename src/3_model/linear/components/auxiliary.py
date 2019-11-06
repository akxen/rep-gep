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
    def __init__(self, first_year, final_year, scenarios_per_year, transition_year):
        self.common = CommonComponents(first_year, final_year, scenarios_per_year)
        self.analysis = AnalyseResults()
        self.prices = PriceSetter()
        self.transition_year = transition_year

        # Solver options
        self.keepfiles = False
        self.solver_options = {}  # 'MIPGap': 0.0005
        self.opt = SolverFactory('cplex', solver_io='lp')

    @staticmethod
    def define_sets(m):
        """Define model sets"""

        # Generators subject to penalties / credits under policy
        m.G_ELIGIBLE = m.G_THERM.union(m.G_C_WIND, m.G_C_SOLAR)

        return m

    def define_parameters(self, m):
        """Define model parameters"""

        # Power output placeholder
        m.POWER = Param(m.G_ELIGIBLE, m.Y, m.S, m.T, initialize=0, mutable=True)

        # Permit price
        m.PERMIT_PRICE = Param(m.Y, initialize=0, mutable=True)

        # Year after which a Refunded Emissions Payment scheme is enforced
        m.TRANSITION_YEAR = Param(initialize=self.transition_year, mutable=True)

        # Initial average price in year prior to model start
        m.YEAR_AVERAGE_PRICE_0 = Param(initialize=float(40), mutable=True)

        return m

    @staticmethod
    def define_variables(m):
        """Define model variables"""

        # Emissions intensity baseline
        # m.baseline = Var(m.Y, initialize=0, within=NonNegativeReals)
        m.baseline = Var(m.Y, initialize=0)

        # Dummy variables used to minimise price difference between years in model horizon
        m.z_p1 = Var(m.Y, initialize=0, within=NonNegativeReals)
        m.z_p2 = Var(m.Y, initialize=0, within=NonNegativeReals)

        return m

    def define_expressions(self, m, psg_results):
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

        def net_cost_rule(_m, g, y):
            """Max net marginal cost for each generator (assuming baseline = 0 tCO2/MWh)"""

            if g in m.G_ELIGIBLE:
                return m.C_MC[g, y] + (m.EMISSIONS_RATE[g] * m.PERMIT_PRICE[y])
            else:
                return m.C_MC[g, y]

        # Net marginal cost
        m.NET_COST = Expression(m.G, m.Y, rule=net_cost_rule)

        def max_net_cost(_m, y):
            """Max net marginal cost"""

            # Max net marginal cost in each year
            cost = [m.NET_COST[g, y].expr() for g in m.G]

            return max(cost)

        # Max net marginal cost
        m.MAX_NET_COST = Expression(m.Y, rule=max_net_cost)

        # Construct dictionary for faster lookup in following expressions
        max_net_cost = {y: m.MAX_NET_COST[y].expr() for y in m.Y}

        def approximate_price_rule(_m, z, y, s, t):
            """Approximated price from price setting generator"""

            # DUID of the price setting generator
            g = price_setters_dict['generator'][(z, y, s, t)]

            # Under load shedding changes to the baseline do not change marginal costs
            if price_setters_dict['price_normalised'][(z, y, s, t)] > max_net_cost[y]:
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

            return m.z_p1[y] + m.z_p2[y]

        # Absolute price difference between consecutive years
        m.YEAR_ABSOLUTE_PRICE_DIFFERENCE = Expression(m.Y, rule=year_absolute_price_difference_rule)

        # Total absolute price difference
        m.TOTAL_ABSOLUTE_PRICE_DIFFERENCE = Expression(expr=sum(m.YEAR_ABSOLUTE_PRICE_DIFFERENCE[y] for y in m.Y))

        def year_absolute_price_difference_weighted_rule(_m, y):
            """Weighted absolute price difference"""

            return m.YEAR_ABSOLUTE_PRICE_DIFFERENCE[y] * m.PRICE_WEIGHTS[y]

        # Weighted absolute price difference for each year
        m.YEAR_ABSOLUTE_PRICE_DIFFERENCE_WEIGHTED = Expression(m.Y, rule=year_absolute_price_difference_weighted_rule)

        # Weighted total absolute difference
        m.TOTAL_ABSOLUTE_PRICE_DIFFERENCE_WEIGHTED = Expression(expr=sum(m.YEAR_ABSOLUTE_PRICE_DIFFERENCE_WEIGHTED[y]
                                                                         for y in m.Y if y <= self.transition_year))

        def year_cumulative_price_difference_weighted_rule(_m, y):
            """Cumulative weighted price difference"""

            return sum(m.YEAR_ABSOLUTE_PRICE_DIFFERENCE_WEIGHTED[j] for j in m.Y if j <= y)

        # Weighted cumulative absolute price difference for each year in model horizon
        m.YEAR_CUMULATIVE_PRICE_DIFFERENCE_WEIGHTED = Expression(m.Y,
                                                                 rule=year_cumulative_price_difference_weighted_rule)

        def year_sum_cumulative_price_difference_weighted_rule(_m, y):
            """Sum of cumulative price difference up to year, y"""

            return sum(m.YEAR_CUMULATIVE_PRICE_DIFFERENCE_WEIGHTED[j] for j in m.Y if j <= y)

        # Sum of cumulative price differences for each year in model horizon
        m.YEAR_SUM_CUMULATIVE_PRICE_DIFFERENCE_WEIGHTED = Expression(m.Y,
                                                                     rule=year_sum_cumulative_price_difference_weighted_rule)

        return m

    def define_constraints(self, m):
        """Define model constraints"""

        def price_change_deviation_1_rule(_m, y):
            """Constraints used to compute absolute difference in average prices between successive years"""

            if y == m.Y.first():
                return m.z_p1[y] >= m.YEAR_AVERAGE_PRICE[y] - m.YEAR_AVERAGE_PRICE_0
            else:
                return m.z_p1[y] >= m.YEAR_AVERAGE_PRICE[y] - m.YEAR_AVERAGE_PRICE[y - 1]

        # Price difference dummy constraints
        m.PRICE_CHANGE_DEVIATION_1 = Constraint(m.Y, rule=price_change_deviation_1_rule)
        m.PRICE_CHANGE_DEVIATION_1.deactivate()

        def price_change_deviation_2_rule(_m, y):
            """Constraints used to compute absolute difference in average prices between successive years"""

            if y == m.Y.first():
                return m.z_p2[y] >= m.YEAR_AVERAGE_PRICE_0 - m.YEAR_AVERAGE_PRICE[y]
            else:
                return m.z_p2[y] >= m.YEAR_AVERAGE_PRICE[y - 1] - m.YEAR_AVERAGE_PRICE[y]

        # Price difference dummy constraints
        m.PRICE_CHANGE_DEVIATION_2 = Constraint(m.Y, rule=price_change_deviation_2_rule)
        m.PRICE_CHANGE_DEVIATION_2.deactivate()

        def price_bau_deviation_1_rule(_m, y):
            """Constraints used to compute absolute difference in average prices between successive years"""

            return m.z_p1[y] >= m.YEAR_AVERAGE_PRICE[y] - m.YEAR_AVERAGE_PRICE_0

        # Price difference dummy constraints
        m.PRICE_BAU_DEVIATION_1 = Constraint(m.Y, rule=price_bau_deviation_1_rule)
        m.PRICE_BAU_DEVIATION_1.deactivate()

        def price_bau_deviation_2_rule(_m, y):
            """Constraints used to compute absolute difference in average prices between successive years"""

            return m.z_p2[y] >= m.YEAR_AVERAGE_PRICE_0 - m.YEAR_AVERAGE_PRICE[y]

        # Price difference dummy constraints
        m.PRICE_BAU_DEVIATION_2 = Constraint(m.Y, rule=price_bau_deviation_2_rule)
        m.PRICE_BAU_DEVIATION_2.deactivate()

        def year_scheme_revenue_neutral_rule(_m, y):
            """Ensure that net scheme revenue in each year = 0 (equivalent to a REP scheme)"""

            return m.YEAR_SCHEME_REVENUE[y] == 0

        # Ensure scheme is revenue neutral in each year of model horizon (equivalent to a REP scheme)
        m.YEAR_NET_SCHEME_REVENUE_NEUTRAL_CONS = Constraint(m.Y, rule=year_scheme_revenue_neutral_rule)
        m.YEAR_NET_SCHEME_REVENUE_NEUTRAL_CONS.deactivate()

        def non_negative_transition_revenue_rule(_m):
            """Ensure that net scheme revenue over transition period >= 0 (equivalent to a REP scheme)"""

            return sum(m.YEAR_SCHEME_REVENUE[y] for y in m.Y if y <= self.transition_year) >= 0

        # Ensure scheme is revenue neutral over transition period
        m.NON_NEGATIVE_TRANSITION_REVENUE_CONS = Constraint(rule=non_negative_transition_revenue_rule)
        m.NON_NEGATIVE_TRANSITION_REVENUE_CONS.deactivate()

        return m

    def define_objective(self, m):
        """Define objective function"""

        # Minimise price difference between consecutive years
        m.OBJECTIVE = Objective(expr=m.YEAR_SUM_CUMULATIVE_PRICE_DIFFERENCE_WEIGHTED[self.transition_year],
                                sense=minimize)

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
    first_year_model, final_year_model, scenarios_per_year_model, transition_year_model = 2016, 2031, 5, 2028

    # Output directory
    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, 'output', 'local')

    # Results file to load
    results_filename = 'rep_case.pickle'

    # Object used to compute baseline trajectory
    baseline = BaselineUpdater(first_year_model, final_year_model, scenarios_per_year_model, transition_year_model)

    # Model results
    r_rep = baseline.analysis.load_results(output_directory, results_filename)

    # Get price setting generator results
    psg = baseline.prices.get_price_setting_generators_from_model_results(r_rep['stage_2_rep'][3])

    # Construct model
    model = baseline.construct_model(psg)

