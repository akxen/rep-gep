"""Benders decomposition master problem"""

import os
import pickle

from pyomo.environ import *
from pyomo.util.infeasible import log_infeasible_constraints

from data import ModelData
from components import CommonComponents
from calculations import Calculations


class InvestmentPlan:
    # Pre-processed data for model construction
    data = ModelData()

    # Common model components to investment plan and operating sub-problems (sets)
    components = CommonComponents()

    # Object used to perform common calculations. E.g. discount factors.
    calculations = Calculations()

    def __init__(self):
        # Solver options
        self.keepfiles = False
        self.solver_options = {}  # 'MIPGap': 0.0005
        self.opt = SolverFactory('gurobi', solver_io='mps')

    def define_sets(self, m):
        """Define investment plan sets"""
        pass

    def define_parameters(self, m):
        """Investment plan model parameters"""

        def solar_build_limits_rule(_m, z):
            """Solar build limits for each NEM zone"""

            return float(self.data.candidate_unit_build_limits_dict[z]['SOLAR'])

        # Maximum solar capacity allowed per zone
        m.SOLAR_BUILD_LIMITS = Param(m.Z, rule=solar_build_limits_rule)

        def wind_build_limits_rule(_m, z):
            """Wind build limits for each NEM zone"""

            return float(self.data.candidate_unit_build_limits_dict[z]['WIND'])

        # Maximum wind capacity allowed per zone
        m.WIND_BUILD_LIMITS = Param(m.Z, rule=wind_build_limits_rule)

        def storage_build_limits_rule(_m, z):
            """Storage build limits for each NEM zone"""

            return float(self.data.candidate_unit_build_limits_dict[z]['STORAGE'])

        # Maximum storage capacity allowed per zone
        m.STORAGE_BUILD_LIMITS = Param(m.Z, rule=storage_build_limits_rule)

        def candidate_unit_build_costs_rule(_m, g, y):
            """
            Candidate unit build costs [$/MW]

            Note: build cost in $/MW. May need to scale if numerical conditioning problems.
            """

            if g in m.G_C_STORAGE:
                return float(self.data.battery_build_costs_dict[y][g] * 1000)

            else:
                return float(self.data.candidate_units_dict[('BUILD_COST', y)][g] * 1000)

        # Candidate unit build cost
        m.I_C = Param(m.G_C, m.Y, rule=candidate_unit_build_costs_rule)

        def candidate_unit_life_rule(_m, g):
            """Asset life of candidate units [years]"""
            # TODO: Use data from NTNPD. Just making an assumption for now.
            return float(25)

        # Candidate unit life
        m.A = Param(m.G_C, rule=candidate_unit_life_rule)

        def amortisation_rate_rule(_m, g):
            """Amortisation rate for a given investment"""

            # Numerator for amortisation rate expression
            num = self.data.WACC * ((1 + self.data.WACC) ** m.A[g])

            # Denominator for amortisation rate expression
            den = ((1 + self.data.WACC) ** m.A[g]) - 1

            # Amortisation rate
            amortisation_rate = num / den

            return amortisation_rate

        # Amortisation rate for a given investment
        m.GAMMA = Param(m.G_C, rule=amortisation_rate_rule)

        def fixed_operations_and_maintenance_cost_rule(_m, g):
            """Fixed FOM cost [$/MW/year]

            Note: Data in NTNDP is in terms of $/kW/year. Must multiply by 1000 to convert to $/MW/year
            """

            if g in m.G_E:
                return float(self.data.existing_units_dict[('PARAMETERS', 'FOM')][g] * 1000)

            elif g in m.G_C_THERM.union(m.G_C_WIND, m.G_C_SOLAR):
                return float(self.data.candidate_units_dict[('PARAMETERS', 'FOM')][g] * 1000)

            elif g in m.G_STORAGE:
                # TODO: Need to find reasonable FOM cost for storage units - setting = MEL-WIND for now
                return float(self.data.candidate_units_dict[('PARAMETERS', 'FOM')]['MEL-WIND'] * 1000)

            else:
                raise Exception(f'Unexpected generator encountered: {g}')

        # Fixed operations and maintenance cost
        m.C_FOM = Param(m.G, rule=fixed_operations_and_maintenance_cost_rule)

        # Weighted cost of capital - interest rate assumed in discounting + amortisation calculations
        m.WACC = Param(initialize=float(self.data.WACC))

        # Candidate capacity dual variable obtained from sub-problem solution. Updated each iteration.
        m.PSI_FIXED = Param(m.G_C, m.Y, m.S, initialize=0, mutable=True)

        # Lower bound for Benders auxiliary variable
        m.ALPHA_LOWER_BOUND = Param(initialize=float(-10e9))

        return m

    @staticmethod
    def define_variables(m):
        """Define investment plan variables"""

        # Investment in each time period
        m.x_c = Var(m.G_C, m.Y, within=NonNegativeReals, initialize=0)

        # Binary variable to select capacity size
        m.d = Var(m.G_C_THERM, m.Y, m.G_C_THERM_SIZE_OPTIONS, within=Binary, initialize=0)

        # Auxiliary variable - total capacity available at each time period
        m.a = Var(m.G_C, m.Y, initialize=0)

        # Auxiliary variable - gives lower bound for subproblem solution
        m.alpha = Var()

        return m

    def define_expressions(self, m):
        """Define investment plan expressions"""

        def investment_cost_rule(_m, y):
            """Total amortised investment cost for a given year [$]"""
            return sum(m.GAMMA[g] * m.I_C[g, y] * m.x_c[g, y] for g in m.G_C)

        # Investment cost in a given year
        m.INV = Expression(m.Y, rule=investment_cost_rule)

        def fom_cost_rule(_m, y):
            """Total fixed operations and maintenance cost for a given year (not discounted)"""

            # FOM costs for candidate units
            candidate_fom = sum(m.C_FOM[g] * m.a[g, y] for g in m.G_C)

            # FOM costs for existing units - note no FOM cost paid if unit retires
            existing_fom = sum(m.C_FOM[g] * m.P_MAX[g] * (1 - m.F[g, y]) for g in m.G_E)

            # Expression for total FOM cost
            total_fom = candidate_fom + existing_fom

            return total_fom

        # Fixed operating cost for candidate existing generators for each year in model horizon
        m.FOM = Expression(m.Y, rule=fom_cost_rule)

        def total_fom_cost_rule(_m):
            """Total discounted FOM cost over all years in model horizon"""

            return sum(self.calculations.discount_factor(y, m.Y.first()) * m.FOM[y] for y in m.Y)

        # Total discounted FOM cost over model horizon
        m.FOM_TOTAL = Expression(rule=total_fom_cost_rule)

        def total_fom_end_of_horizon_rule(_m):
            """FOM cost assumed to propagate beyond end of model horizon"""

            # Discount factor
            discount_factor = self.calculations.discount_factor(m.Y.last(), m.Y.first() * m.FOM[m.Y.last()])

            # Assumes discounted FOM cost in final year paid in perpetuity
            total_cost = (discount_factor / m.WACC) * m.FOM[m.Y.last()]

            return total_cost

        # Accounting for FOM beyond end of model horizon (EOH)
        m.FOM_EOH = Expression(rule=total_fom_end_of_horizon_rule)

        def total_investment_cost_rule(_m):
            """Total discounted amortised investment cost"""

            # Total discounted amortised investment cost
            total_cost = sum((self.calculations.discount_factor(y, m.Y.first()) / m.WACC) * m.INV[y] for y in m.Y)

            return total_cost

        # Total investment cost
        m.INV_TOTAL = Expression(rule=total_investment_cost_rule)

        # Total discounted investment and FOM cost (taking into account costs beyond end of model horizon)
        m.TOTAL_PLANNING_COST = Expression(expr=m.FOM_TOTAL + m.FOM_EOH + m.INV_TOTAL)

        return m

    @staticmethod
    def define_constraints(m):
        """Define feasible investment plan constraints"""

        def discrete_thermal_size_rule(_m, g, y):
            """Discrete sizing rule for candidate thermal units"""

            # Discrete size options for candidate thermal units
            size_options = {0: 0, 1: 100, 2: 400, 3: 400}

            return m.x_c[g, y] - sum(m.d[g, y, n] * float(size_options[n]) for n in m.G_C_THERM_SIZE_OPTIONS) == 0

        # Discrete investment size for candidate thermal units
        m.DISCRETE_THERMAL_SIZE = Constraint(m.G_C_THERM, m.Y, rule=discrete_thermal_size_rule)

        def single_discrete_selection_rule(_m, g, y):
            """Can only select one size option per investment period"""

            return sum(m.d[g, y, n] for n in m.G_C_THERM_SIZE_OPTIONS) - float(1) == 0

        # Single size selection constraint per investment period
        m.SINGLE_DISCRETE_SELECTION = Constraint(m.G_C_THERM, m.Y, rule=single_discrete_selection_rule)

        def total_capacity_rule(_m, g, y):
            """Total installed capacity in a given year"""

            return m.a[g, y] - sum(m.x_c[g, j] for j in m.Y if j <= y) == 0

        # Total installed capacity for each candidate technology type at each point in model horizon
        m.TOTAL_CAPACITY = Constraint(m.G_C, m.Y, rule=total_capacity_rule)

        def solar_build_limits_cons_rule(_m, z, y):
            """Enforce solar build limits in each NEM zone"""

            # Solar generators belonging to zone 'z'
            gens = [g for g in m.G_C_SOLAR if g.split('-')[0] == z]

            if gens:
                return sum(m.a[g, y] for g in gens) - m.SOLAR_BUILD_LIMITS[z] <= 0
            else:
                return Constraint.Skip

        # Storage build limit constraint for each NEM zone
        m.SOLAR_BUILD_LIMIT_CONS = Constraint(m.Z, m.Y, rule=solar_build_limits_cons_rule)

        def wind_build_limits_cons_rule(_m, z, y):
            """Enforce wind build limits in each NEM zone"""

            # Wind generators belonging to zone 'z'
            gens = [g for g in m.G_C_WIND if g.split('-')[0] == z]

            if gens:
                return sum(m.a[g, y] for g in gens) - m.WIND_BUILD_LIMITS[z] <= 0
            else:
                return Constraint.Skip

        # Wind build limit constraint for each NEM zone
        m.WIND_BUILD_LIMIT_CONS = Constraint(m.Z, m.Y, rule=wind_build_limits_cons_rule)

        def wind_build_limits_cons_rule(_m, z, y):
            """Enforce storage build limits in each NEM zone"""

            # Storage generators belonging to zone 'z'
            gens = [g for g in m.G_C_STORAGE if g.split('-')[0] == z]

            if gens:
                return sum(m.a[g, y] for g in gens) - m.STORAGE_BUILD_LIMITS[z] <= 0
            else:
                return Constraint.Skip

        # Storage build limit constraint for each NEM zone
        m.STORAGE_BUILD_LIMIT_CONS = Constraint(m.Z, m.Y, rule=wind_build_limits_cons_rule)

        # Bound on Benders decomposition lower-bound variable
        m.ALPHA_LOWER_BOUND_CONS = Constraint(expr=m.alpha >= m.ALPHA_LOWER_BOUND)

        # Container for benders cuts
        m.BENDERS_CUTS = ConstraintList()

        return m

    @staticmethod
    def define_objective(m):
        """Define objective function"""

        def objective_function_rule(_m):
            """Investment plan objective function"""

            # Objective function
            objective_function = m.TOTAL_PLANNING_COST + m.alpha

            return objective_function

        # Investment plan objective function
        m.OBJECTIVE = Objective(rule=objective_function_rule, sense=minimize)

        return m

    def construct_model(self):
        """Construct investment plan model"""

        # Initialise model object
        m = ConcreteModel()

        # Prepare to import dual variables
        m.dual = Suffix(direction=Suffix.IMPORT)

        # Define sets
        m = self.components.define_sets(m)

        # Define parameters common to all sub-problems
        m = self.components.define_parameters(m)

        # Define parameters
        m = self.define_parameters(m)

        # Define variables
        m = self.define_variables(m)

        # Define expressions
        m = self.define_expressions(m)

        # Define constraints
        m = self.define_constraints(m)

        # Define objective
        m = self.define_objective(m)

        return m

    def solve_model(self, m):
        """Solve model instance"""

        # Solve model
        self.opt.solve(m, tee=False, options=self.solver_options, keepfiles=self.keepfiles)

        # Log infeasible constraints if they exist
        log_infeasible_constraints(m)

        return m

    def get_cut_component(self, m, iteration, year, scenario, investment_solution_dir, uc_solution_dir):
        """Construct cut component"""

        # Discount factor
        discount = self.calculations.discount_factor(year, base_year=2016)

        # Duration
        duration = self.calculations.scenario_duration_days(year, scenario)

        with open(os.path.join(uc_solution_dir, f'uc-results_{iteration}_{year}_{scenario}.pickle'), 'rb') as f:
            uc_result = pickle.load(f)

        with open(os.path.join(investment_solution_dir, f'investment-results_{iteration}.pickle'), 'rb') as f:
            inv_result = pickle.load(f)

        # Construct cut component
        cut = discount * duration * sum(
            uc_result['PSI_FIXED'][g] * (m.a[g, year] - inv_result['a'][(g, year)]) for g in m.G_C)

        return cut

    def get_benders_cut(self, m, iteration, investment_solution_dir, uc_solution_dir):
        """Construct a Benders optimality cut"""

        # Cost accounting for total operating cost for each scenario over model horizon
        scenario_cost = self.calculations.get_total_discounted_operating_scenario_cost(iteration, uc_solution_dir)

        # Cost account for operating costs beyond end of model horizon
        scenario_cost_eoh = self.calculations.get_end_of_horizon_operating_cost(m, iteration, uc_solution_dir)

        # Total (fixed) cost to include in Benders cut
        total_cost = scenario_cost + scenario_cost_eoh

        # Cut components containing dual variables
        cut_components = sum(self.get_cut_component(m, iteration, y, s, investment_solution_dir, uc_solution_dir)
                             for y in m.Y for s in m.S)

        # Benders cut
        cut = m.alpha >= total_cost + cut_components

        return cut

    def add_benders_cut(self, m, iteration, investment_solution_dir, uc_solution_dir):
        """Update model parameters"""

        # Construct Benders optimality cut
        cut = self.get_benders_cut(m, iteration, investment_solution_dir, uc_solution_dir)

        # Add Benders cut to constraint list
        m.BENDERS_CUTS.add(expr=cut)

        return m

    @staticmethod
    def fix_binary_variables(m):
        """Fix all binary variables"""

        for g in m.G_C_THERM:
            for y in m.Y:
                for n in m.G_C_THERM_SIZE_OPTIONS:
                    # Fix binary variables related to discrete sizing decision for candidate thermal units
                    m.d[g, y, n].fix()

        return m

    @staticmethod
    def unfix_binary_variables(m):
        """Unfix all binary variables"""

        for g in m.G_C_THERM:
            for y in m.Y:
                for n in m.G_C_THERM_SIZE_OPTIONS:
                    # Unfix binary variables related to discrete sizing decision for candidate thermal units
                    m.d[g, y, n].unfix()

        return m

    @staticmethod
    def save_solution(m, iteration, investment_solution_dir):
        """Save model solution"""

        # Save investment plan
        output = {'a': m.a.get_values(),
                  'x_c': m.x_c.get_values(),
                  'OBJECTIVE': m.OBJECTIVE.expr()}

        # Save investment plan results
        with open(os.path.join(investment_solution_dir, f'investment-results_{iteration}.pickle'), 'wb') as f:
            pickle.dump(output, f)

    @staticmethod
    def initialise_investment_plan(m, solution_dir):
        """Initial values for investment plan - used in first iteration. Assume candidate capacity = 0"""

        # Feasible investment plan for first iteration
        plan = {'a': {(g, y): float(0) for g in m.G_C for y in m.Y},
                'x_c': {(g, y): float(0) for g in m.G_C for y in m.Y},
                'OBJECTIVE': -1e9}

        # Save investment plan
        with open(os.path.join(solution_dir, 'investment-results_1.pickle'), 'wb') as f:
            pickle.dump(plan, f)

        return plan


if __name__ == '__main__':
    # Object used to solve investment plan
    investment_plan = InvestmentPlan()

    # Construct model to solve
    model = investment_plan.construct_model()

    # Directory where unit commitment subproblem results can be found
    uc_results_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, 'output', 'operational_plan')

    # Directory in which to store investment plan (master problem) results
    investment_plan_results_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, 'output',
                                                     'investment_plan')

    # Initialise feasible solution
    investment_plan.initialise_investment_plan(model, investment_plan_results_directory)
