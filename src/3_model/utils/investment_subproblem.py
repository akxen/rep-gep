"""Investment plan subproblem"""

import time

from pyomo.environ import *
from pyomo.util.infeasible import log_infeasible_constraints

from data import ModelData
from components import CommonComponents


class InvestmentPlan:
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
                # TODO: Update this value
                return float(100)

            if g in m.G_C_STORAGE:
                # TODO: Need to find reasonable FOM cost for storage units - setting = to MEL-WIND for now
                return float(self.data.candidate_units_dict[('PARAMETERS', 'FOM')]['MEL-WIND'])
            else:
                return float(self.data.candidate_units_dict[('PARAMETERS', 'FOM')][g])

        # Fixed operations and maintenance cost
        m.C_FOM = Param(m.G, rule=fixed_operations_and_maintenance_cost_rule)

        def discount_factor_rule(_m, y):
            """Discount factor"""

            # Discount factor
            discount_factor = 1 / ((1 + self.data.WACC) ** (y - m.Y.first()))

            return discount_factor

        # Discount factor - used to take into account the time value of money and compute present values
        m.DISCOUNT_FACTOR = Param(m.Y, rule=discount_factor_rule)

        # Total emissions (obtained by computing emissions from all sub-problems). Updated each iteration.
        m.TOTAL_EMISSIONS = Param(initialize=0, mutable=True)

        # Cumulative emissions target
        m.CUMULATIVE_EMISSIONS_TARGET = Param(initialize=0, mutable=True)

        # Cost of violating cumulative emissions constraint (assumed) [$/tCO2]
        m.C_E = Param(initialize=10000)

        # Weighted cost of capital - interest rate assumed in discounting + amortisation calculations
        m.WACC = Param(initialize=float(self.data.WACC))

        # Candidate capacity dual variable obtained from sub-problem solution. Updated each iteration.
        m.PSI_FIXED = Param(m.G_C, m.Y, m.S, initialize=1000000, mutable=True)

        # Candidate capacity variable obtained from sub-problem
        m.CANDIDATE_CAPACITY_FIXED = Param(m.G_C, m.Y, m.S, initialize=0, mutable=True)

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

        # Cumulative emissions constraint violation
        m.f_e = Var(within=NonNegativeReals, initialize=0)

        return m

    @staticmethod
    def define_expressions(m):
        """Define investment plan expressions"""

        def investment_cost_rule(_m, y):
            """Total amortised investment cost for a given year [$]"""
            return sum(m.GAMMA[g] * m.I_C[g, y] * m.x_c[g, y] for g in m.G_C)

        # Investment cost in a given year
        m.INV = Expression(m.Y, rule=investment_cost_rule)

        # Emissions constraint violation penalty
        m.PEN = Expression(expr=m.C_E * m.f_e)

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
            print('SOLAR LIMITS', z, y, m.SOLAR_BUILD_LIMITS[z], gens)

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
            print('WIND LIMITS', z, y, m.WIND_BUILD_LIMITS[z], gens)

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
            print('STORAGE LIMITS', z, y, m.STORAGE_BUILD_LIMITS[z], gens)

            if gens:
                return sum(m.a[g, y] for g in gens) - m.STORAGE_BUILD_LIMITS[z] <= 0
            else:
                return Constraint.Skip

        # Storage build limit constraint for each NEM zone
        m.STORAGE_BUILD_LIMIT_CONS = Constraint(m.Z, m.Y, rule=wind_build_limits_cons_rule)

        # Cumulative emissions target
        m.EMISSIONS_CONSTRAINT = Constraint(expr=m.TOTAL_EMISSIONS - m.CUMULATIVE_EMISSIONS_TARGET - m.f_e <= 0)

        return m

    @staticmethod
    def define_objective(m):
        """Define objective function"""

        def objective_function_rule(_m):
            """Investment plan objective function"""

            # Total investment cost over model horizon
            investment_cost = sum((m.DISCOUNT_FACTOR[y] / m.WACC) * m.INV[y] for y in m.Y)

            # Fixed operations and maintenance cost over model horizon
            fom_cost = sum(m.DISCOUNT_FACTOR[y] * m.FOM[y] for y in m.Y)

            # End of year operating costs (assumed to be paid in perpetuity to take into account end-of-year effects)
            end_of_year_cost = (m.DISCOUNT_FACTOR[m.Y.last()] / m.WACC) * m.FOM[m.Y.last()]

            # Sub-problem dual information
            dual_cost = sum(m.PSI_FIXED[g, y, s] * (m.CANDIDATE_CAPACITY_FIXED[g, y, s] - m.a[g, y])
                            for g in m.G_C for y in m.Y for s in m.S)

            # Objective function - note: also considers penalty (m.PEN) associated with emissions constraint violation
            objective_function = investment_cost + fom_cost + end_of_year_cost + m.PEN + dual_cost

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
        self.opt.solve(m, tee=True, options=self.solver_options, keepfiles=self.keepfiles)

        # Log infeasible constraints if they exist
        log_infeasible_constraints(m)

        return m

    def update_parameters(self):
        """Update model parameters"""
        pass

    @staticmethod
    def fix_binary_variables(m):
        """Fix all binary variables"""
        Var(m.G_C_THERM, m.Y, m.G_C_THERM_SIZE_OPTIONS, within=Binary, initialize=0)
        for g in m.G_C_THERM:
            for y in m.Y:
                for n in m.G_C_THERM_SIZE_OPTIONS:
                    # Fix binary variables related to discrete sizing decision for candidate thermal units
                    m.d[g, y, n].fix()

        return m

    def save_solution(self):
        """Save model solution"""
        pass


if __name__ == '__main__':
    # Object used to solve investment plan
    investment_plan = InvestmentPlan()

    # Construct model to solve
    model = investment_plan.construct_model()

    # Solve model
    start = time.time()
    model = investment_plan.solve_model(model)

    # Fix binary variables
    model = investment_plan.fix_binary_variables(model)

    # Re-solve to obtain dual variables for emissions constraint
    model = investment_plan.solve_model(model)
    print(f'Solved model in: {time.time() - start}s')
