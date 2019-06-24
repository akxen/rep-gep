"""Class used to construct investment planning master problem"""

import time

from pyomo.environ import *

from data import ModelData
from components import CommonComponents


class MasterProblem:
    # Pre-processed model data
    data = ModelData()

    def __init__(self):
        # Solver options
        self.keepfiles = False
        self.solver_options = {'Method': 1}  # 'MIPGap': 0.0005
        self.opt = SolverFactory('gurobi', solver_io='lp')

    @staticmethod
    def define_expressions(m):
        """Define master problem expressions"""

        def capacity_sizing_rule(_m, g, i):
            """Size of candidate units"""

            # Continuous size option for wind, solar, and storage units
            if g in m.G_C_WIND.union(m.G_C_SOLAR).union(m.G_C_STORAGE):
                return m.x_c[g, i]

            # Discrete size options for candidate thermal units
            elif g in m.G_C_THERM:
                return sum(m.d[g, i, n] * m.X_C_THERM_SIZE[g, n] for n in m.G_C_THERM_SIZE_OPTIONS)

            else:
                raise Exception(f'Unidentified generator: {g}')

        # Capacity sizing for candidate units
        m.X_C = Expression(m.G_C.union(m.G_C_STORAGE), m.I, rule=capacity_sizing_rule)

        def fom_cost_rule(_m):
            """Fixed operating and maintenance cost for candidate generators over model horizon"""

            return sum(m.C_FOM[g] * m.X_C[g, i] for g in m.G_C.union(m.G_C_SOLAR) for i in m.I)

        # Fixed operation and maintenance cost - absolute cost [$]
        m.C_FOM_TOTAL = Expression(rule=fom_cost_rule)

        def investment_cost_rule(_m):
            """Cost to invest in candidate technologies"""

            return sum(m.C_INV[g, i] * m.X_C[g, i] for g in m.G_C.union(m.G_C_STORAGE) for i in m.I)

        # Investment cost (fixed in subproblem) - absolute cost [$]
        m.C_INV_TOTAL = Expression(rule=investment_cost_rule)

        def max_generator_power_output_rule(_m, g, i):
            """
            Maximum power output from existing and candidate generators

            Note: candidate units will have their max power output determined by investment decisions which
            are made known in the master problem. Need to update these values each time model is run.
            """

            # Max output for existing generators equal to registered capacities
            if g in m.G_E:
                return m.EXISTING_GEN_REG_CAP[g]

            # Max output for candidate generators equal to installed capacities (variable in master problem)
            elif g in m.G_C.union(m.G_C_STORAGE):
                return sum(m.X_C[g, y] for y in m.I if y <= i)

            else:
                raise Exception(f'Unexpected generator: {g}')

        # Maximum power output for existing and candidate units (must be updated each time model is run)
        m.P_MAX = Expression(m.G, m.I, rule=max_generator_power_output_rule)

        def thermal_startup_cost_rule(_m, g, i):
            """Startup cost for existing and candidate thermal generators"""

            return m.C_SU_MW[g] * m.P_MAX[g, i]

        # Startup cost - absolute cost [$]
        m.C_SU = Expression(m.G_E_THERM.union(m.G_C_THERM), m.I, rule=thermal_startup_cost_rule)

        def thermal_shutdown_cost_rule(_m, g, i):
            """Startup cost for existing and candidate thermal generators"""
            # TODO: For now set shutdown cost = 0
            return m.C_SD_MW[g] * 0

        # Shutdown cost - absolute cost [$]
        m.C_SD = Expression(m.G_E_THERM.union(m.G_C_THERM), m.I, rule=thermal_shutdown_cost_rule)

        # Penalty imposed for violating emissions constraint
        m.C_EMISSIONS_VIOLATION = Expression(expr=m.emissions_target_exceeded * m.EMISSIONS_EXCEEDED_PENALTY)

        # Penalty imposed for violating revenue constraint
        m.C_REVENUE_VIOLATION = Expression(expr=m.revenue_shortfall * m.REVENUE_SHORTFALL_PENALTY)

        return m

    def define_blocks(self, m):
        """Define blocks for operating logic constraints"""

        def operating_scenario_block_rule(s, i, o):
            """Define blocks corresponding to each operating scenario"""

            def define_block_parameters(s):
                """Define parameters for each block"""

                # Energy output for a given generator (must be updated each time solution from subproblem available)
                s.FIXED_ENERGY = Param(m.G, m.T, initialize=0, mutable=True)

                # Fixed energy into storage units
                s.FIXED_ENERGY_IN = Param(m.G_C_STORAGE, m.T, initialize=0, mutable=True)

                # Fixed energy out of storage units
                s.FIXED_ENERGY_OUT = Param(m.G_C_STORAGE, m.T, initialize=0, mutable=True)

                # Fixed lost load (up)
                s.FIXED_p_lost_up = Param(m.Z, m.T, initialize=0, mutable=True)

                # Fixed lost load (down)
                s.FIXED_p_lost_down = Param(m.Z, m.T, initialize=0, mutable=True)

                return s

            def define_block_variables(s):
                """Define variables for each block"""

                # Startup state variable
                s.v = Var(m.G_E_THERM.union(m.G_C_THERM), m.T, within=Binary, initialize=0)

                # Shutdown state variable
                s.w = Var(m.G_E_THERM.union(m.G_C_THERM), m.T, within=Binary, initialize=0)

                # On-state variable
                s.u = Var(m.G_E_THERM.union(m.G_C_THERM), m.T, within=Binary, initialize=1)

                return s

            def define_block_expressions(s):
                """Define expressions for each block"""

                def thermal_operating_costs_rule(_s):
                    """Cost to operate existing and candidate thermal units"""

                    return (
                        sum((m.C_MC[g, i] + (m.EMISSIONS_RATE[g] - m.baseline[i]) * m.permit_price[i]) * s.FIXED_ENERGY[g, t]
                            + (m.C_SU[g, i] * s.v[g, t]) + (m.C_SD[g, i] * s.w[g, t])
                            for g in m.G_E_THERM.union(m.G_C_THERM) for t in m.T))

                # Existing and candidate thermal unit operating costs for given scenario
                s.C_OP_THERM = Expression(rule=thermal_operating_costs_rule)

                def hydro_operating_costs_rule(_s):
                    """Cost to operate existing hydro generators"""

                    return sum(m.C_MC[g, i] * s.FIXED_ENERGY[g, t] for g in m.G_E_HYDRO for t in m.T)

                # Existing hydro unit operating costs (no candidate hydro generators)
                s.C_OP_HYDRO = Expression(rule=hydro_operating_costs_rule)

                def solar_operating_costs_rule(_s):
                    """Cost to operate existing and candidate solar units"""

                    return (sum(m.C_MC[g, i] * s.FIXED_ENERGY[g, t] for g in m.G_E_SOLAR for t in m.T)
                            + sum(
                                (m.C_MC[g, i] - m.baseline[i] * m.permit_price[i]) * s.FIXED_ENERGY[g, t] for g in m.G_C_SOLAR
                                for t in
                                m.T))

                # Existing and candidate solar unit operating costs (only candidate solar eligible for credits)
                s.C_OP_SOLAR = Expression(rule=solar_operating_costs_rule)

                def wind_operating_costs_rule(_s):
                    """Cost to operate existing and candidate wind generators"""

                    return (sum(m.C_MC[g, i] * s.FIXED_ENERGY[g, t] for g in m.G_E_WIND for t in m.T)
                            + sum(
                                (m.C_MC[g, i] - m.baseline[i] * m.permit_price[i]) * s.FIXED_ENERGY[g, t] for g in m.G_C_WIND
                                for t in
                                m.T))

                # Existing and candidate solar unit operating costs (only candidate solar eligible for credits)
                s.C_OP_WIND = Expression(rule=wind_operating_costs_rule)

                def storage_unit_charging_cost_rule(_s):
                    """Cost to charge storage unit"""

                    return sum(m.C_MC[g, i] * s.FIXED_ENERGY_IN[g, t] for g in m.G_C_STORAGE for t in m.T)

                # Charging cost rule - no subsidy received when purchasing energy
                s.C_OP_STORAGE_CHARGING = Expression(rule=storage_unit_charging_cost_rule)

                def storage_unit_discharging_cost_rule(_s):
                    """
                    Cost to charge storage unit

                    Note: If storage units are included in the scheme this could create an undesirable outcome. Units
                    would be subsidised for each MWh they generate. Therefore they could be incentivised to continually charge
                    and then immediately discharge in order to receive the subsidy. For now assume the storage units are not
                    eligible to receive a subsidy for each MWh under the policy.
                    """
                    # - (m.baseline * m.permit_price))
                    return sum(m.C_MC[g, i] * s.FIXED_ENERGY_OUT[g, t] for g in m.G_C_STORAGE for t in m.T)

                # Discharging cost rule - assumes storage units are eligible under REP scheme
                s.C_OP_STORAGE_DISCHARGING = Expression(rule=storage_unit_discharging_cost_rule)

                # Candidate storage unit operating costs
                s.C_OP_STORAGE = Expression(expr=s.C_OP_STORAGE_CHARGING + s.C_OP_STORAGE_DISCHARGING)

                def lost_load_cost_rule(_s):
                    """Value of lost-load"""

                    return sum((s.FIXED_p_lost_up[z, t] + s.FIXED_p_lost_down[z, t]) * m.C_LOST_LOAD for z in m.Z for t in m.T)

                # Total cost of lost-load
                s.C_OP_LOST_LOAD = Expression(rule=lost_load_cost_rule)

                def total_operating_cost_rule(_s):
                    """Total operating cost"""

                    return s.C_OP_THERM + s.C_OP_HYDRO + s.C_OP_SOLAR + s.C_OP_WIND + s.C_OP_STORAGE + s.C_OP_LOST_LOAD

                # Total operating cost
                s.C_OP_TOTAL = Expression(rule=total_operating_cost_rule)

                return s

            def define_block_constraints(s):
                """Define constraints for each block representing an operating scenario"""

                def operating_state_logic_rule(_s, g, t):
                    """
                    Determine the operating state of the generator (startup, shutdown
                    running, off)
                    """

                    if t == m.T.first():
                        # Must use U0 if first period (otherwise index out of range)
                        return s.u[g, t] - m.U0[g] == s.v[g, t] - s.w[g, t]

                    else:
                        # Otherwise operating state is coupled to previous period
                        return s.u[g, t] - s.u[g, t - 1] == s.v[g, t] - s.w[g, t]

                # Unit operating state
                s.OPERATING_STATE = Constraint(m.G_E_THERM.union(m.G_C_THERM), m.T, rule=operating_state_logic_rule)

                def minimum_on_time_rule(_s, g, t):
                    """Minimum number of hours generator must be on"""

                    # Hours for existing units
                    if g in self.data.existing_units.index:
                        hours = self.data.existing_units_dict[('PARAMETERS', 'MIN_ON_TIME')][g]

                    # Hours for candidate units
                    elif g in self.data.candidate_units.index:
                        hours = self.data.candidate_units_dict[('PARAMETERS', 'MIN_ON_TIME')][g]

                    else:
                        raise Exception(f'Min on time hours not found for generator: {g}')

                    # Time index used in summation
                    time_index = [k for k in range(t - int(hours), t) if k >= 0]

                    # Constraint only defined over subset of timestamps
                    if t >= hours:
                        return sum(s.v[g, j] for j in time_index) <= s.u[g, t]
                    else:
                        return Constraint.Skip

                # Minimum on time constraint
                s.MINIMUM_ON_TIME = Constraint(m.G_E_THERM.union(m.G_C_THERM), m.T, rule=minimum_on_time_rule)

                def minimum_off_time_rule(_s, g, t):
                    """Minimum number of hours generator must be off"""

                    # Hours for existing units
                    if g in self.data.existing_units.index:
                        hours = self.data.existing_units_dict[('PARAMETERS', 'MIN_OFF_TIME')][g]

                    # Hours for candidate units
                    elif g in self.data.candidate_units.index:
                        hours = self.data.candidate_units_dict[('PARAMETERS', 'MIN_OFF_TIME')][g]

                    else:
                        raise Exception(f'Min off time hours not found for generator: {g}')

                    # Time index used in summation
                    time_index = [k for k in range(t - int(hours) + 1, t) if k >= 0]

                    # Constraint only defined over subset of timestamps
                    if t >= hours:
                        return sum(s.w[g, j] for j in time_index) <= 1 - s.u[g, t]
                    else:
                        return Constraint.Skip

                # Minimum off time constraint
                s.MINIMUM_OFF_TIME = Constraint(m.G_E_THERM.union(m.G_C_THERM), m.T, rule=minimum_off_time_rule)

                return s

            def construct_block(s):
                """Construct block for operating scenario"""

                # Define block parameters for given operating scenario
                start = time.time()
                s = define_block_parameters(s)
                print(f'Defined block parameters in: {time.time() - start}s')

                # Define block variables for given operating scenario
                start = time.time()
                s = define_block_variables(s)
                print(f'Defined block variables in: {time.time() - start}s')

                # Define block expressions for given operating scenario
                start = time.time()
                s = define_block_expressions(s)
                print(f'Defined block expressions in: {time.time() - start}s')

                # Define block constraints for given operating scenario
                start = time.time()
                s = define_block_constraints(s)
                print(f'Defined block constraints in: {time.time() - start}s')

                return s

            # Construct block
            construct_block(s)

        # Construct block
        m.SCENARIO = Block(m.I, m.O, rule=operating_scenario_block_rule)

        return m

    @staticmethod
    def define_constraints(m):
        """Define master problem constraints"""

        def discrete_investment_options_rule(m, g, i):
            """Can only select one capacity decision each year for units with discrete sizing options"""
            return sum(m.d[g, i, n] for n in m.G_C_THERM_SIZE_OPTIONS) == 1

        # Only one discrete investment option can be chosen
        m.DISCRETE_INVESTMENT_OPTIONS = Constraint(m.G_C_THERM, m.I, rule=discrete_investment_options_rule)

        def solar_build_limit_rule(m, i, z):
            """Ensure zone build limits are not violated in any period - solar units"""

            return sum(sum(m.x_c[g, i] for g in m.G_C_SOLAR) for j in m.I if j <= i) <= m.BUILD_LIMITS['SOLAR', z]

        # Max amount of solar capacity that can be built in a given zone
        m.SOLAR_BUILD_LIMIT = Constraint(m.I, m.Z, rule=solar_build_limit_rule)

        def wind_build_limit_rule(m, i, z):
            """Ensure zone build limits are not violated in any period - wind units"""

            return sum(sum(m.x_c[g, i] for g in m.G_C_WIND) for j in m.I if j <= i) <= m.BUILD_LIMITS['WIND', z]

        # Max amount of wind capacity that can be built in a given zone
        m.WIND_BUILD_LIMIT = Constraint(m.I, m.Z, rule=wind_build_limit_rule)

        def storage_build_limit_rule(m, i, z):
            """Ensure zone build limits are not violated in any period - storage units"""

            return sum(sum(m.x_c[g, i] for g in m.G_C_STORAGE) for j in m.I if j <= i) <= m.BUILD_LIMITS['STORAGE', z]

        # Max amount of storage capacity that can be built in a given zone
        m.STORAGE_BUILD_LIMIT = Constraint(m.I, m.Z, rule=storage_build_limit_rule)

        def scheme_revenue_constraint_rule(m):
            """Scheme must be revenue neutral over model horizon"""

            # TODO: Need to extract energy output values from subproblem
            return (sum(
                (m.EMISSIONS_RATE[g] - m.baseline[i]) * m.permit_price[i] * m.SCENARIO[i, o].FIXED_ENERGY[g, t] for i in
                m.I for o in m.O for g in m.G_E_THERM.union(m.G_C_THERM).union(m.G_C_WIND).union(m.G_C_SOLAR) for t in
                m.T)
                    <= m.EMISSIONS_TARGET + m.emissions_target_exceeded)

        # Revenue constraint - must break-even over model horizon
        m.REVENUE_CONSTRAINT = Constraint(rule=scheme_revenue_constraint_rule)

        def scheme_emissions_constraint_rule(m):
            """Emissions limit over model horizon"""

            # TODO: Need to extract energy output values from subproblem
            return (sum(
                (m.EMISSIONS_RATE[g] * m.SCENARIO[i, o].FIXED_ENERGY[g, t] for i in m.I for o in m.O
                 for g in m.G_E_THERM.union(m.G_C_THERM).union(m.G_C_WIND).union(m.G_C_SOLAR) for t in m.T))
                    <= m.EMISSIONS_TARGET + m.emissions_target_exceeded)

        # Emissions constraint - must be less than some target, else penalty imposed for each unit above target
        m.EMISSIONS_CONSTRAINT = Constraint(rule=scheme_emissions_constraint_rule)

        return m

    @staticmethod
    def define_objective(m):
        """Define master problem objective"""

        # Minimise total operating cost - include penalty for revenue / emissions constraint violations
        m.OBJECTIVE = Objective(expr=sum(m.SCENARIO[i, o].C_OP_TOTAL for i in m.I for o in m.O)
                                     + m.C_FOM_TOTAL
                                     + m.C_INV_TOTAL
                                     + m.C_EMISSIONS_VIOLATION + m.C_REVENUE_VIOLATION, sense=minimize)

        return m

    def construct_model(self):
        """Construct master problem model components"""

        # Used to define sets and parameters common to both master and master problem
        common_components = CommonComponents()

        # Initialise base model object
        m = ConcreteModel()

        # Define sets - common to both master and master problem
        m = common_components.define_sets(m)

        # Define parameters - common to both master and master problem
        m = common_components.define_parameters(m)

        # Define variables - common to both master and master problem
        m = common_components.define_variables(m)

        # Define expressions
        m = self.define_expressions(m)

        # Define blocks
        m = self.define_blocks(m)

        # Define constraints
        m = self.define_constraints(m)

        # Define objective
        m = self.define_objective(m)

        return m

    @staticmethod
    def add_cuts(m, subproblem_results):
        """
        Add Benders cuts to model using data obtained from subproblem solution

        Parameters
        ----------
        m : pyomo model object
            Master problem model object

        subproblem_results : dict
            Results obtained from subproblem solution

        Returns
        -------
        m : pyomo model object
            Master problem model object with additional constraints (Benders cuts) added
        """

        return m

    def solve_model(self, m):
        """Solve model"""

        # Solve model
        self.opt.solve(m, tee=True, options=self.solver_options, keepfiles=self.keepfiles)

        return m


if __name__ == '__main__':
    # Create object used to construct master problem
    master = MasterProblem()

    # Create master problem
    master_problem = master.construct_model()

    # Solve model
    master_problem = master.solve_model(master_problem)
