"""Unit commitment model subproblem"""

import time
from collections import OrderedDict

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

        # Minimum up reserve
        m.RESERVE_UP = Param(m.R, rule=minimum_region_up_reserve_rule)

        def ramp_rate_startup_rule(_m, g):
            """Startup ramp-rate (MW)"""

            if g in m.G_E_THERM:
                # Startup ramp-rate for existing thermal generators
                startup_ramp = self.data.existing_units.loc[g, ('PARAMETERS', 'RR_STARTUP')]

            elif g in m.G_C_THERM:
                # Startup ramp-rate for candidate thermal generators
                startup_ramp = self.data.candidate_units.loc[g, ('PARAMETERS', 'RR_STARTUP')]

            else:
                raise Exception(f'Unexpected generator {g}')

            return float(startup_ramp)

        # Startup ramp-rate for existing and candidate thermal generators
        m.RR_SU = Param(m.G_THERM, rule=ramp_rate_startup_rule)

        def ramp_rate_shutdown_rule(_m, g):
            """Shutdown ramp-rate (MW)"""

            if g in m.G_E_THERM:
                # Shutdown ramp-rate for existing thermal generators
                shutdown_ramp = self.data.existing_units.loc[g, ('PARAMETERS', 'RR_SHUTDOWN')]

            elif g in m.G_C_THERM:
                # Shutdown ramp-rate for candidate thermal generators
                shutdown_ramp = self.data.candidate_units.loc[g, ('PARAMETERS', 'RR_SHUTDOWN')]

            else:
                raise Exception(f'Unexpected generator {g}')

            return float(shutdown_ramp)

        # Shutdown ramp-rate for existing and candidate thermal generators
        m.RR_SD = Param(m.G_THERM, rule=ramp_rate_shutdown_rule)

        def ramp_rate_normal_up_rule(_m, g):
            """Ramp-rate up (MW/h) - when running"""

            if g in m.G_E_THERM:
                # Ramp-rate up for existing generators
                ramp_up = self.data.existing_units.loc[g, ('PARAMETERS', 'RR_UP')]

            elif g in m.G_C_THERM:
                # Ramp-rate up for candidate generators
                ramp_up = self.data.candidate_units.loc[g, ('PARAMETERS', 'RR_UP')]

            else:
                raise Exception(f'Unexpected generator {g}')

            return float(ramp_up)

        # Ramp-rate up (normal operation)
        m.RR_UP = Param(m.G_THERM, rule=ramp_rate_normal_up_rule)

        def ramp_rate_normal_down_rule(_m, g):
            """Ramp-rate down (MW/h) - when running"""

            if g in m.G_E_THERM:
                # Ramp-rate down for existing generators
                ramp_down = self.data.existing_units.loc[g, ('PARAMETERS', 'RR_DOWN')]

            elif g in m.G_C_THERM:
                # Ramp-rate down for candidate generators
                ramp_down = self.data.candidate_units.loc[g, ('PARAMETERS', 'RR_DOWN')]

            else:
                raise Exception(f'Unexpected generator {g}')

            return float(ramp_down)

        # Ramp-rate down (normal operation)
        m.RR_DOWN = Param(m.G_THERM, rule=ramp_rate_normal_down_rule)

        def min_power_output_rule(_m, g):
            """Minimum power output for thermal generators"""

            return float(self.data.existing_units.loc[g, ('PARAMETERS', 'MIN_GEN')])

        # Minimum power output
        m.P_MIN = Param(m.G_E_THERM, rule=min_power_output_rule)

        def min_power_output_proportion_rule(_m, g):
            """Minimum generator power output as a proportion of maximum output"""

            # Minimum power output for candidate thermal generators
            min_output = float(self.data.candidate_units.loc[g, ('PARAMETERS', 'MIN_GEN_PERCENT')] / 100)

            # Check that min output is non-negative
            assert min_output >= 0

            return min_output

        # Minimum power output (as a proportion of max capacity) for existing and candidate thermal generators
        m.P_MIN_PROP = Param(m.G_C_THERM, rule=min_power_output_proportion_rule)

        def battery_efficiency_rule(_m, g):
            """Battery efficiency"""

            if g in m.G_E_STORAGE:
                # Assumed efficiency based on candidate unit battery efficiency
                return float(0.92)

            elif g in m.G_C_STORAGE:
                return float(self.data.battery_properties.loc[g, 'CHARGE_EFFICIENCY'])

            else:
                raise Exception(f'Unknown storage ID encountered: {g}')

        # Lower bound for energy in storage unit at end of interval (assume = 0)
        m.Q_INTERVAL_END_LB = Param(m.G_C_STORAGE, initialize=0)

        # Battery efficiency
        m.BATTERY_EFFICIENCY = Param(m.G_STORAGE, rule=battery_efficiency_rule)

        # Energy in battering in interval prior to model start (assume battery initially completely discharged)
        m.Q0 = Param(m.G_C_STORAGE, initialize=0)

        # TODO: This is a placeholder. May not include existing units within model if horizon starts from 2016.
        m.STORAGE_ENERGY_CAPACITY = Param(m.G_E_STORAGE, initialize=100)

        def network_incidence_matrix_rule(_m, l, z):
            """Incidence matrix describing connections between adjacent NEM zones"""

            return float(self.data.network_incidence_matrix.loc[l, z])

        # Network incidence matrix
        m.INCIDENCE_MATRIX = Param(m.L, m.Z, rule=network_incidence_matrix_rule)

        def powerflow_min_rule(_m, l):
            """Minimum powerflow over network link"""

            return float(-self.data.powerflow_limits[l]['reverse'] * 100)

        # Lower bound for powerflow over link
        m.POWERFLOW_MIN = Param(m.L_I, rule=powerflow_min_rule)

        def powerflow_max_rule(_m, l):
            """Maximum powerflow over network link"""

            return float(self.data.powerflow_limits[l]['forward'] * 100)

        # Lower bound for powerflow over link
        m.POWERFLOW_MAX = Param(m.L_I, rule=powerflow_max_rule)

        def emissions_intensity_rule(_m, g):
            """Emissions intensity (tCO2/MWh)"""

            if g in m.G_E_THERM:
                # Emissions intensity
                emissions = self.data.existing_units.loc[g, ('PARAMETERS', 'EMISSIONS')]

            elif g in m.G_C_THERM:
                # Emissions intensity
                emissions = self.data.candidate_units.loc[g, ('PARAMETERS', 'EMISSIONS')]

            else:
                # Set emissions intensity = 0 for all solar, wind, hydro, and storage units
                emissions = 0

            return float(emissions)

        # Emissions intensities for all generators
        m.EMISSIONS_RATE = Param(m.G, rule=emissions_intensity_rule)

        def marginal_cost_rule(_m, g):
            """Marginal costs for existing and candidate generators

            Note: Marginal costs for existing and candidate thermal plant depend on fuel costs, which are time
            varying. Therefore marginal costs for thermal plant must be updated for each year in model horizon.
            """

            if g in m.G_THERM:
                # Placeholder - actual marginal cost updated each time model run / year parameters updated
                marginal_cost = float(0)

            elif (g in m.G_E_WIND) or (g in m.G_E_SOLAR) or (g in m.G_E_HYDRO):
                # Marginal cost = VOM cost for wind and solar generators
                marginal_cost = self.data.existing_units.loc[g, ('PARAMETERS', 'VOM')]

            elif (g in m.G_C_WIND) or (g in m.G_C_SOLAR):
                # Marginal cost = VOM cost for wind and solar generators
                marginal_cost = self.data.candidate_units.loc[g, ('PARAMETERS', 'VOM')]

            elif g in m.G_C_STORAGE:
                # Assume marginal cost = VOM cost of typical hydro generator (7 $/MWh)
                marginal_cost = 7

            else:
                raise Exception(f'Unexpected generator: {g}')

            assert marginal_cost >= 0, 'Cannot have negative marginal cost'

            return float(marginal_cost)

        # Marginal costs for all generators and time periods
        m.C_MC = Param(m.G, rule=marginal_cost_rule, mutable=True)

        # # TODO: Need to fix startup costs - incorporate variable capacity in startup cost formulation
        # def thermal_startup_cost_rule(_m, g):
        #     """Startup cost for existing and candidate thermal generators"""
        #
        #     return m.C_SU_MW[g] * m.P_MAX[g]
        #
        # # Startup cost - absolute cost [$]
        # m.C_SU = Expression(m.G_THERM, initialize=0, rule=thermal_startup_cost_rule)
        #
        # def thermal_shutdown_cost_rule(_m, g, i):
        #     """Startup cost for existing and candidate thermal generators"""
        #     # TODO: For now set shutdown cost = 0
        #     return m.C_SD_MW[g] * 0
        #
        # # Shutdown cost - absolute cost [$]
        # m.C_SD = Expression(m.G_E_THERM.union(m.G_C_THERM), m.I, rule=thermal_shutdown_cost_rule)

        def candidate_unit_big_m_rule(_m, g):
            """Big-M parameter for candidate unit capacity variable - used in linearisation"""

            # NEM zone
            zone = g.split('-')[0]

            if g in m.G_C_WIND:
                return float(self.data.candidate_unit_build_limits_dict[zone]['WIND'])

            elif g in m.G_C_SOLAR:
                return float(self.data.candidate_unit_build_limits_dict[zone]['SOLAR'])

            elif g in m.G_C_STORAGE:
                return float(self.data.candidate_unit_build_limits_dict[zone]['STORAGE'])

            elif g in m.G_C_THERM:
                # Arbitrarily large upper-bound for thermal units
                return float(99999)
            else:
                raise Exception(f'Unexpected generator encountered: {g}')

        # Big-M parameter for candidate unit capacity (want tight upper-bound for a capacity variable)
        m.B_UP = Param(m.G_C, rule=candidate_unit_big_m_rule)

        def startup_cost_rule(_m, g):
            """
            Startup costs for existing and candidate thermal units

            Note: costs are normalised by plant capacity e.g. $/MW
            """

            if g in m.G_E_THERM:
                # Startup cost for existing thermal units
                startup_cost = (self.data.existing_units.loc[g, ('PARAMETERS', 'SU_COST_WARM')]
                                / self.data.existing_units.loc[g, ('PARAMETERS', 'REG_CAP')])

            elif g in m.G_C_THERM:
                # Startup cost for candidate thermal units
                startup_cost = self.data.candidate_units.loc[g, ('PARAMETERS', 'SU_COST_WARM_MW')]

            else:
                raise Exception(f'Unexpected generator encountered: {g}')

            # Shutdown cost cannot be negative
            assert startup_cost >= 0, 'Negative startup cost'

            return float(startup_cost)

        # Generator startup costs - per MW
        m.C_SU_MW = Param(m.G_THERM, rule=startup_cost_rule)

        # Generator shutdown costs - per MW - assume zero for now TODO: May need to update this assumption
        m.C_SD_MW = Param(m.G_THERM, initialize=0)

        # Value of lost load [$/MWh]
        m.C_L = Param(initialize=10000)

        # -------------------------------------------------------------------------------------------------------------
        # Parameters to update each time model is run
        # -------------------------------------------------------------------------------------------------------------
        # Initial on-state rule - must be updated each time model is run
        m.U0 = Param(m.G_THERM, within=Binary, mutable=True, initialize=1)

        # Power output in interval prior to model start (assume = 0 for now)
        m.P0 = Param(m.G, mutable=True, within=NonNegativeReals, initialize=0)

        # Indicates if unit is available for given operating scenario
        m.F_SCENARIO = Param(m.G_E, mutable=True, within=Binary, initialize=1)

        # Wind capacity factor
        m.Q_WIND = Param(m.G_E_WIND.union(m.G_C_WIND), m.T, mutable=True, within=NonNegativeReals, initialize=0)

        # Solar capacity factor
        m.Q_SOLAR = Param(m.G_E_SOLAR.union(m.G_C_SOLAR), m.T, mutable=True, within=NonNegativeReals, initialize=0)

        # Hydro output
        m.P_H = Param(m.G_E_HYDRO, m.T, mutable=True, within=NonNegativeReals, initialize=0)

        # Demand
        m.DEMAND = Param(m.Z, m.T, mutable=True, within=NonNegativeReals, initialize=0)

        # Scenario duration
        m.RHO = Param(initialize=0, mutable=True)

        # Discount factor
        m.DISCOUNT_FACTOR = Param(initialize=0, mutable=True)

        # Dual variable associated with emissions constraint in investment plan problem
        m.FIXED_LAMBDA = Param(initialize=0, mutable=True)

        # Fixed candidate capacity - determined in investment plan subproblem
        m.FIXED_CAPACITY = Param(m.G_C, initialize=0, mutable=True)

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

        # Power output above minimum dispatchable level
        m.p = Var(m.G_THERM, m.T, within=NonNegativeReals, initialize=0)

        # Total power output TODO: May need to add existing storage as well
        m.p_total = Var(m.G.difference(m.G_C_STORAGE), m.T, within=NonNegativeReals, initialize=0)

        # Auxiliary variable - used to linearise interaction between candidate installed capacity and on indicator
        m.x = Var(m.G_C_THERM, m.T, within=NonNegativeReals, initialize=0)

        # Auxiliary variable - used to linearise interaction between candidate installed capacity and startup indicator
        m.y = Var(m.G_C_THERM, m.T, within=NonNegativeReals, initialize=0)

        # Auxiliary variable - used to linearise interaction between candidate installed capacity and shutdown indicator
        m.z = Var(m.G_C_THERM, m.T, within=NonNegativeReals, initialize=0)

        # Installed capacity for candidate units
        m.b = Var(m.G_C, within=NonNegativeReals, initialize=0)

        # Storage unit charging power
        m.p_in = Var(m.G_STORAGE, m.T, within=NonNegativeReals, initialize=0)

        # Storage unit discharging power
        m.p_out = Var(m.G_STORAGE, m.T, within=NonNegativeReals, initialize=0)

        # Storage unit energy (state of charge)
        m.q = Var(m.G_STORAGE, m.T, within=NonNegativeReals, initialize=0)

        # Powerflow between NEM zones
        m.p_flow = Var(m.L, m.T, initialize=0)

        # Lost-load
        m.p_V = Var(m.Z, m.T, within=NonNegativeReals, initialize=0)

        # Baseline - fix policy parameter
        m.baseline = Var(initialize=0)
        m.baseline.fix()

        # Permit price - fix policy parameter
        m.permit_price = Var(initialize=0)
        m.permit_price.fix()

        return m

    @staticmethod
    def define_expressions(m):
        """Define unit commitment problem expressions"""

        def energy_output_rule(_m, g, t):
            """Energy output"""

            # If a storage unit
            if g in m.G_STORAGE:
                if t != m.T.first():
                    return (m.p_out[g, t - 1] + m.p_out[g, t]) / 2
                else:
                    return (m.P0[g] + m.p_out[g, t]) / 2

            # For all other units
            elif g in m.G.difference(m.G_STORAGE):
                if t != m.T.first():
                    return (m.p_total[g, t - 1] + m.p_total[g, t]) / 2
                else:
                    return (m.P0[g] + m.p_total[g, t]) / 2

        # Energy output
        m.e = Expression(m.G, m.T, rule=energy_output_rule)

        def lost_load_energy_rule(_m, z, t):
            """
            Amount of lost-load energy.

            Note: Assumes lost-load energy in interval prior to model start (t=0) is zero.
            """

            if t != m.T.first():
                return (m.p_V[z, t] + m.p_V[z, t - 1]) / 2

            else:
                # Assume no lost energy in interval preceding model start
                return m.p_V[z, t] / 2

        # Lost-load energy
        m.e_V = Expression(m.Z, m.T, rule=lost_load_energy_rule)

        def thermal_operating_costs_rule(_m):
            """Cost to operate thermal generators for given scenario"""

            # Operating cost related to energy output + emissions charge
            operating_costs = (m.RHO
                               * sum((m.C_MC[g] + ((m.EMISSIONS_RATE[g] - m.baseline) * m.permit_price) * m.e[g, t])
                                     for g in m.G_THERM for t in m.T))

            # Existing unit start-up costs
            existing_on_off_costs = (m.RHO
                                     * sum((m.C_SU_MW[g] * m.P_MAX[g] * m.v[g, t])
                                           + (m.C_SD_MW[g] * m.P_MAX[g] * m.w[g, t]) for g in m.G_E_THERM for t in m.T))

            # Candidate unit start-up costs (note this depends on installed capacity)
            candidate_on_off_costs = (m.RHO
                                      * sum((m.C_SU_MW[g] * m.y[g, t]) + (m.C_SD_MW[g] * m.z[g, t])
                                            for g in m.G_C_THERM for t in m.T))

            # Total thermal unit costs
            total_cost = operating_costs + existing_on_off_costs + candidate_on_off_costs

            return total_cost

        # Operating cost - thermal units
        m.OP_T = Expression(rule=thermal_operating_costs_rule)

        def hydro_operating_costs_rule(_m):
            """Cost to operate hydro generators"""

            return m.RHO * sum(m.C_MC[g] * m.e[g, t] for g in m.G_E_HYDRO for t in m.T)

        # Operating cost - hydro generators
        m.OP_H = Expression(rule=hydro_operating_costs_rule)

        def wind_operating_costs_rule(_m):
            """Cost to operate wind generators"""

            # Existing wind generators - not eligible for subsidy
            existing = m.RHO * sum(m.C_MC[g] * m.e[g, t] for g in m.G_E_WIND for t in m.T)

            # Candidate wind generators - eligible for subsidy
            candidate = (m.RHO * sum((m.C_MC[g] - (m.baseline * m.permit_price)) * m.e[g, t]
                                     for g in m.G_C_WIND for t in m.T))

            # Total cost to operate wind units for the scenario
            total_cost = existing + candidate

            return total_cost

        # Operating cost - wind units
        m.OP_W = Expression(rule=wind_operating_costs_rule)

        def solar_operating_costs_rule(_m):
            """Cost to operate solar generators"""

            # Existing wind generators - not eligible for subsidy
            existing = m.RHO * sum(m.C_MC[g] * m.e[g, t] for g in m.G_E_SOLAR for t in m.T)

            # Candidate wind generators - eligible for subsidy
            candidate = (m.RHO * sum((m.C_MC[g] - (m.baseline * m.permit_price)) * m.e[g, t]
                                     for g in m.G_C_SOLAR for t in m.T))

            # Total cost to operate wind units for the scenario
            total_cost = existing + candidate

            return total_cost

        # Operating cost - solar units
        m.OP_S = Expression(rule=solar_operating_costs_rule)

        def storage_operating_costs_rule(_m):
            """Cost to operate storage units"""

            return m.RHO * sum(m.C_MC[g] * m.e[g, t] for g in m.G_STORAGE for t in m.T)

        # Operating cost - storage units
        m.OP_Q = Expression(rule=storage_operating_costs_rule)

        def lost_load_value_rule(_m):
            """Vale of lost load"""

            return m.RHO * sum(m.C_L * m.e_V[z, t] for z in m.Z for t in m.T)

        # Value of lost load
        m.OP_L = Expression(rule=lost_load_value_rule)

        def reserve_violation_penalty_rule(_m):
            """Penalty for violating reserve requirements"""

            return m.RHO * sum(m.C_L * m.r_up_violation[r, t] for r in m.R for t in m.T)

        # Value of reserve violation penalty - assumed penalty factor is same as that for lost load
        m.OP_R = Expression(rule=reserve_violation_penalty_rule)

        # Total operating cost for a given scenario
        m.SCEN = Expression(expr=m.OP_T + m.OP_H + m.OP_W + m.OP_S + m.OP_Q + m.OP_L + m.OP_R)

        # Total cost for operating scenario - including discounting
        m.COST = Expression(expr=m.DISCOUNT_FACTOR * m.SCEN)

        def emission_target_cost_rule(_m):
            """Penalty arising from emissions constraint in investment plan subproblem"""

            return m.FIXED_LAMBDA * m.RHO * sum(m.e[g, t] * m.EMISSIONS_RATE[g] for g in m.G_THERM for t in m.T)

        # Cost arising from emissions target - approximation based on investment plan subproblem solution
        m.EMISSIONS_COST_APPROX = Expression(rule=emission_target_cost_rule)

        # Objective function - sum of operational costs + emissions target
        m.OBJECTIVE_FUNCTION = Expression(expr=m.COST + m.EMISSIONS_COST_APPROX)

        return m

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
                return m.u[g, t] - m.u[g, t - 1] == m.v[g, t] - m.w[g, t]

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
            time_index = [k for k in range(t - int(hours) + 1, t + 1) if k >= 1]

            # Constraint only defined over subset of timestamps
            if t >= hours:
                return sum(m.v[g, j] for j in time_index) <= m.u[g, t]
            else:
                return Constraint.Skip

        # Minimum on time constraint
        m.MINIMUM_ON_TIME = Constraint(m.G_THERM, m.T, rule=minimum_on_time_rule)

        def minimum_off_time_rule(_m, g, t):
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
            time_index = [k for k in range(t - int(hours) + 1, t + 1) if k >= 1]

            # Constraint only defined over subset of timestamps
            if t >= hours:
                return sum(m.w[g, j] for j in time_index) <= 1 - m.u[g, t]
            else:
                return Constraint.Skip

        # Minimum off time constraint
        m.MINIMUM_OFF_TIME = Constraint(m.G_THERM, m.T, rule=minimum_off_time_rule)

        def ramp_rate_up_rule(_m, g, t):
            """Ramp-rate up constraint - normal operation"""

            # For all other intervals apart from the first
            if t > m.T.first():
                return (m.p[g, t] + m.r_up[g, t]) - m.p[g, t - 1] <= m.RR_UP[g]

            else:
                # Ramp-rate for first interval
                return m.p[g, t] + m.r_up[g, t] - m.P0[g] <= m.RR_UP[g]

        # Ramp-rate up limit
        m.RAMP_RATE_UP = Constraint(m.G_E_THERM.union(m.G_C_THERM), m.T, rule=ramp_rate_up_rule)

        def ramp_rate_down_rule(_m, g, t):
            """Ramp-rate down constraint - normal operation"""

            # For all other intervals apart from the first
            if t > m.T.first():
                return - m.p[g, t] + m.p[g, t - 1] <= m.RR_DOWN[g]

            else:
                # Ramp-rate for first interval
                return - m.p[g, t] + m.P0[g] <= m.RR_DOWN[g]

        # Ramp-rate up limit
        m.RAMP_RATE_DOWN = Constraint(m.G_THERM, m.T, rule=ramp_rate_down_rule)

        def power_output_within_limits_rule(_m, g, t):
            """Ensure power output + reserves within capacity limits"""

            # Left hand-side of constraint
            lhs = m.p[g, t] + m.r_up[g, t]

            # Existing thermal units - fixed capacity
            if g in m.G_E_THERM:
                rhs_1 = (m.P_MAX[g] - m.P_MIN[g]) * m.u[g, t]

                if t < m.T.last():
                    rhs_2 = (m.P_MAX[g] - m.RR_SD[g]) * m.w[g, t + 1]
                    rhs_3 = (m.RR_SU[g] - m.P_MIN[g]) * m.v[g, t + 1]

            # Candidate thermal units - must take into account variable capacity
            elif g in m.G_C_THERM:
                rhs_1 = (1 - m.P_MIN_PROP[g]) * m.x[g, t]

                if t < m.T.last():
                    rhs_2 = m.z[g, t] - (m.RR_SD[g] * m.w[g, t + 1])
                    rhs_3 = (m.RR_SU[g] * m.v[g, t + 1]) - (m.P_MIN_PROP[g] * m.y[g, t + 1])

            else:
                raise Exception(f'Unknown generator: {g}')

            # If not any period except for the last
            if t < m.T.last():
                return lhs <= rhs_1 - rhs_2 + rhs_3

            # If the last period, assumed that startup and shutdown indicators = 0
            else:
                return lhs <= rhs_1

        # Power output and reserves within limits
        m.POWER_OUTPUT_WITHIN_LIMITS = Constraint(m.G_THERM, m.T, rule=power_output_within_limits_rule)

        def total_power_thermal_rule(_m, g, t):
            """Total power output for thermal generators"""

            # Existing quick-start thermal generators
            if g in m.G_E_THERM.intersection(m.G_THERM_QUICK):

                # If not the last index
                if t != m.T.last():
                    return m.p_total[g, t] == m.P_MIN[g] * (m.u[g, t] + m.v[g, t + 1]) + m.p[g, t]

                # If the last index assume shutdown and startup indicator = 0
                else:
                    return m.p_total[g, t] == (m.P_MIN[g] * m.u[g, t]) + m.p[g, t]

            # Candidate quick-start generators (assume all candidate generators are quick-start)
            elif g in m.G_C_THERM.intersection(m.G_THERM_QUICK):

                # If not the last index
                if t != m.T.last():
                    return m.p_total[g, t] == m.P_MIN_PROP[g] * (m.x[g, t] + m.y[g, t + 1]) + m.p[g, t]

                # If the last index assume shutdown and startup indicator = 0
                else:
                    return m.p_total[g, t] == m.P_MIN_PROP[g] * m.x[g, t]

            # Existing slow-start thermal generators
            elif g in m.G_E_THERM.intersection(m.G_THERM_SLOW):
                # Startup duration
                SU_D = ceil(m.P_MIN[g] / m.RR_SU[g])

                # Startup power output trajectory increment
                ramp_up_increment = m.P_MIN[g] / SU_D

                # Startup power output trajectory
                P_SU = OrderedDict({k + 1: ramp_up_increment * k for k in range(0, SU_D + 1)})

                # Shutdown duration
                SD_D = ceil(m.P_MIN[g] / m.RR_SD[g])

                # Shutdown power output trajectory increment
                ramp_down_increment = m.P_MIN[g] / SD_D

                # Shutdown power output trajectory
                P_SD = OrderedDict({k + 1: m.P_MIN[g] - (ramp_down_increment * k) for k in range(0, SD_D + 1)})

                if t != m.T.last():
                    return (m.p_total[g, t]
                            == ((m.P_MIN[g] * (m.u[g, t] + m.v[g, t + 1])) + m.p[g, t]
                                + sum(P_SU[k] * m.v[g, t - k + SU_D + 2] if t - k + SU_D + 2 in m.T else 0 for k in
                                      range(1, SU_D + 1))
                                + sum(P_SD[k] * m.w[g, t - k + 2] if t - k + 2 in m.T else 0 for k in
                                      range(2, SD_D + 2))))
                else:
                    return (m.p_total[g, t]
                            == ((m.P_MIN[g] * m.u[g, t]) + m.p[g, t]
                                + sum(P_SU[k] * m.v[g, t - k + SU_D + 2] if t - k + SU_D + 2 in m.T else 0 for k in
                                      range(1, SU_D + 1))
                                + sum(P_SD[k] * m.w[g, t - k + 2] if t - k + 2 in m.T else 0 for k in
                                      range(2, SD_D + 2))))
            else:
                raise Exception(f'Unexpected generator: {g}')

        # Constraint yielding total power output
        m.TOTAL_POWER_THERMAL = Constraint(m.G_THERM, m.T, rule=total_power_thermal_rule)

        def max_power_output_thermal_rule(_m, g, t):
            """Ensure max power + reserve is always less than installed capacity for thermal generators"""

            # Existing thermal generators
            if g in m.G_E_THERM:
                return m.p_total[g, t] + m.r_up[g, t] <= m.P_MAX[g] * (1 - m.F_SCENARIO[g])

            # Candidate thermal generators
            elif g in m.G_C_THERM:
                return m.p_total[g, t] + m.r_up[g, t] <= m.b[g]

        # Max power output + reserve is always less than installed capacity
        m.MAX_POWER_THERMAL = Constraint(m.G_THERM, m.T, rule=max_power_output_thermal_rule)

        def max_power_output_wind_rule(_m, g, t):
            """Max power output from wind generators"""

            # Existing wind generators
            if g in m.G_E_WIND:
                return m.p_total[g, t] <= m.Q_WIND[g, t] * m.P_MAX[g]

            # Candidate wind generators
            if g in m.G_C_WIND:
                return m.p_total[g, t] <= m.Q_WIND[g, t] * m.b[g]

        # Max power output from wind generators
        m.MAX_POWER_WIND = Constraint(m.G_E_WIND.union(m.G_C_WIND), m.T, rule=max_power_output_wind_rule)

        def max_power_output_solar_rule(_m, g, t):
            """Max power output from solar generators"""

            # Existing solar generators
            if g in m.G_E_SOLAR:
                return m.p_total[g, t] <= m.Q_SOLAR[g, t] * m.P_MAX[g]

            # Candidate wind generators
            if g in m.G_C_SOLAR:
                return m.p_total[g, t] <= m.Q_SOLAR[g, t] * m.b[g]

        # Max power output from wind generators
        m.MAX_POWER_SOLAR = Constraint(m.G_E_SOLAR.union(m.G_C_SOLAR), m.T, rule=max_power_output_solar_rule)

        def max_power_output_hydro_rule(_m, g, t):
            """Max power output from hydro generators"""

            return m.p_total[g, t] <= m.P_H[g, t]

        # Max power output from hydro generators
        m.MAX_POWER_HYDRO = Constraint(m.G_E_HYDRO, m.T, rule=max_power_output_hydro_rule)

        def storage_max_power_out_rule(_m, g, t):
            """
            Maximum discharging power of storage unit - set equal to energy capacity. Assumes
            storage unit can completely discharge in 1 hour
            """

            if g in m.G_E_STORAGE:
                return m.p_in[g, t] <= m.STORAGE_ENERGY_CAPACITY[g]

            elif g in m.G_C_STORAGE:
                return m.p_in[g, t] <= m.b[g]

            else:
                raise Exception(f'Unknown storage unit: {g}')

        # Max MW out of storage device - discharging
        m.P_STORAGE_MAX_OUT = Constraint(m.G_STORAGE, m.T, rule=storage_max_power_out_rule)

        def storage_max_power_in_rule(_m, g, t):
            """
            Maximum charging power of storage unit - set equal to energy capacity. Assumes
            storage unit can completely charge in 1 hour
            """

            # Existing storage units
            if g in m.G_E_STORAGE:
                return m.p_out[g, t] + m.r_up[g, t] <= m.STORAGE_ENERGY_CAPACITY[g]

            # Candidate storage units
            elif g in m.G_C_STORAGE:
                return m.p_out[g, t] + m.r_up[g, t] <= m.b[g]

            else:
                raise Exception(f'Unknown storage unit: {g}')

        # Max MW into storage device - charging
        m.P_STORAGE_MAX_IN = Constraint(m.G_STORAGE, m.T, rule=storage_max_power_in_rule)

        def storage_energy_rule(_m, g, t):
            """Ensure storage unit energy is within unit's capacity"""

            # Existing storage units
            if g in m.G_E_STORAGE:
                return m.q[g, t] <= m.STORAGE_ENERGY_CAPACITY[g]

            # Candidate storage units
            elif g in m.G_C_STORAGE:
                return m.q[g, t] <= m.b[g]

            else:
                raise Exception(f'Unknown storage unit: {g}')

        # Storage unit energy is within unit's limits
        m.STORAGE_ENERGY_BOUNDS = Constraint(m.G_STORAGE, m.T, rule=storage_energy_rule)

        def storage_energy_transition_rule(_m, g, t):
            """Constraint that couples energy + power between periods for storage units"""

            if t != m.T.first():
                return (m.q[g, t]
                        == m.q[g, t - 1] + (m.BATTERY_EFFICIENCY[g] * m.p_in[g, t])
                        - (m.p_out[g, t] / m.BATTERY_EFFICIENCY[g]))
            else:
                # Assume battery completely discharged in first period
                return (m.q[g, t]
                        == m.Q0[g] + (m.BATTERY_EFFICIENCY[g] * m.p_in[g, t])
                        - (m.p_out[g, t] / m.BATTERY_EFFICIENCY[g]))

        # Account for inter-temporal energy transition within storage units
        m.STORAGE_ENERGY_TRANSITION = Constraint(m.G_C_STORAGE, m.T, rule=storage_energy_transition_rule)

        def storage_interval_end_lower_bound_rule(_m, g):
            """Ensure energy within storage unit at end of interval is greater than desired lower bound"""

            return m.Q_INTERVAL_END_LB[g] <= m.q[g, m.T.last()]

        # Ensure energy in storage unit at end of interval is above some desired lower bound
        m.STORAGE_INTERVAL_END_LOWER_BOUND = Constraint(m.G_STORAGE, rule=storage_interval_end_lower_bound_rule)

        def storage_interval_end_upper_bound_rule(_m, g):
            """
            Ensure energy within storage unit at end of interval is less than desired upper bound

            Note: Assuming upper bound for desired energy in unit at end of interval = installed capacity
            """

            # Existing units
            if g in m.G_E_STORAGE:
                return m.q[g, m.T.last()] <= m.STORAGE_ENERGY_CAPACITY[g]

            # Candidate units
            elif g in m.G_C_STORAGE:
                return m.q[g, m.T.last()] <= m.b[g]

        # Ensure energy in storage unit at end of interval is above some desired lower bound
        m.STORAGE_INTERVAL_END_UPPER_BOUND = Constraint(m.G_STORAGE, rule=storage_interval_end_upper_bound_rule)

        def power_balance_rule(_m, z, t):
            """Power balance for each NEM zone"""

            # Existing units within zone
            existing_units = [gen for gen, zone in self.data.existing_units_dict[('PARAMETERS', 'NEM_ZONE')].items()
                              if zone == z]

            # Candidate units within zone
            candidate_units = [gen for gen, zone in self.data.candidate_units_dict[('PARAMETERS', 'ZONE')].items()
                               if zone == z]

            # All generators within a given zone
            generators = existing_units + candidate_units

            # Storage units within a given zone TODO: will need to update if existing storage units are included
            storage_units = [gen for gen, zone in self.data.battery_properties_dict['NEM_ZONE'].items() if zone == z]

            return (sum(m.p_total[g, t] for g in generators) - m.DEMAND[z, t]
                    - sum(m.INCIDENCE_MATRIX[l, z] * m.p_flow[l, t] for l in m.L)
                    + sum(m.p_out[g, t] - m.p_in[g, t] for g in storage_units)
                    + m.p_V[z, t] == 0)

        # Power balance constraint for each zone and time period
        m.POWER_BALANCE = Constraint(m.Z, m.T, rule=power_balance_rule)

        def powerflow_lower_bound_rule(_m, l, t):
            """Minimum powerflow over a link connecting adjacent NEM zones"""

            return m.p_flow[l, t] >= m.POWERFLOW_MIN[l]

        # Constrain max power flow over given network link
        m.POWERFLOW_MIN_CONS = Constraint(m.L_I, m.T, rule=powerflow_lower_bound_rule)

        def powerflow_max_constraint_rule(_m, l, t):
            """Maximum powerflow over a link connecting adjacent NEM zones"""

            return m.p_flow[l, t] <= m.POWERFLOW_MAX[l]

        # Constrain max power flow over given network link
        m.POWERFLOW_MAX_CONS = Constraint(m.L_I, m.T, rule=powerflow_max_constraint_rule)

        def capacity_on_state_lin_cons_1_rule(_m, g, t):
            """Installed capacity and on-state variable - linearisation constraint 1"""

            return m.x[g, t] <= m.B_UP[g] * m.u[g, t]

        # Installed capacity and on-state variable - linearisation constraint 1
        m.CAPACITY_ON_LIN_CONS_1 = Constraint(m.G_C_THERM, m.T, rule=capacity_on_state_lin_cons_1_rule)

        def capacity_on_state_lin_cons_2_rule(_m, g, t):
            """Installed capacity and on-state variable - linearisation constraint 2"""

            return m.x[g, t] <= m.b[g]

        # Installed capacity and on-state variable - linearisation constraint 2
        m.CAPACITY_ON_LIN_CONS_2 = Constraint(m.G_C_THERM, m.T, rule=capacity_on_state_lin_cons_2_rule)

        def capacity_on_state_lin_cons_3_rule(_m, g, t):
            """Installed capacity and on-state variable - linearisation constraint 3"""

            return m.x[g, t] >= m.b[g] - (m.B_UP[g] * (1 - m.u[g, t]))

        # Installed capacity and on-state variable - linearisation constraint 3
        m.CAPACITY_ON_LIN_CONS_3 = Constraint(m.G_C_THERM, m.T, rule=capacity_on_state_lin_cons_3_rule)

        def capacity_startup_state_lin_cons_1_rule(_m, g, t):
            """Installed capacity and startup-state variable - linearisation constraint 1"""

            return m.y[g, t] <= m.B_UP[g] * m.v[g, t]

        # Installed capacity and startup-state variable - linearisation constraint 1
        m.CAPACITY_STARTUP_LIN_CONS_1 = Constraint(m.G_C_THERM, m.T, rule=capacity_startup_state_lin_cons_1_rule)

        def capacity_startup_state_lin_cons_2_rule(_m, g, t):
            """Installed capacity and startup-state variable - linearisation constraint 2"""

            return m.y[g, t] <= m.b[g]

        # Installed capacity and on-state variable - linearisation constraint 2
        m.CAPACITY_STARTUP_LIN_CONS_2 = Constraint(m.G_C_THERM, m.T, rule=capacity_startup_state_lin_cons_2_rule)

        def capacity_startup_state_lin_cons_3_rule(_m, g, t):
            """Installed capacity and startup-state variable - linearisation constraint 3"""

            return m.y[g, t] >= m.b[g] - (m.B_UP[g] * (1 - m.v[g, t]))

        # Installed capacity and on-state variable - linearisation constraint 3
        m.CAPACITY_STARTUP_LIN_CONS_3 = Constraint(m.G_C_THERM, m.T, rule=capacity_startup_state_lin_cons_3_rule)

        def capacity_shutdown_state_lin_cons_1_rule(_m, g, t):
            """Installed capacity and shutdown-state variable - linearisation constraint 1"""

            return m.z[g, t] <= m.B_UP[g] * m.w[g, t]

        # Installed capacity and startup-state variable - linearisation constraint 1
        m.CAPACITY_SHUTDOWN_LIN_CONS_1 = Constraint(m.G_C_THERM, m.T, rule=capacity_shutdown_state_lin_cons_1_rule)

        def capacity_shutdown_state_lin_cons_2_rule(_m, g, t):
            """Installed capacity and shutdown-state variable - linearisation constraint 2"""

            return m.z[g, t] <= m.b[g]

        # Installed capacity and on-state variable - linearisation constraint 2
        m.CAPACITY_SHUTDOWN_LIN_CONS_2 = Constraint(m.G_C_THERM, m.T, rule=capacity_shutdown_state_lin_cons_2_rule)

        def capacity_shutdown_state_lin_cons_3_rule(_m, g, t):
            """Installed capacity and shutdown-state variable - linearisation constraint 3"""

            return m.z[g, t] >= m.b[g] - (m.B_UP[g] * (1 - m.w[g, t]))

        # Installed capacity and on-state variable - linearisation constraint 3
        m.CAPACITY_SHUTDOWN_LIN_CONS_3 = Constraint(m.G_C_THERM, m.T, rule=capacity_shutdown_state_lin_cons_3_rule)

        def investment_capacity_coupling_rule(_m, g):
            """
            Constraint coupling investment subproblem solution to subproblems describing unit operation.

            Note: Dual variable will be used to update parameter values in investment plan subproblem
            """

            return m.b[g] - m.FIXED_CAPACITY[g] == 0

        # Fix capacity in subproblem to that value determined in investment plan subproblem
        m.FIXED_SUBPROBLEM_CAPACITY = Constraint(m.G_C, rule=investment_capacity_coupling_rule)

        return m

    @staticmethod
    def define_objective(m):
        """Define unit commitment problem objective function"""

        # Objective function
        m.OBJECTIVE = Objective(expr=m.OBJECTIVE_FUNCTION, sense=minimize)

        return m

    def construct_model(self):
        """Construct unit commitment model"""

        # Initialise model object
        m = ConcreteModel()

        # Add component allowing dual variables to be imported
        m.dual = Suffix(direction=Suffix.IMPORT)

        # Define sets
        m = self.components.define_sets(m)

        # Define parameters common to unit commitment sub-problems and investment plan
        m = self.components.define_parameters(m)

        # Define parameters specific to unit commitment sub-problem
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

    @staticmethod
    def update_parameters(m, parameters):
        """Populate model object with parameters for a given operating scenario"""

        # Keys = name of parameter to update, value = inner dict mapping parameter index to new values
        for parameter, values in parameters.items():

            if isinstance(values, dict):
                # Inner dictionary - keys = parameter index, value = new parameter value
                for index, new_value in values.items():
                    m.__getattribute__(parameter)[index] = new_value

            elif isinstance(values, float):
                # Only one value associated with parameter (not indexed)
                m.__getattribute__(parameter).value = values

            else:
                raise Exception(f'Unexpected data type encountered when updating parameters: {values}')

        return m

    def get_scenario_parameters(self, m, year, scenario):
        """Get parameters relating to a given operating scenario"""

        # Map between existing solar DUIDs and labels in input_traces database
        solar_map = {'BROKENH1': 'BROKENHILL', 'MOREESF1': 'MOREE', 'NYNGAN1': 'NYNGAN'}

        # Initialise parameter containers
        demand = {}
        hydro = {}
        solar = {}
        wind = {}

        # For all timestamps
        for t in m.T:
            # For all NEM zones
            for z in m.Z:
                # Demand in each NEM zone for a given operating scenario
                value = self.data.input_traces_dict[('DEMAND', z, t)][(year, scenario)]

                # Check that value is sufficiently large (prevent numerical ill conditioning)
                if value < 0.1:
                    value = 0

                demand[(z, t)] = float(value)

            # For all hydro generators
            for g in m.G_E_HYDRO:
                # Historic hydro output - for operating scenario
                value = self.data.input_traces_dict[('HYDRO', g, t)][(year, scenario)]

                # Set output to zero if negative or very small (SCADA data collection may cause -ve values)
                if value < 0.1:
                    value = 0

                # Hydro output values
                hydro[(g, t)] = float(value)

            # For all solar technologies
            for g in m.G_C_SOLAR:
                # Get zone and technology
                zone, tech = g.split('-')[0], g.split('-')[-1]

                # FFP used in model, but but FFP2 in traces database - convert so both are the same
                if tech == 'FFP':
                    tech = 'FFP2'

                # Solar capacity factors - candidate units
                value = self.data.input_traces_dict[('SOLAR', f'{zone}|{tech}', t)][(year, scenario)]

                # Check that value is sufficiently large (prevent numerical ill conditioning)
                if value < 0.1:
                    value = 0

                solar[(g, t)] = float(value)

            for g in m.G_E_SOLAR:
                # Solar capacity factor - existing units
                value = self.data.input_traces_dict[('SOLAR', solar_map[g], t)][(year, scenario)]

                # Check that value is sufficiently large (prevent numerical ill conditioning)
                if value < 0.1:
                    value = 0

                solar[(g, t)] = float(value)

            for g in m.G_E_WIND:
                # Wind bubble to which existing wind generator belongs
                bubble = self.data.existing_wind_bubble_map_dict['BUBBLE'][g]

                # Check that value is sufficiently large (prevent numerical ill conditioning)
                value = self.data.input_traces_dict[('WIND', bubble, t)][year, scenario]

                if value < 0.1:
                    value = 0

                # Wind capacity factor - existing units
                wind[(g, t)] = float(value)

            for g in m.G_C_WIND:
                # Wind bubble to which candidate wind generator belongs
                bubble = self.data.candidate_units_dict[('PARAMETERS', 'WIND_BUBBLE')][g]

                # Check that value is sufficiently large (prevent numerical ill conditioning)
                value = self.data.input_traces_dict[('WIND', bubble, t)][year, scenario]

                if value < 0.1:
                    value = 0

                # Wind capacity factor - existing units
                wind[(g, t)] = float(value)

        # Update scenario duration
        rho = self.data.input_traces_dict[('K_MEANS', 'METRIC', 'NORMALISED_DURATION')][(year, scenario)] * 365

        # All scenario parameters
        scenario_parameters = {'DEMAND': demand, 'P_H': hydro, 'Q_SOLAR': solar, 'Q_WIND': wind, 'RHO': rho}

        return scenario_parameters

    @staticmethod
    def get_iteration_parameters(m):
        """Get parameters that only need to be update once per iteration"""

        # TODO: Read stored values and set
        # Fixed dual variable associated with emissions constraint in investment plan subproblem
        fixed_lambda = float(0)

        # Fixed capacity determined in investment plan subproblem
        fixed_capacity = {g: float(0) for g in m.G_C}

        # All parameters that should be updated once per iteration
        parameters = {'FIXED_LAMBDA': fixed_lambda, 'FIXED_CAPACITY': fixed_capacity}

        return parameters

    def get_year_parameters(self, m, year):
        """Get year specific for a given year"""

        retirement_indicator = {}
        initial_on_state = {}
        initial_power_output = {}
        marginal_costs = {}

        for g in m.G_E:
            # Availability indicator - parameter defined in investment plan for each year
            retirement_indicator[g] = m.F[g, year]

        for g in m.G_E_THERM:
            # If existing thermal unit is not retired
            if retirement_indicator[g] == 1:

                # Initial on-state set to 0 if unit no longer active
                initial_on_state[g] = float(0)

            else:
                # Assume slow-start units are initially on (baseload generators)
                if g in m.G_THERM_SLOW:
                    initial_on_state[g] = float(1)

                # Assume other units are initially off (these will be quick-start thermal units)
                else:
                    initial_on_state[g] = float(0)

        for g in m.G:
            # If unit thermal unit is on in the first interval, set preceding interval output = min output
            if g in m.G_E_THERM:

                # If unit is on in the preceding interval, assume power output = min power output
                if initial_on_state[g] == 1:
                    initial_power_output[g] = model.P_MIN[g]

                # Else, assume power output is zero in preceding interval
                else:
                    initial_power_output[g] = float(0)

            else:
                # Assume power output in preceding interval = 0 for all other units TODO: May need to revise this
                initial_power_output[g] = float(0)

            if g in m.G_E_THERM:

                # Last year in the dataset for which fuel cost information exists
                max_year = self.data.existing_units.loc[g, 'FUEL_COST'].index.max()

                # If year in model horizon exceeds max year for which data are available use values for last
                # available year
                if year > max_year:
                    # Use final year in dataset to max year
                    year = max_year

                marginal_costs[g] = float(self.data.existing_units.loc[g, ('FUEL_COST', year)]
                                          * self.data.existing_units.loc[g, ('PARAMETERS', 'HEAT_RATE')])

            elif g in m.G_C_THERM:
                # Last year in the dataset for which fuel cost information exists
                max_year = self.data.candidate_units.loc[g, 'FUEL_COST'].index.max()

                # If year in model horizon exceeds max year for which data are available use values for last
                # available year
                if year > max_year:
                    # Use final year in dataset to max year
                    year = max_year

                marginal_costs[g] = float(self.data.candidate_units.loc[g, ('FUEL_COST', year)])

        # Discount factor applying to given year - assume computation in terms of 2016 present values
        if 2016 <= year < 2050:
            discount = 1 / ((1 + self.data.WACC) ** (year - 2016))

        # If the last year in the model horizon (2050), discount such that operating costs are paid in perpetuity
        elif year == 2050:
            discount = (1 / ((1 + self.data.WACC) ** (year - 2016))) * ((self.data.WACC + 1) / self.data.WACC)
        else:
            raise Exception(f'Check year. Value should be between 2016-2050, with 2016 as the base year: {year}')

        # All year-specific parameters
        year_parameters = {'F_SCENARIO': retirement_indicator, 'U0': initial_on_state, 'P0': initial_power_output,
                           'C_MC': marginal_costs, 'DISCOUNT_FACTOR': discount}

        return year_parameters

    def solve_model(self, m):
        """Solve model"""

        # Solve model
        self.opt.solve(m, tee=True, options=self.solver_options, keepfiles=self.keepfiles)

        # Log infeasible constraints if they exist
        log_infeasible_constraints(m)

        return m

    @staticmethod
    def fix_binary_variables(m):
        """Fix all binary variables"""

        for t in m.T:
            for g in m.G_THERM:
                m.u[g, t].fix()
                m.v[g, t].fix()
                m.w[g, t].fix()

                if g in m.G_C_THERM:
                    m.x[g, t].fix()
                    m.y[g, t].fix()
                    m.z[g, t].fix()

        return m

    @staticmethod
    def unfix_binary_variables(m):
        """Unfix all binary variables"""

        for t in m.T:
            for g in m.G_THERM:
                m.u[g, t].unfix()
                m.v[g, t].unfix()
                m.w[g, t].unfix()

                if g in m.G_C_THERM:
                    m.x[g, t].unfix()
                    m.y[g, t].unfix()
                    m.z[g, t].unfix()

        return m

    @staticmethod
    def save_subproblem_results(m):
        """Save selected results from subproblem - parameters to be passed to investment plan subproblem"""

        pass


if __name__ == '__main__':
    # Initialise object used to construct model
    uc = UnitCommitment()

    # Construct model object
    model = uc.construct_model()

    start = time.time()

    # Parameters obtained from investment plan subproblem - updated once per model iteration
    iteration_parameters = uc.get_iteration_parameters(model)

    # Parameters depending on a given year
    year_parameters = uc.get_year_parameters(model, 2035)

    # Parameters depending on a given operating scenario
    scenario_parameters = uc.get_scenario_parameters(model, 2035, 8)

    # Update model parameters
    model = uc.update_parameters(model, iteration_parameters)
    model = uc.update_parameters(model, year_parameters)
    model = uc.update_parameters(model, scenario_parameters)

    # Solve model
    model = uc.solve_model(model)

    # Fix all binary variables
    model = uc.fix_binary_variables(model)

    # Re-solve to obtain dual variables
    model = uc.solve_model(model)
    print(f'Solved in {time.time() - start:.2f}s')
