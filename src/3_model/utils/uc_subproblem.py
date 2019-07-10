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

        # Minimum upward reserve
        m.RESERVE_UP = Param(m.R, rule=minimum_region_up_reserve_rule)

        def ramp_rate_startup_rule(m, g):
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

        def ramp_rate_shutdown_rule(m, g):
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

        # -------------------------------------------------------------------------------------------------------------
        # Parameters to update each time model is run
        # -------------------------------------------------------------------------------------------------------------
        # Initial on-state rule - must be updated each time model is run
        m.U0 = Param(m.G_THERM, within=Binary, mutable=True, initialize=1)

        # Power output in interval prior to model start (assume = 0 for now)
        m.P0 = Param(m.G, mutable=True, within=NonNegativeReals, initialize=0)

        # Indicates if unit is available for given operating scenario
        m.F_SCENARIO = Param(m.G, mutable=True, within=Binary, initialize=1)

        # Wind capacity factor
        m.Q_WIND = Param(m.G_E_WIND.union(m.G_C_WIND), m.T, mutable=True, within=NonNegativeReals, initialize=0)

        # Solar capacity factor
        m.Q_SOLAR = Param(m.G_E_SOLAR.union(m.G_C_SOLAR), m.T, mutable=True, within=NonNegativeReals, initialize=0)

        # Hydro output
        m.P_H = Param(m.G_E_HYDRO, m.T, mutable=True, within=NonNegativeReals, initialize=0)

        # Demand
        m.D = Param(m.Z, m.T, mutable=True, within=NonNegativeReals, initialize=0)

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
        m.p_V = Var(m.Z, m.T, initialize=0)

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
            # if t < m.T.last():

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

            # Storage units within a given zone TODO: will need to update if exisiting storage units are included
            storage_units = [gen for gen, zone in self.data.battery_properties_dict['NEM_ZONE'].items() if zone == z]

            return (sum(m.p_total[g, t] for g in generators) - m.DEMAND[z, t]
                    - sum(m.INCIDENCE_MATRIX[l, z] * m.p_flow[l, t] for l in m.L)
                    + sum(m.p_out[g, t] - m.p_in[g, t] for g in storage_units)
                    + m.p_lost_up[z, t] - m.p_lost_down[z, t] == 0)

        # Power balance constraint for each zone and time period
        m.POWER_BALANCE = Constraint(m.Z, m.T, rule=power_balance_rule)

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
