"""Classes used to define mathematical program with primal and dual constraints for generation expansion planning"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'base'))

from pyomo.environ import *
from pyomo.util.infeasible import log_infeasible_constraints

from base.data import ModelData
from base.components import CommonComponents


class Primal:
    def __init__(self):
        self.data = ModelData()
        self.components = CommonComponents()

        # Solver options
        self.keepfiles = False
        self.solver_options = {}  # 'MIPGap': 0.0005
        self.opt = SolverFactory('cplex', solver_io='mps')

    @staticmethod
    def define_variables(m):
        """Primal problem variables"""

        # Investment capacity in each year
        m.x_c = Var(m.G_C, m.Y, initialize=0)

        # Cumulative installed capacity
        m.a = Var(m.G_C, m.Y, initialize=0)

        # Power output
        m.p = Var(m.G.difference(m.G_STORAGE), m.Y, m.S, m.T, initialize=0)

        # Max charging power for storage units
        m.p_in = Var(m.G_STORAGE, m.Y, m.S, m.T, initialize=0)

        # Max discharging power for storage units
        m.p_out = Var(m.G_STORAGE, m.Y, m.S, m.T, initialize=0)

        # Storage unit energy
        m.q = Var(m.G_STORAGE, m.Y, m.S, m.T, initialize=0)

        # Lost load
        m.p_V = Var(m.Z, m.Y, m.S, m.T, initialize=0)

        # Powerflow of links connecting NEM zones
        m.p_L = Var(m.L, m.Y, m.S, m.T, initialize=0)

        # Energy output
        m.e = Var(m.G, m.Y, m.S, m.T, initialize=0)

        # Lost load energy
        m.e_V = Var(m.Z, m.Y, m.S, m.T, initialize=0)

        # Fix emissions intensity baseline [tCO2/MWh]
        m.baseline.fix(0)

        # Fix permit price [$/tCO2]
        m.permit_price.fix(0)

        return m

    @staticmethod
    def define_expressions(m):
        """Primal problem expressions"""

        def yearly_investment_cost_rule(_m, y):
            """Yearly investment cost"""
            return (1 / m.INTEREST_RATE) * sum(m.GAMMA[g] * m.I_C[g, y] * m.x_c[g, y] for g in m.G_C)

        # Yearly investment cost
        m.INV = Expression(m.Y, rule=yearly_investment_cost_rule)

        def yearly_fom_cost_rule(_m, y):
            """Yearly fixed operations and maintenance cost"""

            # FOM cost for existing units
            existing = sum(m.C_FOM[g] * (m.P_MAX[g] * (1 - m.F[g, y])) for g in m.G_E)

            # FOM cost for candidate units
            candidate = sum(m.C_FOM[g] * m.a[g, y] for g in m.G_C)

            return existing + candidate

        # Yearly FOM cost
        m.FOM = Expression(m.Y, rule=yearly_fom_cost_rule)

        def thermal_operating_costs_rule(_m, y, s):
            """Cost to operating thermal units for a given scenario"""

            return sum((m.C_MC[g, y] + ((m.EMISSIONS_RATE[g] - m.baseline[y]) * m.permit_price[y])) * m.e[g, y, s, t]
                       for g in m.G_THERM for t in m.T)

        # Operating costs for a given scenario - thermal units
        m.OP_T = Expression(m.Y, m.S, rule=thermal_operating_costs_rule)

        def hydro_operating_costs_rule(_m, y, s):
            """Costs to operate hydro units for a given scenario"""

            return sum(m.C_MC[g, y] * m.e[g, y, s, t] for g in m.G_E_HYDRO for t in m.T)

        # Operating costs for a given scenario - hydro units
        m.OP_H = Expression(m.Y, m.S, rule=hydro_operating_costs_rule)

        def wind_operating_costs_rule(_m, y, s):
            """Cost to operate wind units for a given scenario"""

            # Cost for existing wind units
            existing = sum(m.C_MC[g, y] * m.e[g, y, s, t] for g in m.G_E_WIND for t in m.T)

            # Cost for candidate wind units
            candidate = sum((m.C_MC[g, y] - (m.baseline[y] * m.permit_price[y])) * m.e[g, y, s, t]
                            for g in m.G_C_WIND for t in m.T)

            return existing + candidate

        # Operating costs for a given scenario - wind units
        m.OP_W = Expression(m.Y, m.S, rule=wind_operating_costs_rule)

        def solar_operating_costs_rule(_m, y, s):
            """Cost to operate solar units for a given scenario"""

            # Cost for existing solar units
            existing = sum(m.C_MC[g, y] * m.e[g, y, s, t] for g in m.G_E_SOLAR for t in m.T)

            # Cost for candidate solar units
            candidate = sum((m.C_MC[g, y] - (m.baseline[y] * m.permit_price[y])) * m.e[g, y, s, t]
                            for g in m.G_C_SOLAR for t in m.T)

            return existing + candidate

        # Operating costs for a given scenario - solar units
        m.OP_S = Expression(m.Y, m.S, rule=solar_operating_costs_rule)

        def storage_operating_costs_rule(_m, y, s):
            """Cost to operate storage units for a given scenario"""

            return sum(m.C_MC[g, y] * m.e[g, y, s, t] for g in m.G_STORAGE for t in m.T)

        # Operating costs for a given scenario - storage units
        m.OP_Q = Expression(m.Y, m.S, rule=storage_operating_costs_rule)

        def lost_load_value_rule(_m, y, s):
            """Value of lost load in a given scenario"""

            return sum(m.C_L * m.e_V[z, y, s, t] for z in m.Z for t in m.T)

        # Operating costs for a given scenario - lost load
        m.OP_L = Expression(m.Y, m.S, rule=lost_load_value_rule)

        def scenario_cost_rule(_m, y, s):
            """Total operating cost for a given scenario"""

            return m.OP_T[y, s] + m.OP_H[y, s] + m.OP_W[y, s] + m.OP_S[y, s] + m.OP_Q[y, s] + m.OP_L[y, s]

        # Total cost for a given operating scenario
        m.SCEN = Expression(m.Y, m.S, rule=scenario_cost_rule)

        def year_cost_rule(_m, y):
            """Total cost for a given year (not discounted)"""

            return sum(m.RHO[y, s] * m.SCEN[y, s] for s in m.S)

        # Total cost for a given year
        m.OP = Expression(m.Y, rule=year_cost_rule)

        def end_of_horizon_cost_rule(_m):
            """Operating cost beyond model horizon"""

            return (m.DELTA[m.Y.last()] / m.INTEREST_RATE) * (m.OP[m.Y.last()] + m.FOM[m.Y.last()])

        # End of horizon cost
        m.EOH = Expression(rule=end_of_horizon_cost_rule)

        # Total present value
        def total_present_value_rule(_m):
            """Total present value - investment + operating costs + end of horizon cost"""

            return sum(m.DELTA[y] * (m.INV[y] + m.FOM[y] + m.OP[y]) for y in m.Y) + m.EOH

        # Total present value
        m.TPV = Expression(rule=total_present_value_rule)

        return m

    def define_constraints(self, m):
        """Primal problem constraints"""

        def non_negative_capacity_rule(_m, g, y):
            """Non-negative capacity for candidate units"""

            return - m.x_c[g, y] <= 0

        # Non-negative capacity for candidate units
        m.NON_NEGATIVE_CAPACITY = Constraint(m.G_C, m.Y, rule=non_negative_capacity_rule)

        def cumulative_capacity_rule(_m, g, y):
            """Total installed capacity for each year of model horizon"""

            return m.a[g, y] - sum(m.x_c[g, j] for j in m.Y if j <= y) == 0

        # Cumulative capacity rule
        m.CUMULATIVE_CAPACITY = Constraint(m.G_C, m.Y, rule=cumulative_capacity_rule)

        def solar_build_limit_rule(_m, z, y):
            """Enforce solar build limits in each NEM zone"""

            # Solar generators belonging to zone 'z'
            gens = [g for g in m.G_C_SOLAR if g.split('-')[0] == z]

            if gens:
                return sum(m.a[g, y] for g in gens) - m.SOLAR_BUILD_LIMITS[z] <= 0
            else:
                return Constraint.Skip

        # Storage build limit constraint for each NEM zone
        m.SOLAR_BUILD_LIMIT_CONS = Constraint(m.Z, m.Y, rule=solar_build_limit_rule)

        def wind_build_limit_rule(_m, z, y):
            """Enforce wind build limits in each NEM zone"""

            # Wind generators belonging to zone 'z'
            gens = [g for g in m.G_C_WIND if g.split('-')[0] == z]

            if gens:
                return sum(m.a[g, y] for g in gens) - m.WIND_BUILD_LIMITS[z] <= 0
            else:
                return Constraint.Skip

        # Wind build limit constraint for each NEM zone
        m.WIND_BUILD_LIMIT_CONS = Constraint(m.Z, m.Y, rule=wind_build_limit_rule)

        def storage_build_limit_rule(_m, z, y):
            """Enforce storage build limits in each NEM zone"""

            # Storage generators belonging to zone 'z'
            gens = [g for g in m.G_C_STORAGE if g.split('-')[0] == z]

            if gens:
                return sum(m.a[g, y] for g in gens) - m.STORAGE_BUILD_LIMITS[z] <= 0
            else:
                return Constraint.Skip

        # Storage build limit constraint for each NEM zone
        m.STORAGE_BUILD_LIMIT_CONS = Constraint(m.Z, m.Y, rule=storage_build_limit_rule)

        def min_power_rule(_m, g, y, s, t):
            """Minimum power output"""

            return - m.p[g, y, s, t] + m.P_MIN[g] <= 0

        # Minimum power output
        m.MIN_POWER_CONS = Constraint(m.G.difference(m.G_STORAGE), m.Y, m.S, m.T, rule=min_power_rule)

        def max_power_existing_thermal_rule(_m, g, y, s, t):
            """Max power from existing thermal generators"""

            return m.p[g, y, s, t] - (m.P_MAX[g] * (1 - m.F[g, y])) <= 0

        # Max power from existing thermal units
        m.MAX_POWER_EXISTING_THERMAL = Constraint(m.G_E_THERM, m.Y, m.S, m.T, rule=max_power_existing_thermal_rule)

        def max_power_candidate_thermal_rule(_m, g, y, s, t):
            """Max power from existing thermal generators"""

            return m.p[g, y, s, t] - m.a[g, y] <= 0

        # Max power from candidate thermal units
        m.MAX_POWER_CANDIDATE_THERMAL = Constraint(m.G_C_THERM, m.Y, m.S, m.T, rule=max_power_candidate_thermal_rule)

        def max_power_existing_wind_rule(_m, g, y, s, t):
            """Max power from existing wind generators"""

            return m.p[g, y, s, t] - (m.Q_W[g, y, s, t] * m.P_MAX[g] * (1 - m.F[g, y])) <= 0

        # Max power from existing wind generators
        m.MAX_POWER_EXISTING_WIND = Constraint(m.G_E_WIND, m.Y, m.S, m.T, rule=max_power_existing_wind_rule)

        def max_power_candidate_wind_rule(_m, g, y, s, t):
            """Max power from candidate wind generators"""

            return m.p[g, y, s, t] - (m.Q_W[g, y, s, t] * m.a[g, y]) <= 0

        # Max power from candidate wind generators
        m.MAX_POWER_CANDIDATE_WIND = Constraint(m.G_C_WIND, m.Y, m.S, m.T, rule=max_power_candidate_wind_rule)

        def max_power_existing_solar_rule(_m, g, y, s, t):
            """Max power from existing solar generators"""

            return m.p[g, y, s, t] - (m.Q_S[g, y, s, t] * m.P_MAX[g] * (1 - m.F[g, y])) <= 0

        # Max power from existing solar generators
        m.MAX_POWER_EXISTING_SOLAR = Constraint(m.G_E_SOLAR, m.Y, m.S, m.T, rule=max_power_existing_solar_rule)

        def max_power_candidate_solar_rule(_m, g, y, s, t):
            """Max power from candidate solar generators"""

            return m.p[g, y, s, t] - (m.Q_S[g, y, s, t] * m.a[g, y]) <= 0

        # Max power from candidate wind generators
        m.MAX_POWER_CANDIDATE_SOLAR = Constraint(m.G_C_SOLAR, m.Y, m.S, m.T, rule=max_power_candidate_solar_rule)

        def max_power_hydro_rule(_m, g, y, s, t):
            """Max power from hydro units"""

            return m.p[g, y, s, t] - (m.P_H[g, y, s, t] * (1 - m.F[g, y])) <= 0

        # Max power from existing hydro units
        m.MAX_POWER_EXISTING_HYDRO = Constraint(m.G_E_HYDRO, m.Y, m.S, m.T, rule=max_power_hydro_rule)

        def min_power_in_storage_rule(_m, g, y, s, t):
            """Min charging power for storage units"""

            return - m.p_in[g, y, s, t] <= 0

        # Min charging power for storage units
        m.MIN_POWER_IN_STORAGE = Constraint(m.G_STORAGE, m.Y, m.S, m.T, rule=min_power_in_storage_rule)

        def min_power_out_storage_rule(_m, g, y, s, t):
            """Min discharging power for storage units"""

            return - m.p_out[g, y, s, t] <= 0

        # Min discharging power for storage units
        m.MIN_POWER_OUT_STORAGE = Constraint(m.G_STORAGE, m.Y, m.S, m.T, rule=min_power_out_storage_rule)

        def max_power_in_existing_storage_rule(_m, g, y, s, t):
            """Max charging power for existing storage units"""

            return m.p_in[g, y, s, t] - (m.P_IN_MAX[g] * (1 - m.F[g, y])) <= 0

        # Max charging power for existing storage units
        m.MAX_POWER_IN_EXISTING_STORAGE = Constraint(m.G_E_STORAGE, m.Y, m.S, m.T,
                                                     rule=max_power_in_existing_storage_rule)

        def max_power_in_candidate_storage_rule(_m, g, y, s, t):
            """Max charging power for candidate storage units"""

            return m.p_in[g, y, s, t] - m.a[g, y] <= 0

        # Max charging power for candidate storage units
        m.MAX_POWER_IN_CANDIDATE_STORAGE = Constraint(m.G_C_STORAGE, m.Y, m.S, m.T,
                                                      rule=max_power_in_candidate_storage_rule)

        def max_power_out_existing_storage_rule(_m, g, y, s, t):
            """Max discharging power for existing storage units"""

            return m.p_out[g, y, s, t] - (m.P_OUT_MAX[g] * (1 - m.F[g, y])) <= 0

        # Max discharging power for existing storage units
        m.MAX_POWER_OUT_EXISTING_STORAGE = Constraint(m.G_E_STORAGE, m.Y, m.S, m.T,
                                                      rule=max_power_out_existing_storage_rule)

        def max_power_out_candidate_storage_rule(_m, g, y, s, t):
            """Max discharging power for candidate storage units"""

            return m.p_out[g, y, s, t] - m.a[g, y] <= 0

        # Max discharging power for candidate storage units
        m.MAX_POWER_OUT_CANDIDATE_STORAGE = Constraint(m.G_C_STORAGE, m.Y, m.S, m.T,
                                                       rule=max_power_out_candidate_storage_rule)

        def min_energy_storage_rule(_m, g, y, s, t):
            """Min energy for storage units"""

            return - m.q[g, y, s, t] <= 0

        # Min storage unit energy level
        m.MIN_ENERGY_STORAGE = Constraint(m.G_STORAGE, m.Y, m.S, m.T, rule=min_energy_storage_rule)

        def max_energy_existing_storage_rule(_m, g, y, s, t):
            """Max energy for existing storage units"""

            return m.q[g, y, s, t] - m.Q_MAX <= 0

        # Max energy for existing storage units
        m.MAX_ENERGY_EXISTING_STORAGE = Constraint(m.G_E_STORAGE, m.Y, m.S, m.T, rule=max_energy_existing_storage_rule)

        def max_energy_candidate_storage_rule(_m, g, y, s, t):
            """Max energy for candidate storage units"""

            return m.q[g, y, s, t] - m.a[g, y] <= 0

        # Max energy for candidate storage units
        m.MAX_ENERGY_CANDIDATE_STORAGE = Constraint(m.G_C_STORAGE, m.Y, m.S, m.T,
                                                    rule=max_energy_candidate_storage_rule)

        def min_energy_interval_end_storage_rule(_m, g, y, s):
            """Min energy in storage unit at end of scenario interval"""

            return m.Q_END_MIN[g] - m.q[g, y, s, m.T.last()] <= 0

        # Min energy in storage unit at end of scenario
        m.MIN_ENERGY_INTERVAL_END_STORAGE = Constraint(m.G_STORAGE, m.Y, m.S, rule=min_energy_interval_end_storage_rule)

        def max_energy_interval_end_storage_rule(_m, g, y, s):
            """Max energy in storage unit at end of scenario interval"""

            return m.q[g, y, s, m.T.last()] - m.Q_END_MAX[g] <= 0

        # Min energy in storage unit at end of scenario
        m.MAX_ENERGY_INTERVAL_END_STORAGE = Constraint(m.G_STORAGE, m.Y, m.S, rule=max_energy_interval_end_storage_rule)

        def storage_energy_transition_rule(_m, g, y, s, t):
            """Energy transition rule for storage units"""

            if t == m.T.first():
                return (- m.q[g, y, s, t] + m.Q0[g, y, s] + (m.ETA[g] * m.p_in[g, y, s, t])
                        - ((1 / m.ETA[g]) * m.p_out[g, y, s, t])
                        == 0)

            else:
                return (- m.q[g, y, s, t] + m.q[g, y, s, t - 1] + (m.ETA[g] * m.p_in[g, y, s, t])
                        - ((1 / m.ETA[g]) * m.p_out[g, y, s, t])
                        == 0)

        # Energy transition rule for storage units
        m.ENERGY_TRANSITION_STORAGE = Constraint(m.G_STORAGE, m.Y, m.S, m.T, rule=storage_energy_transition_rule)

        def ramp_up_rule(_m, g, y, s, t):
            """Ramp up constraint"""

            if t == m.T.first():
                return m.p[g, y, s, t] - m.P0[g, y, s] - m.RR_UP[g] <= 0
            else:
                return m.p[g, y, s, t] - m.p[g, y, s, t - 1] - m.RR_UP[g] <= 0

        # Ramp-rate up constraint
        m.RAMP_UP = Constraint(m.G.difference(m.G_STORAGE), m.Y, m.S, m.T, rule=ramp_up_rule)

        def ramp_up_charging_rule(_m, g, y, s, t):
            """Ramp-rate up for storage charging power"""

            if t == m.T.first():
                return m.p_in[g, y, s, t] - m.P_IN_0[g, y, s] - m.RR_UP[g] <= 0
            else:
                return m.p_in[g, y, s, t] - m.p_in[g, y, s, t - 1] - m.RR_UP[g] <= 0

        # Ramp-rate up charging
        m.RAMP_UP_CHARGING = Constraint(m.G_STORAGE, m.Y, m.S, m.T, rule=ramp_up_charging_rule)

        def ramp_up_discharging_rule(_m, g, y, s, t):
            """Ramp-rate up for storage discharging power"""

            if t == m.T.first():
                return m.p_out[g, y, s, t] - m.P_OUT_0[g, y, s] - m.RR_UP[g] <= 0
            else:
                return m.p_out[g, y, s, t] - m.p_out[g, y, s, t - 1] - m.RR_UP[g] <= 0

        # Ramp-rate up charging
        m.RAMP_UP_DISCHARGING = Constraint(m.G_STORAGE, m.Y, m.S, m.T, rule=ramp_up_discharging_rule)

        def ramp_down_rule(_m, g, y, s, t):
            """Ramp down constraint"""

            if t == m.T.first():
                return - m.p[g, y, s, t] + m.P0[g, y, s] - m.RR_DOWN[g] <= 0
            else:
                return - m.p[g, y, s, t] + m.p[g, y, s, t - 1] - m.RR_DOWN[g] <= 0

        # Ramp-rate down constraint
        m.RAMP_DOWN = Constraint(m.G.difference(m.G_STORAGE), m.Y, m.S, m.T, rule=ramp_down_rule)

        def ramp_down_charging_rule(_m, g, y, s, t):
            """Ramp-rate down for storage charging power"""

            if t == m.T.first():
                return - m.p_in[g, y, s, t] + m.P_IN_0[g, y, s] - m.RR_DOWN[g] <= 0
            else:
                return - m.p_in[g, y, s, t] + m.p_in[g, y, s, t - 1] - m.RR_DOWN[g] <= 0

        # Ramp-rate down constraint
        m.RAMP_DOWN_CHARGING = Constraint(m.G_STORAGE, m.Y, m.S, m.T, rule=ramp_down_charging_rule)

        def ramp_down_discharging_rule(_m, g, y, s, t):
            """Ramp-rate down for storage charging power"""

            if t == m.T.first():
                return - m.p_out[g, y, s, t] + m.P_OUT_0[g, y, s] - m.RR_DOWN[g] <= 0
            else:
                return - m.p_out[g, y, s, t] + m.p_out[g, y, s, t - 1] - m.RR_DOWN[g] <= 0

        # Ramp-rate down constraint
        m.RAMP_DOWN_DISCHARGING = Constraint(m.G_STORAGE, m.Y, m.S, m.T, rule=ramp_down_discharging_rule)

        def non_negative_lost_load_rule(_m, z, y, s, t):
            """Non-negative lost load"""

            return - m.p_V[z, y, s, t] <= 0

        # Non-negative lost load
        m.NON_NEGATIVE_LOST_LOAD = Constraint(m.Z, m.Y, m.S, m.T, rule=non_negative_lost_load_rule)

        def min_powerflow_rule(_m, l, y, s, t):
            """Minimum powerflow over link connecting NEM zones"""

            return m.POWERFLOW_MIN[l] - m.p_L[l, y, s, t] <= 0

        # Minimum powerflow over links connecting NEM zones
        m.MIN_FLOW = Constraint(m.L, m.Y, m.S, m.T, rule=min_powerflow_rule)

        def max_powerflow_rule(_m, l, y, s, t):
            """Maximum powerflow over link connecting NEM zones"""

            return m.p_L[l, y, s, t] - m.POWERFLOW_MAX[l] <= 0

        # Minimum powerflow over links connecting NEM zones
        m.MAX_FLOW = Constraint(m.L, m.Y, m.S, m.T, rule=max_powerflow_rule)

        def power_balance_rule(_m, z, y, s, t):
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

            return (m.DEMAND[z, y, s, t] - sum(m.p[g, y, s, t] for g in generators)
                    + sum(m.INCIDENCE_MATRIX[l, z] * m.p_L[l, y, s, t] for l in m.L)
                    - sum(m.p_out[g, y, s, t] - m.p_in[g, y, s, t] for g in storage_units)
                    - m.p_V[z, y, s, t]
                    == 0)

        # Power balance constraint for each zone and time period
        m.POWER_BALANCE = Constraint(m.Z, m.Y, m.S, m.T, rule=power_balance_rule)

        def generator_energy_output_rule(_m, g, y, s, t):
            """Energy output from generators (excluding storage)"""

            if t == m.T.first():
                return m.e[g, y, s, t] - ((m.P0[g, y, s] + m.p[g, y, s, t]) / 2) == 0

            else:
                return m.e[g, y, s, t] - ((m.p[g, y, s, t - 1] + m.p[g, y, s, t]) / 2) == 0

        # Generator energy output
        m.GENERATOR_ENERGY_OUT = Constraint(m.G.difference(m.G_STORAGE), m.Y, m.S, m.T,
                                            rule=generator_energy_output_rule)

        def storage_energy_output_rule(_m, g, y, s, t):
            """Energy output from storage units"""

            if t == m.T.first():
                return m.e[g, y, s, t] - ((m.P_OUT_0[g, y, s] + m.p_out[g, y, s, t]) / 2) == 0
            else:
                return m.e[g, y, s, t] - ((m.p_out[g, y, s, t - 1] + m.p_out[g, y, s, t]) / 2) == 0

        # Storage energy output
        m.STORAGE_ENERGY_OUT = Constraint(m.G_STORAGE, m.Y, m.S, m.T, rule=storage_energy_output_rule)

        def lost_load_energy_rule(_m, z, y, s, t):
            """Lost load energy"""

            if t == m.T.first():
                return m.e_V[z, y, s, t] - ((m.P_V0[z, y, s] + m.p_V[z, y, s, t]) / 2) == 0
            else:
                return m.e_V[z, y, s, t] - ((m.p_V[z, y, s, t - 1] + m.p_V[z, y, s, t]) / 2) == 0

        # Lost load energy
        m.LOST_LOAD_ENERGY = Constraint(m.Z, m.Y, m.S, m.T, rule=lost_load_energy_rule)

        return m

    @staticmethod
    def define_objective(m):
        """Primal program objective"""

        # Minimise total present value for primal problem
        m.OBJECTIVE = Objective(expr=m.TPV, sense=minimize)

        return m

    def construct_model(self):
        """Construct primal program"""

        # Initialise model
        m = ConcreteModel()

        # Add common components
        m = self.components.define_sets(m)
        m = self.components.define_parameters(m)
        m = self.components.define_variables(m)

        # Primal problem variables
        m = self.define_variables(m)

        # Primal problem expressions
        m = self.define_expressions(m)

        # Primal problem constraints
        m = self.define_constraints(m)

        # Primal problem objective
        m = self.define_objective(m)

        return m

    def solve_model(self, m):
        """Solve model"""

        # Solve model
        solve_status = self.opt.solve(m, tee=True, options=self.solver_options, keepfiles=self.keepfiles)

        # Log infeasible constraints if they exist
        log_infeasible_constraints(m)

        return m, solve_status


class Dual:
    def __init__(self):
        self.data = ModelData()
        self.components = CommonComponents()

        # Solver options
        self.keepfiles = False
        self.solver_options = {}  # 'MIPGap': 0.0005
        self.opt = SolverFactory('cplex', solver_io='mps')

    def k(self, m, g):
        """Mapping generator to the NEM zone to which it belongs"""

        if g in m.G_E:
            return self.data.existing_units_dict[('PARAMETERS', 'NEM_ZONE')][g]

        elif g in m.G_C.difference(m.G_STORAGE):
            return self.data.candidate_units_dict[('PARAMETERS', 'ZONE')][g]

        elif g in m.G_STORAGE:
            return self.data.battery_properties_dict['NEM_ZONE'][g]

        else:
            raise Exception(f'Unexpected generator: {g}')

    @staticmethod
    def g(l):
        """Mapping link ID to link's 'from' node ID"""

        # Extract from node information for link ID
        from_node, _ = l.split('-')

        return from_node

    @staticmethod
    def h(l):
        """Mapping link ID to link's 'to' node ID"""

        # Extract from node information for link ID
        _, to_node = l.split('-')

        return to_node

    def define_variables(self, m):
        """Define dual problem variables"""

        # Non-negative candidate capacity
        m.mu_1 = Var(m.G_C, m.Y, within=NonNegativeReals, initialize=0)

        # Solar build limits
        m.mu_2 = Var(m.Z, m.Y, within=NonNegativeReals, initialize=0)

        # Wind build limits
        m.mu_3 = Var(m.Z, m.Y, within=NonNegativeReals, initialize=0)

        # Storage build limits
        m.mu_4 = Var(m.Z, m.Y, within=NonNegativeReals, initialize=0)

        # Cumulative candidate capacity
        m.nu_1 = Var(m.G_C, m.Y, initialize=0)

        # Min power output (all generators excluding storage units)
        m.sigma_1 = Var(m.G.difference(m.G_STORAGE), m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Max power output - existing thermal
        m.sigma_2 = Var(m.G_E_THERM, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Max power output - candidate thermal
        m.sigma_3 = Var(m.G_C_THERM, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Max power output - existing wind
        m.sigma_4 = Var(m.G_E_WIND, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Max power output - candidate wind
        m.sigma_5 = Var(m.G_C_WIND, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Max power output - existing solar
        m.sigma_6 = Var(m.G_E_SOLAR, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Max power output - candidate solar
        m.sigma_7 = Var(m.G_C_SOLAR, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Max power output - hydro
        m.sigma_8 = Var(m.G_E_HYDRO, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Min charging power - storage units
        m.sigma_9 = Var(m.G_STORAGE, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Min discharging power - storage_units
        m.sigma_10 = Var(m.G_STORAGE, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Max power output - existing storage
        m.sigma_11 = Var(m.G_E_STORAGE, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Max power output - candidate storage
        m.sigma_12 = Var(m.G_C_STORAGE, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Max power output - candidate storage
        m.sigma_14 = Var(m.G_C_STORAGE, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Min energy - storage units
        m.sigma_15 = Var(m.G_STORAGE, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Max energy - candidate storage
        m.sigma_17 = Var(m.G_C_STORAGE, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Min energy - interval end
        m.sigma_18 = Var(m.G_STORAGE, m.Y, m.S, within=NonNegativeReals, initialize=0)

        # Max energy - interval end
        m.sigma_19 = Var(m.G_STORAGE, m.Y, m.S, within=NonNegativeReals, initialize=0)

        # Ramp-rate up (all generators excluding storage)
        m.sigma_20 = Var(m.G.difference(m.G_STORAGE), m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Ramp-rate down (all generators excluding storage)
        m.sigma_21 = Var(m.G.difference(m.G_STORAGE), m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Non-negative lost load power
        m.sigma_22 = Var(m.Z, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Min powerflow
        m.sigma_23 = Var(m.L, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Max powerflow
        m.sigma_24 = Var(m.L, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Storage energy transition
        m.zeta_1 = Var(m.G_STORAGE, m.Y, m.S, m.T, initialize=0)

        # Energy output (all generators excluding storage)
        m.zeta_2 = Var(m.G.difference(m.G_STORAGE), m.Y, m.S, m.T, initialize=0)

        # Energy output (storage units)
        m.zeta_3 = Var(m.G_STORAGE, m.Y, m.S, m.T, initialize=0)

        # Energy - lost load
        m.zeta_4 = Var(m.Z, m.Y, m.S, m.T, initialize=0)

        # Power balance (locational marginal price)
        m.lamb = Var(m.Z, m.Y, m.S, m.T, initialize=0)

        # Fix values for policy variables
        m.baseline.fix(0)
        m.permit_price.fix(0)

        return m

    def define_expressions(self, m):
        """Define dual problem expressions"""

        def dual_objective_expression_rule(_m):
            """Expression for dual objective function"""

            # Build limits
            t_1 = sum(- (m.mu_2[z, y] * m.SOLAR_BUILD_LIMITS[z]) - (m.mu_3[z, y] * m.WIND_BUILD_LIMITS[z]) - (m.mu_4[z, y] * m.STORAGE_BUILD_LIMITS[z]) for z in m.Z for y in m.Y)

            # Min power output
            t_2 = sum(m.sigma_1[g, y, s, t] * m.P_MIN[g] for g in m.G.difference(m.G_STORAGE) for y in m.Y for s in m.S for t in m.T)

            # Max power - existing generators
            t_3 = sum(- m.sigma_2[g, y, s, t] * m.P_MAX[g] * (1 - m.F[g, y]) for g in m.G_E_THERM for y in m.Y for s in m.S for t in m.T)

            # Max power - existing wind
            t_4 = sum(- m.sigma_4[g, y, s, t] * m.Q_W[g, y, s, t] * m.P_MAX[g] * (1 - m.F[g, y]) for g in m.G_E_WIND for y in m.Y for s in m.S for t in m.T)

            # Max power - existing solar
            t_5 = sum(- m.sigma_6[g, y, s, t] * m.Q_S[g, y, s, t] * m.P_MAX[g] * (1 - m.F[g, y]) for g in m.G_E_SOLAR for y in m.Y for s in m.S for t in m.T)

            # Max power - existing hydro
            t_6 = sum(- m.sigma_8[g, y, s, t] * m.P_H[g, y, s, t] * (1 - m.F[g, y]) for g in m.G_E_HYDRO for y in m.Y for s in m.S for t in m.T)

            # Max charging power - existing storage
            t_7 = sum(- m.sigma_11[g, y, s, t] * m.P_IN_MAX[g] * (1 - m.F[g, y]) for g in m.G_E_STORAGE for y in m.Y for s in m.S for t in m.T)

            # Max discharging power - existing storage
            t_8 = sum(- m.sigma_13[g, y, s, t] * m.P_OUT_MAX[g] * (1 - m.F[g, y]) for g in m.G_E_STORAGE for y in m.Y for s in m.S for t in m.T)

            # Max energy - existing storage units
            t_9 = sum(- m.sigma_16[g, y, s, t] * m.Q_MAX[g] for g in m.G_E_STORAGE for y in m.Y for s in m.S for t in m.T)

            # Min energy - interval end
            t_10 = sum(m.sigma_18[g, y, s] * m.Q_END_MIN[g] for g in m.G_E_STORAGE for y in m.Y for s in m.S)

            # Max energy - interval end
            t_11 = sum(- m.sigma_19[g, y, s] * m.Q_END_MAX[g] for g in m.G_E_STORAGE for y in m.Y for s in m.S)

            # Ramp-up constraint
            t_12 = sum(- m.sigma_20[g, y, s, t] * m.RR_UP[g] for g in m.G.difference(m.G_STORAGE) for y in m.Y for s in m.S for t in m.T)

            # Initial power output
            t_13 = sum(- m.sigma_20[g, y, s, m.T.last()] * m.P0[g, y, s] for g in m.G.difference(m.G_STORAGE) for y in m.Y for s in m.S)

            # Ramp-down constraint
            t_14 = sum(- m.sigma_21[g, y, s, t] * m.RR_DOWN[g] for g in m.G.difference(m.G_STORAGE) for y in m.Y for s in m.S for t in m.T)

            # Initial power output
            t_15 = sum(m.sigma_21[g, y, s, m.T.last()] * m.P0[g, y, s] for g in m.G.difference(m.G_STORAGE) for y in m.Y for s in m.S)

            # Min powerflow
            t_16 = sum(m.sigma_23[l, y, s, t] * m.POWERFLOW_MIN[l] for l in m.L for y in m.Y for s in m.S for t in m.T)

            # Max powerflow
            t_17 = sum(- m.sigma_24[l, y, s, t] * m.POWERFLOW_MAX[l] for l in m.L for y in m.Y for s in m.S for t in m.T)

            # Demand
            t_18 = sum(m.lamb[z, y, s, t] * m.DEMAND[z, y, s, t] for z in m.Z for y in m.Y for s in m.S for t in m.T)

            # Initial storage unit energy
            t_19 = sum(m.zeta_1[g, y, s, m.T.first()] * m.Q0[g, y, s] for g in m.G_STORAGE for y in m.Y for s in m.S)

            # Initial generator power output
            t_20 = sum(- m.zeta_2[g, y, s, m.T.first()] * (m.P0[g, y, s] / 2) for g in m.G.difference(m.G_STORAGE) for y in m.Y for s in m.S)

            # Initial storage unit power output
            t_21 = sum(- m.zeta_3[g, y, s, m.T.first()] * (m.P_OUT_0[g, y, s] / 2) for g in m.G_STORAGE for y in m.Y for s in m.S)

            # Initial lost-load power
            t_22 = sum(- m.zeta_4[z, y, s, m.T.first()] * (m.P_V0[z, y, s] / 2) for z in m.Z for y in m.Y for s in m.S)

            return t_1 + t_2 + t_3 + t_4 + t_5 + t_6 + t_7 + t_8 + t_9 + t_10 + t_11 + t_12 + t_13 + t_14 + t_15 + t_16 + t_17 + t_18 + t_19 + t_20 + t_21 + t_22

        # Dual objective expression
        m.DUAL_OBJECTIVE_EXPRESSION = Expression(rule=dual_objective_expression_rule)

        return m

    def define_constraints(self, m):
        """Define dual problem constraints"""

        def investment_decisions_rule(_m, g, y):
            """Investment decision constraint (x_c)"""

            return ((m.DELTA[y] / m.INTEREST_RATE) * m.GAMMA[g] * m.I_C[g, y]) - m.mu_1[g, y] + sum(-m.nu_1[g, j] for j in m.Y if j >= y) == 0

        # Investment decision in each year
        m.INVESTMENT_DECISION = Constraint(m.G_C, m.Y, rule=investment_decisions_rule)

        def total_capacity_thermal_rule(_m, g, y):
            """Total thermal generator installed capacity"""

            if y != m.Y.last():
                return m.nu_1[g, y] + (m.DELTA[y] * m.C_FOM[g]) + sum(- m.sigma_3[g, y, s, t] for s in m.S for t in m.T) == 0
            else:
                return m.nu_1[g, m.Y.last()] + (m.DELTA[m.Y.last()] * (1 + (1 / m.INTEREST_RATE)) * m.C_FOM[g]) + sum(- m.sigma_3[g, m.Y.last(), s, t] for s in m.S for t in m.T) == 0

        # Total installed capacity
        m.TOTAL_THERMAL_CAPACITY = Constraint(m.G_C_THERM, m.Y, rule=total_capacity_thermal_rule)

        def total_capacity_solar_rule(_m, g, y):
            """Total solar generator installed capacity"""

            if y != m.Y.last():
                return m.mu_2[self.k(m, g), y] + m.nu_1[g, y] + (m.DELTA[y] * m.C_FOM[g]) + sum(- m.Q_S[g, y, s, t] * m.sigma_7[g, y, s, t] for s in m.S for t in m.T) == 0
            else:
                return m.mu_2[self.k(m, g), y] + m.nu_1[g, m.Y.last()] + (m.DELTA[m.Y.last()] * (1 + (1 / m.INTEREST_RATE)) * m.C_FOM[g]) + sum(- m.Q_S[g, y, s, t] * m.sigma_7[g, m.Y.last(), s, t] for s in m.S for t in m.T) == 0

        # Total installed capacity
        m.TOTAL_SOLAR_CAPACITY = Constraint(m.G_C_SOLAR, m.Y, rule=total_capacity_solar_rule)

        def total_capacity_wind_rule(_m, g, y):
            """Total wind generator installed capacity"""

            if y != m.Y.last():
                return m.mu_3[self.k(m, g), y] + m.nu_1[g, y] + (m.DELTA[y] * m.C_FOM[g]) + sum(- m.Q_W[g, y, s, t] * m.sigma_5[g, y, s, t] for s in m.S for t in m.T) == 0
            else:
                return m.mu_3[self.k(m, g), y] + m.nu_1[g, m.Y.last()] + (m.DELTA[m.Y.last()] * (1 + (1 / m.INTEREST_RATE)) * m.C_FOM[g]) + sum(- m.Q_W[g, y, s, t] * m.sigma_5[g, m.Y.last(), s, t] for s in m.S for t in m.T) == 0

        # Total installed capacity
        m.TOTAL_WIND_CAPACITY = Constraint(m.G_C_WIND, m.Y, rule=total_capacity_wind_rule)

        def total_capacity_storage_rule(_m, g, y):
            """Total storage installed capacity"""

            if y != m.Y.last():
                return m.mu_4[self.k(m, g), y] + m.nu_1[g, y] + (m.DELTA[y] * m.C_FOM[g]) + sum(- m.sigma_14[g, y, s, t] for s in m.S for t in m.T) == 0
            else:
                return m.mu_4[self.k(m, g), y] + m.nu_1[g, m.Y.last()] + (m.DELTA[m.Y.last()] * (1 + (1 / m.INTEREST_RATE)) * m.C_FOM[g]) + sum(- m.sigma_14[g, m.Y.last(), s, t] for s in m.S for t in m.T) == 0

        # Total installed capacity
        m.TOTAL_STORAGE_CAPACITY = Constraint(m.G_C_STORAGE, m.Y, rule=total_capacity_storage_rule)

        def power_output_existing_thermal_rule(_m, g, y, s, t):
            """Power output from existing thermal generators"""

            if t != m.T.last():
                return (- m.sigma_1[g, y, s, t] + m.sigma_2[g, y, s, t] + m.sigma_20[g, y, s, t]
                        - m.sigma_20[g, y, s, t+1] - m.sigma_21[g, y, s, t] + m.sigma_21[g, y, s, t+1]
                        - m.lamb[self.k(m, g), y, s, t]
                        - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        == 0)
            else:
                return (- m.sigma_1[g, y, s, t] + m.sigma_2[g, y, s, t] + m.sigma_20[g, y, s, t]
                        - m.sigma_21[g, y, s, t]
                        - m.lamb[self.k(m, g), y, s, t]
                        - (m.zeta_2[g, y, s, t] / 2)
                        == 0)

        # Total power output from thermal generators
        m.POWER_OUTPUT_EXISTING_THERMAL = Constraint(m.G_E_THERM, m.Y, m.S, m.T, rule=power_output_existing_thermal_rule)

        def power_output_candidate_thermal_rule(_m, g, y, s, t):
            """Power output from candidate thermal generators"""

            if t != m.T.last():
                return (- m.sigma_1[g, y, s, t] + m.sigma_3[g, y, s, t] + m.sigma_20[g, y, s, t]
                        - m.sigma_20[g, y, s, t+1] - m.sigma_21[g, y, s, t] + m.sigma_21[g, y, s, t+1]
                        - m.lamb[self.k(m, g), y, s, t]
                        - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        == 0)
            else:
                return (- m.sigma_1[g, y, s, t] + m.sigma_3[g, y, s, t] + m.sigma_20[g, y, s, t]
                        - m.sigma_21[g, y, s, t]
                        - m.lamb[self.k(m, g), y, s, t]
                        - (m.zeta_2[g, y, s, t] / 2)
                        == 0)

        # Total power output from candidate thermal generators
        m.POWER_OUTPUT_CANDIDATE_THERMAL = Constraint(m.G_C_THERM, m.Y, m.S, m.T, rule=power_output_candidate_thermal_rule)

        def power_output_existing_wind_rule(_m, g, y, s, t):
            """Power output from existing wind generators"""

            if t != m.T.last():
                return (- m.sigma_1[g, y, s, t] + m.sigma_4[g, y, s, t] + m.sigma_20[g, y, s, t]
                        - m.sigma_20[g, y, s, t+1] - m.sigma_21[g, y, s, t] + m.sigma_21[g, y, s, t+1]
                        - m.lamb[self.k(m, g), y, s, t]
                        - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        == 0)
            else:
                return (- m.sigma_1[g, y, s, t] + m.sigma_4[g, y, s, t] + m.sigma_20[g, y, s, t]
                        - m.sigma_21[g, y, s, t]
                        - m.lamb[self.k(m, g), y, s, t]
                        - (m.zeta_2[g, y, s, t] / 2)
                        == 0)

        # Total power output from existing wind generators
        m.POWER_OUTPUT_EXISTING_WIND = Constraint(m.G_E_WIND, m.Y, m.S, m.T, rule=power_output_existing_wind_rule)

        def power_output_candidate_wind_rule(_m, g, y, s, t):
            """Power output from candidate wind generators"""

            if t != m.T.last():
                return (- m.sigma_1[g, y, s, t] + m.sigma_5[g, y, s, t] + m.sigma_20[g, y, s, t]
                        - m.sigma_20[g, y, s, t+1] - m.sigma_21[g, y, s, t] + m.sigma_21[g, y, s, t+1]
                        - m.lamb[self.k(m, g), y, s, t]
                        - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        == 0)
            else:
                return (- m.sigma_1[g, y, s, t] + m.sigma_5[g, y, s, t] + m.sigma_20[g, y, s, t]
                        - m.sigma_21[g, y, s, t]
                        - m.lamb[self.k(m, g), y, s, t]
                        - (m.zeta_2[g, y, s, t] / 2)
                        == 0)

        # Total power output from candidate wind generators
        m.POWER_OUTPUT_CANDIDATE_WIND = Constraint(m.G_C_WIND, m.Y, m.S, m.T, rule=power_output_candidate_wind_rule)

        def power_output_existing_solar_rule(_m, g, y, s, t):
            """Power output from existing solar generators"""

            if t != m.T.last():
                return (- m.sigma_1[g, y, s, t] + m.sigma_6[g, y, s, t] + m.sigma_20[g, y, s, t]
                        - m.sigma_20[g, y, s, t+1] - m.sigma_21[g, y, s, t] + m.sigma_21[g, y, s, t+1]
                        - m.lamb[self.k(m, g), y, s, t]
                        - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        == 0)

            else:
                return (- m.sigma_1[g, y, s, t] + m.sigma_6[g, y, s, t] + m.sigma_20[g, y, s, t]
                        - m.sigma_21[g, y, s, t]
                        - m.lamb[self.k(m, g), y, s, t]
                        - (m.zeta_2[g, y, s, t] / 2)
                        == 0)

        # Total power output from existing solar generators
        m.POWER_OUTPUT_EXISTING_SOLAR = Constraint(m.G_E_SOLAR, m.Y, m.S, m.T, rule=power_output_existing_solar_rule)

        def power_output_candidate_solar_rule(_m, g, y, s, t):
            """Power output from candidate solar generators"""

            if t != m.T.last():
                return (- m.sigma_1[g, y, s, t] + m.sigma_7[g, y, s, t] + m.sigma_20[g, y, s, t]
                        - m.sigma_20[g, y, s, t+1] - m.sigma_21[g, y, s, t] + m.sigma_21[g, y, s, t+1]
                        - m.lamb[self.k(m, g), y, s, t]
                        - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        == 0)

            else:
                return (- m.sigma_1[g, y, s, t] + m.sigma_7[g, y, s, t] + m.sigma_20[g, y, s, t]
                        - m.sigma_21[g, y, s, t]
                        - m.lamb[self.k(m, g), y, s, t]
                        - (m.zeta_2[g, y, s, t] / 2)
                        == 0)

        # Total power output from candidate solar generators
        m.POWER_OUTPUT_CANDIDATE_SOLAR = Constraint(m.G_C_SOLAR, m.Y, m.S, m.T, rule=power_output_candidate_solar_rule)

        def power_output_hydro_rule(_m, g, y, s, t):
            """Power output from hydro generators"""

            if t != m.T.last():
                return (- m.sigma_1[g, y, s, t] + m.sigma_8[g, y, s, t] + m.sigma_20[g, y, s, t]
                        - m.sigma_20[g, y, s, t+1] - m.sigma_21[g, y, s, t] + m.sigma_21[g, y, s, t+1]
                        - m.lamb[self.k(m, g), y, s, t]
                        - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        == 0)
            else:
                return (- m.sigma_1[g, y, s, t] + m.sigma_8[g, y, s, t] + m.sigma_20[g, y, s, t]
                        - m.sigma_21[g, y, s, t]
                        - m.lamb[self.k(m, g), y, s, t]
                        - (m.zeta_2[g, y, s, t] / 2)
                        == 0)

        # Total power output from hydro generators
        m.POWER_OUTPUT_HYDRO = Constraint(m.G_E_HYDRO, m.Y, m.S, m.T, rule=power_output_hydro_rule)

        def charging_power_existing_storage_rule(_m, g, y, s, t):
            """Charging power for existing storage units"""

            return - m.sigma_9[g, y, s, t] + m.sigma_11[g, y, s, t] + m.lamb[self.k(m, g), y, s, t] + (m.ETA[g] * m.zeta_1[g, y, s, t]) == 0

        # Existing storage unit charging power
        m.CHARGING_POWER_EXISTING_STORAGE = Constraint(m.G_E_STORAGE, m.Y, m.S, m.T, rule=charging_power_existing_storage_rule)

        def charging_power_candidate_storage_rule(_m, g, y, s, t):
            """Charging power for candidate storage units"""

            return - m.sigma_9[g, y, s, t] + m.sigma_12[g, y, s, t] + m.lamb[self.k(m, g), y, s, t] + (m.ETA[g] * m.zeta_1[g, y, s, t]) == 0

        # Existing storage unit charging power
        m.CHARGING_POWER_CANDIDATE_STORAGE = Constraint(m.G_C_STORAGE, m.Y, m.S, m.T, rule=charging_power_candidate_storage_rule)

        def discharging_power_existing_storage_rule(_m, g, y, s, t):
            """Discharging power for existing storage units"""

            if t != m.T.last():
                return (- m.sigma_10[g, y, s, t] + m.sigma_13[g, y, s, t] - m.lamb[self.k(m, g), y, s, t]
                        - ((1 / m.ETA[g]) * m.zeta_1[g, y, s, t])
                        - ((m.zeta_3[g, y, s, t] + m.zeta_3[g, y, s, t+1]) / 2)
                        == 0)
            else:
                return (- m.sigma_10[g, y, s, t] + m.sigma_13[g, y, s, t] - m.lamb[self.k(m, g), y, s, t]
                        - ((1 / m.ETA[g]) * m.zeta_1[g, y, s, t])
                        - (m.zeta_3[g, y, s, t] / 2)
                        == 0)

        # Existing storage unit discharging power
        m.DISCHARGING_POWER_EXISTING_STORAGE = Constraint(m.G_E_STORAGE, m.Y, m.S, m.T, rule=discharging_power_existing_storage_rule)

        def discharging_power_candidate_storage_rule(_m, g, y, s, t):
            """Discharging power for candidate storage units"""

            if t != m.T.last():
                return (- m.sigma_10[g, y, s, t] + m.sigma_14[g, y, s, t] - m.lamb[self.k(m, g), y, s, t]
                        - ((1 / m.ETA[g]) * m.zeta_1[g, y, s, t])
                        - ((m.zeta_3[g, y, s, t] + m.zeta_3[g, y, s, t+1]) / 2)
                        == 0)
            else:
                return (- m.sigma_10[g, y, s, t] + m.sigma_14[g, y, s, t] - m.lamb[self.k(m, g), y, s, t]
                        - ((1 / m.ETA[g]) * m.zeta_1[g, y, s, t])
                        - (m.zeta_3[g, y, s, t] / 2)
                        == 0)

        # Candidate storage unit discharging power
        m.DISCHARGING_POWER_CANDIDATE_STORAGE = Constraint(m.G_C_STORAGE, m.Y, m.S, m.T, rule=discharging_power_candidate_storage_rule)

        def energy_existing_storage_unit(_m, g, y, s, t):
            """Storage unit energy"""

            if t != m.T.last():
                return - m.sigma_15[g, y, s, t] + m.sigma_16[g, y, s, t] - m.zeta_1[g, y, s, t] + m.zeta_1[g, y, s, t+1] == 0

            else:
                return - m.sigma_15[g, y, s, t] + m.sigma_16[g, y, s, t] - m.zeta_1[g, y, s, t] - m.sigma_18[g, y, s] + m.sigma_19[g, y, s] == 0

        # Existing storage unit energy
        m.ENERGY_EXISTING_STORAGE = Constraint(m.G_E_STORAGE, m.Y, m.S, m.T, rule=energy_existing_storage_unit)

        def energy_candidate_storage_unit(_m, g, y, s, t):
            """Storage unit energy"""

            if t != m.T.last():
                return - m.sigma_15[g, y, s, t] + m.sigma_17[g, y, s, t] - m.zeta_1[g, y, s, t] + m.zeta_1[g, y, s, t+1] == 0

            else:
                return - m.sigma_15[g, y, s, t] + m.sigma_17[g, y, s, t] - m.zeta_1[g, y, s, t] - m.sigma_18[g, y, s] + m.sigma_19[g, y, s] == 0

        # Existing storage unit energy
        m.ENERGY_CANDIDATE_STORAGE = Constraint(m.G_C_STORAGE, m.Y, m.S, m.T, rule=energy_candidate_storage_unit)

        def energy_output_existing_thermal_rule(_m, g, y, s, t):
            """Existing thermal generator energy output"""

            if y != m.Y.last():
                return m.DELTA[y] * m.RHO[y, s] * (m.C_MC[g, y] + ((m.EMISSIONS_RATE[g] - m.baseline[y]) * m.permit_price[y])) + m.zeta_2[g, y, s, t] == 0
            else:
                return m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * (m.C_MC[g, y] + ((m.EMISSIONS_RATE[g] - m.baseline[y]) * m.permit_price[y])) + m.zeta_2[g, y, s, t] == 0

        # Existing thermal generator energy output
        m.ENERGY_EXISTING_THERMAL = Constraint(m.G_E_THERM, m.Y, m.S, m.T, rule=energy_output_existing_thermal_rule)

        def energy_output_candidate_thermal_rule(_m, g, y, s, t):
            """Candidate thermal generator energy output"""

            if y != m.Y.last():
                return (m.DELTA[y] * m.RHO[y, s] * (m.C_MC[g, y] + ((m.EMISSIONS_RATE[g] - m.baseline[y]) * m.permit_price[y]))) + m.zeta_2[g, y, s, t] == 0
            else:
                return (m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * (m.C_MC[g, y] + ((m.EMISSIONS_RATE[g] - m.baseline[y]) * m.permit_price[y]))) + m.zeta_2[g, y, s, t] == 0

        # Candidate thermal generator energy output
        m.ENERGY_CANDIDATE_THERMAL = Constraint(m.G_C_THERM, m.Y, m.S, m.T, rule=energy_output_candidate_thermal_rule)

        def energy_output_existing_wind_rule(_m, g, y, s, t):
            """Existing wind generator energy output"""

            if y != m.Y.last():
                return (m.DELTA[y] * m.RHO[y, s] * m.C_MC[g, y]) + m.zeta_2[g, y, s, t] == 0
            else:
                return (m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * m.C_MC[g, y]) + m.zeta_2[g, y, s, t] == 0

        # Existing wind generator energy output
        m.ENERGY_EXISTING_WIND = Constraint(m.G_E_WIND, m.Y, m.S, m.T, rule=energy_output_existing_wind_rule)

        def energy_output_candidate_wind_rule(_m, g, y, s, t):
            """Candidate wind generator energy output"""

            if y != m.Y.last():
                return (m.DELTA[y] * m.RHO[y, s] * (m.C_MC[g, y] + ((m.EMISSIONS_RATE[g] - m.baseline[y]) * m.permit_price[y]))) + m.zeta_2[g, y, s, t] == 0
            else:
                return (m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * (m.C_MC[g, y] + ((m.EMISSIONS_RATE[g] - m.baseline[y]) * m.permit_price[y]))) + m.zeta_2[g, y, s, t] == 0

        # Candidate wind generator energy output
        m.ENERGY_CANDIDATE_WIND = Constraint(m.G_C_WIND, m.Y, m.S, m.T, rule=energy_output_candidate_wind_rule)

        def energy_output_existing_solar_rule(_m, g, y, s, t):
            """Existing solar generator energy output"""

            if y != m.Y.last():
                return (m.DELTA[y] * m.RHO[y, s] * m.C_MC[g, y]) + m.zeta_2[g, y, s, t] == 0
            else:
                return (m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * m.C_MC[g, y]) + m.zeta_2[g, y, s, t] == 0

        # Existing wind generator energy output
        m.ENERGY_EXISTING_SOLAR = Constraint(m.G_E_SOLAR, m.Y, m.S, m.T, rule=energy_output_existing_solar_rule)

        def energy_output_candidate_solar_rule(_m, g, y, s, t):
            """Candidate solar generator energy output"""

            if y != m.Y.last():
                return (m.DELTA[y] * m.RHO[y, s] * (m.C_MC[g, y] + ((m.EMISSIONS_RATE[g] - m.baseline[y]) * m.permit_price[y]))) + m.zeta_2[g, y, s, t] == 0
            else:
                return (m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * (m.C_MC[g, y] + ((m.EMISSIONS_RATE[g] - m.baseline[y]) * m.permit_price[y]))) + m.zeta_2[g, y, s, t] == 0

        # Candidate wind generator energy output
        m.ENERGY_CANDIDATE_SOLAR = Constraint(m.G_C_SOLAR, m.Y, m.S, m.T, rule=energy_output_candidate_solar_rule)

        def energy_output_existing_storage_rule(_m, g, y, s, t):
            """Energy output for existing storage units"""

            if y != m.Y.last():
                return (m.DELTA[y] * m.RHO[y, s] * m.C_MC[g, y]) + m.zeta_3[g, y, s, t] == 0
            else:
                return (m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * m.C_MC[g, y]) + m.zeta_3[g, y, s, t] == 0

        # Existing storage unit energy output
        m.ENERGY_OUTPUT_EXISTING_STORAGE = Constraint(m.G_E_STORAGE, m.Y, m.S, m.T, rule=energy_output_existing_storage_rule)

        def energy_output_candidate_storage_rule(_m, g, y, s, t):
            """Energy output for candidate storage units"""

            if y != m.Y.last():
                return (m.DELTA[y] * m.RHO[y, s] * m.C_MC[g, y]) + m.zeta_3[g, y, s, t] == 0
            else:
                return (m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * m.C_MC[g, y]) + m.zeta_3[g, y, s, t] == 0

        # Candidate storage unit energy output
        m.ENERGY_OUTPUT_CANDIDATE_STORAGE = Constraint(m.G_C_STORAGE, m.Y, m.S, m.T, rule=energy_output_candidate_storage_rule)

        def energy_output_hydro_rule(_m, g, y, s, t):
            """Energy output for hydro units"""

            if y != m.Y.last():
                return (m.DELTA[y] * m.RHO[y, s] * m.C_MC[g, y]) + m.zeta_2[g, y, s, t] == 0
            else:
                return (m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * m.C_MC[g, y]) + m.zeta_2[g, y, s, t] == 0

        # Energy output from hydro units
        m.ENERGY_HYDRO = Constraint(m.G_E_HYDRO, m.Y, m.S, m.T, rule=energy_output_hydro_rule)

        def load_shedding_power_rule(_m, z, y, s, t):
            """Load shedding power"""

            return - m.sigma_22[z, y, s, t] - m.lamb[z, y, s, t] == 0

        # Load shedding power
        m.LOAD_SHEDDING_POWER = Constraint(m.Z, m.Y, m.S, m.T, rule=load_shedding_power_rule)

        def load_shedding_energy_rule(_m, z, y, s, t):
            """Load shedding energy"""

            if y != m.Y.last():
                return (m.DELTA[y] * m.RHO[y, s] * m.C_L) + m.zeta_4[z, y, s, t] == 0
            else:
                return (m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * m.C_L) + m.zeta_4[z, y, s, t] == 0

        # Load shedding energy
        m.LOAD_SHEDDING_ENERGY = Constraint(m.Z, m.Y, m.S, m.T, rule=load_shedding_energy_rule)

        def powerflow_rule(_m, l, y, s, t):
            """Powerflow between adjacent NEM zones"""

            return (- m.sigma_23[l, y, s, t] + m.sigma_24[l, y, s, t]
                    + (m.INCIDENCE_MATRIX[l, self.g(l)] * m.lamb[self.g(l), y, s, t])
                    + (m.INCIDENCE_MATRIX[l, self.h(l)] * m.lamb[self.h(l), y, s, t])
                    == 0)

        # Powerflow
        m.POWERFLOW = Constraint(m.L, m.Y, m.S, m.T, rule=powerflow_rule)

        return m

    def define_objective(self, m):
        """Define dual problem objective"""

        # Dual objective function
        m.OBJECTIVE = Objective(expr=m.DUAL_OBJECTIVE_EXPRESSION, sense=maximize)

        return m

    def construct_model(self):
        """Construct dual program"""

        # Initialise model
        m = ConcreteModel()

        # Add common components
        m = self.components.define_sets(m)
        m = self.components.define_parameters(m)
        m = self.components.define_variables(m)

        # Primal problem variables
        m = self.define_variables(m)

        # Primal problem expressions
        m = self.define_expressions(m)

        # Primal problem constraints
        m = self.define_constraints(m)

        # Primal problem objective
        m = self.define_objective(m)

        return m

    def solve_model(self, m):
        """Solve model"""

        # Solve model
        solve_status = self.opt.solve(m, tee=True, options=self.solver_options, keepfiles=self.keepfiles)

        # Log infeasible constraints if they exist
        log_infeasible_constraints(m)

        return m, solve_status


if __name__ == '__main__':
    primal = Primal()
    model = primal.construct_model()
    model, status = primal.solve_model(model)

    # dual = Dual()
    # model = dual.construct_model()
    #
    # model, status = dual.solve_model(model)

