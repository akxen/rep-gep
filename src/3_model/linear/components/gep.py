"""Classes used to define mathematical program with primal and dual constraints for generation expansion planning"""

import os
import sys
import copy
import pickle
import logging

# os.environ['TMPDIR'] = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'base'))

from pyomo.environ import *
from pyomo.util.infeasible import log_infeasible_constraints

from base.data import ModelData
from base.components import CommonComponents


class Primal:
    def __init__(self, final_year, scenarios_per_year):
        self.data = ModelData()
        self.components = CommonComponents(final_year, scenarios_per_year)

        # Solver options
        self.keepfiles = False
        self.solver_options = {}  # 'MIPGap': 0.0005
        self.opt = SolverFactory('cplex', solver_io='mps')

    @staticmethod
    def define_variables(m):
        """Primal problem variables"""

        # Investment capacity in each year
        m.x_c = Var(m.G_C, m.Y, initialize=0)

        # # Cumulative installed capacity
        # m.a = Var(m.G_C, m.Y, initialize=0)

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

        # # Energy output
        # m.e = Var(m.G, m.Y, m.S, m.T, initialize=0)
        #
        # # Lost load energy
        # m.e_V = Var(m.Z, m.Y, m.S, m.T, initialize=0)

        return m

    @staticmethod
    def define_expressions(m):
        """Primal problem expressions"""

        def cumulative_capacity_rule(_m, g, y):
            """Total candidate capacity available in each year"""

            return sum(m.x_c[g, j] for j in m.Y if j <= y)

        # Cumulative capacity available in each y ear
        m.CUMULATIVE_CAPACITY = Expression(m.G_C, m.Y, rule=cumulative_capacity_rule)

        def yearly_investment_cost_rule(_m, y):
            """Yearly investment cost"""
            return (1 / m.INTEREST_RATE) * sum(m.GAMMA[g] * m.I_C[g, y] * m.x_c[g, y] for g in m.G_C)

        # Yearly investment cost
        m.INV = Expression(m.Y, rule=yearly_investment_cost_rule)

        def yearly_fom_cost_rule(_m, y):
            """Yearly fixed operations and maintenance cost"""

            # # FOM cost for existing units
            # existing = sum(m.C_FOM[g] * (m.P_MAX[g] * (1 - m.F[g, y])) for g in m.G_E)

            # FOM cost for candidate units
            candidate = sum(m.C_FOM[g] * sum(m.x_c[g, j] for j in m.Y if j <= y) for g in m.G_C)

            return candidate

        # Yearly FOM cost
        m.FOM = Expression(m.Y, rule=yearly_fom_cost_rule)

        def thermal_operating_costs_rule(_m, y, s):
            """Cost to operating thermal units for a given scenario"""

            return sum((m.C_MC[g, y] + ((m.EMISSIONS_RATE[g] - m.baseline[y]) * m.permit_price[y])) * m.p[g, y, s, t]
                       for g in m.G_THERM for t in m.T)

        # Operating costs for a given scenario - thermal units
        m.OP_T = Expression(m.Y, m.S, rule=thermal_operating_costs_rule)

        def hydro_operating_costs_rule(_m, y, s):
            """Costs to operate hydro units for a given scenario"""

            return sum(m.C_MC[g, y] * m.p[g, y, s, t] for g in m.G_E_HYDRO for t in m.T)

        # Operating costs for a given scenario - hydro units
        m.OP_H = Expression(m.Y, m.S, rule=hydro_operating_costs_rule)

        def wind_operating_costs_rule(_m, y, s):
            """Cost to operate wind units for a given scenario"""

            # Cost for existing wind units
            existing = sum(m.C_MC[g, y] * m.p[g, y, s, t] for g in m.G_E_WIND for t in m.T)

            # Cost for candidate wind units
            candidate = sum((m.C_MC[g, y] - (m.baseline[y] * m.permit_price[y])) * m.p[g, y, s, t]
                            for g in m.G_C_WIND for t in m.T)

            return existing + candidate

        # Operating costs for a given scenario - wind units
        m.OP_W = Expression(m.Y, m.S, rule=wind_operating_costs_rule)

        def solar_operating_costs_rule(_m, y, s):
            """Cost to operate solar units for a given scenario"""

            # Cost for existing solar units
            existing = sum(m.C_MC[g, y] * m.p[g, y, s, t] for g in m.G_E_SOLAR for t in m.T)

            # Cost for candidate solar units
            candidate = sum((m.C_MC[g, y] - (m.baseline[y] * m.permit_price[y])) * m.p[g, y, s, t]
                            for g in m.G_C_SOLAR for t in m.T)

            return existing + candidate

        # Operating costs for a given scenario - solar units
        m.OP_S = Expression(m.Y, m.S, rule=solar_operating_costs_rule)

        def storage_operating_costs_rule(_m, y, s):
            """Cost to operate storage units for a given scenario"""

            return sum(m.C_MC[g, y] * m.p_out[g, y, s, t] for g in m.G_STORAGE for t in m.T)

        # Operating costs for a given scenario - storage units
        m.OP_Q = Expression(m.Y, m.S, rule=storage_operating_costs_rule)

        def lost_load_value_rule(_m, y, s):
            """Value of lost load in a given scenario"""

            return sum(m.C_L[z] * m.p_V[z, y, s, t] for z in m.Z for t in m.T)

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
            # return (m.DELTA[m.Y.last()] / m.INTEREST_RATE) * m.FOM[m.Y.last()]

        # End of horizon cost
        m.EOH = Expression(rule=end_of_horizon_cost_rule)

        # Total present value
        def total_present_value_rule(_m):
            """Total present value - investment + operating costs + end of horizon cost"""

            return sum(m.DELTA[y] * (m.INV[y] + m.FOM[y] + m.OP[y]) for y in m.Y) + m.EOH

        # Total present value
        m.TPV = Expression(rule=total_present_value_rule)

        def scenario_emissions_rule(_m, y, s):
            """Total emissions for a given scenario"""

            return m.RHO[y, s] * sum(m.p[g, y, s, t] * m.EMISSIONS_RATE[g] for g in m.G_THERM for t in m.T)

        # Scenario emissions
        m.SCENARIO_EMISSIONS = Expression(m.Y, m.S, rule=scenario_emissions_rule)

        def year_emissions_rule(_m, y):
            """Total emissions for a given year"""

            return sum(m.SCENARIO_EMISSIONS[y, s] for s in m.S)

        # Year emissions
        m.YEAR_EMISSIONS = Expression(m.Y, rule=year_emissions_rule)

        def scenario_scheme_revenue_rule(_m, y, s):
            """Total scheme revenue for a given scenario"""

            # Net revenue obtained from thermal generators
            thermal = sum(
                m.p[g, y, s, t] * (m.EMISSIONS_RATE[g] - m.baseline[y]) * m.permit_price[y] for g in m.G_THERM for t in
                m.T)

            # Net revenue obtained from candidate renewable generators (existing renewables considered ineligible)
            renewables = sum(
                m.p[g, y, s, t] * (- m.baseline[y] * m.permit_price[y]) for g in m.G_C_WIND.union(m.G_C_SOLAR) for t in
                m.T)

            return m.RHO[y, s] * (thermal + renewables)

        # Scenario scheme revenue
        m.SCENARIO_SCHEME_REVENUE = Expression(m.Y, m.S, rule=scenario_scheme_revenue_rule)

        def scenario_demand_rule(_m, y, s):
            """Total demand for a given scenario (MWh)"""

            return sum(m.RHO[y, s] * m.DEMAND[z, y, s, t] for z in m.Z for t in m.T)

        # Scenario demand
        m.SCENARIO_DEMAND = Expression(m.Y, m.S, rule=scenario_demand_rule)

        def year_demand_rule(_m, y):
            """Total demand for a given year (MWh)"""

            return sum(m.SCENARIO_DEMAND[y, s] for s in m.S)

        # Total demand for a given year
        m.YEAR_DEMAND = Expression(m.Y, rule=year_demand_rule)

        def year_emissions_intensity_rule(_m, y):
            """Emissions intensity for a given year"""

            return m.YEAR_EMISSIONS[y] / m.YEAR_DEMAND[y]

        # Emissions intensity for a given year
        m.YEAR_EMISSIONS_INTENSITY = Expression(m.Y, rule=year_emissions_intensity_rule)

        def year_scheme_revenue_rule(_m, y):
            """Total scheme revenue for a given year"""

            return sum(m.SCENARIO_SCHEME_REVENUE[y, s] for s in m.S)

        # Year scheme revenue
        m.YEAR_SCHEME_REVENUE = Expression(m.Y, rule=year_scheme_revenue_rule)

        def total_scheme_revenue_rule(_m):
            """Total scheme revenue over model horizon"""

            return sum(m.YEAR_SCHEME_REVENUE[y] for y in m.Y)

        # Total scheme revenue
        m.TOTAL_SCHEME_REVENUE = Expression(rule=total_scheme_revenue_rule)

        return m

    def define_constraints(self, m):
        """Primal problem constraints"""

        def non_negative_capacity_rule(_m, g, y):
            """Non-negative capacity for candidate units"""

            return - m.x_c[g, y] <= 0

        # Non-negative capacity for candidate units
        m.NON_NEGATIVE_CAPACITY = Constraint(m.G_C, m.Y, rule=non_negative_capacity_rule)

        # def cumulative_capacity_rule(_m, g, y):
        #     """Total installed capacity for each year of model horizon"""
        #
        #     return m.a[g, y] - sum(m.x_c[g, j] for j in m.Y if j <= y) == 0
        #
        # # Cumulative capacity rule
        # m.CUMULATIVE_CAPACITY = Constraint(m.G_C, m.Y, rule=cumulative_capacity_rule)

        def solar_build_limit_rule(_m, z, y):
            """Enforce solar build limits in each NEM zone"""

            # Solar generators belonging to zone 'z'
            gens = [g for g in m.G_C_SOLAR if g.split('-')[0] == z]

            if gens:
                return sum(m.x_c[g, j] for g in gens for j in m.Y if j <= y) - m.SOLAR_BUILD_LIMITS[z] <= 0
            else:
                return Constraint.Skip

        # Storage build limit constraint for each NEM zone
        m.SOLAR_BUILD_LIMIT_CONS = Constraint(m.Z, m.Y, rule=solar_build_limit_rule)

        def wind_build_limit_rule(_m, z, y):
            """Enforce wind build limits in each NEM zone"""

            # Wind generators belonging to zone 'z'
            gens = [g for g in m.G_C_WIND if g.split('-')[0] == z]

            if gens:
                return sum(m.x_c[g, j] for g in gens for j in m.Y if j <= y) - m.WIND_BUILD_LIMITS[z] <= 0
            else:
                return Constraint.Skip

        # Wind build limit constraint for each NEM zone
        m.WIND_BUILD_LIMIT_CONS = Constraint(m.Z, m.Y, rule=wind_build_limit_rule)

        def storage_build_limit_rule(_m, z, y):
            """Enforce storage build limits in each NEM zone"""

            # Storage generators belonging to zone 'z'
            gens = [g for g in m.G_C_STORAGE if g.split('-')[0] == z]

            if gens:
                return sum(m.x_c[g, j] for g in gens for j in m.Y if j <= y) - m.STORAGE_BUILD_LIMITS[z] <= 0
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

            return m.p[g, y, s, t] - sum(m.x_c[g, j] for j in m.Y if j <= y) <= 0

        # Max power from candidate thermal units
        m.MAX_POWER_CANDIDATE_THERMAL = Constraint(m.G_C_THERM, m.Y, m.S, m.T, rule=max_power_candidate_thermal_rule)

        def max_power_existing_wind_rule(_m, g, y, s, t):
            """Max power from existing wind generators"""

            return m.p[g, y, s, t] - (m.Q_W[g, y, s, t] * m.P_MAX[g] * (1 - m.F[g, y])) <= 0

        # Max power from existing wind generators
        m.MAX_POWER_EXISTING_WIND = Constraint(m.G_E_WIND, m.Y, m.S, m.T, rule=max_power_existing_wind_rule)

        def max_power_candidate_wind_rule(_m, g, y, s, t):
            """Max power from candidate wind generators"""

            return m.p[g, y, s, t] - (m.Q_W[g, y, s, t] * sum(m.x_c[g, j] for j in m.Y if j <= y)) <= 0

        # Max power from candidate wind generators
        m.MAX_POWER_CANDIDATE_WIND = Constraint(m.G_C_WIND, m.Y, m.S, m.T, rule=max_power_candidate_wind_rule)

        def max_power_existing_solar_rule(_m, g, y, s, t):
            """Max power from existing solar generators"""

            return m.p[g, y, s, t] - (m.Q_S[g, y, s, t] * m.P_MAX[g] * (1 - m.F[g, y])) <= 0

        # Max power from existing solar generators
        m.MAX_POWER_EXISTING_SOLAR = Constraint(m.G_E_SOLAR, m.Y, m.S, m.T, rule=max_power_existing_solar_rule)

        def max_power_candidate_solar_rule(_m, g, y, s, t):
            """Max power from candidate solar generators"""

            return m.p[g, y, s, t] - (m.Q_S[g, y, s, t] * sum(m.x_c[g, j] for j in m.Y if j <= y)) <= 0

        # Max power from candidate solar generators
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

            return m.p_in[g, y, s, t] - sum(m.x_c[g, j] for j in m.Y if j <= y) <= 0

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

            return m.p_out[g, y, s, t] - sum(m.x_c[g, j] for j in m.Y if j <= y) <= 0

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

            return m.q[g, y, s, t] - sum(m.x_c[g, j] for j in m.Y if j <= y) <= 0

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
        m.RAMP_UP = Constraint(m.G_THERM.union(m.G_E_HYDRO), m.Y, m.S, m.T, rule=ramp_up_rule)

        # def ramp_up_charging_rule(_m, g, y, s, t):
        #     """Ramp-rate up for storage charging power"""
        #
        #     if t == m.T.first():
        #         return m.p_in[g, y, s, t] - m.P_IN_0[g, y, s] - m.RR_UP[g] <= 0
        #     else:
        #         return m.p_in[g, y, s, t] - m.p_in[g, y, s, t - 1] - m.RR_UP[g] <= 0
        #
        # # Ramp-rate up charging
        # m.RAMP_UP_CHARGING = Constraint(m.G_STORAGE, m.Y, m.S, m.T, rule=ramp_up_charging_rule)
        #
        # def ramp_up_discharging_rule(_m, g, y, s, t):
        #     """Ramp-rate up for storage discharging power"""
        #
        #     if t == m.T.first():
        #         return m.p_out[g, y, s, t] - m.P_OUT_0[g, y, s] - m.RR_UP[g] <= 0
        #     else:
        #         return m.p_out[g, y, s, t] - m.p_out[g, y, s, t - 1] - m.RR_UP[g] <= 0
        #
        # # Ramp-rate up charging
        # m.RAMP_UP_DISCHARGING = Constraint(m.G_STORAGE, m.Y, m.S, m.T, rule=ramp_up_discharging_rule)

        def ramp_down_rule(_m, g, y, s, t):
            """Ramp down constraint"""

            if t == m.T.first():
                return - m.p[g, y, s, t] + m.P0[g, y, s] - m.RR_DOWN[g] <= 0
            else:
                return - m.p[g, y, s, t] + m.p[g, y, s, t - 1] - m.RR_DOWN[g] <= 0

        # Ramp-rate down constraint
        m.RAMP_DOWN = Constraint(m.G_THERM.union(m.G_E_HYDRO), m.Y, m.S, m.T, rule=ramp_down_rule)

        # def ramp_down_charging_rule(_m, g, y, s, t):
        #     """Ramp-rate down for storage charging power"""
        #
        #     if t == m.T.first():
        #         return - m.p_in[g, y, s, t] + m.P_IN_0[g, y, s] - m.RR_DOWN[g] <= 0
        #     else:
        #         return - m.p_in[g, y, s, t] + m.p_in[g, y, s, t - 1] - m.RR_DOWN[g] <= 0
        #
        # # Ramp-rate down constraint
        # m.RAMP_DOWN_CHARGING = Constraint(m.G_STORAGE, m.Y, m.S, m.T, rule=ramp_down_charging_rule)
        #
        # def ramp_down_discharging_rule(_m, g, y, s, t):
        #     """Ramp-rate down for storage charging power"""
        #
        #     if t == m.T.first():
        #         return - m.p_out[g, y, s, t] + m.P_OUT_0[g, y, s] - m.RR_DOWN[g] <= 0
        #     else:
        #         return - m.p_out[g, y, s, t] + m.p_out[g, y, s, t - 1] - m.RR_DOWN[g] <= 0
        #
        # # Ramp-rate down constraint
        # m.RAMP_DOWN_DISCHARGING = Constraint(m.G_STORAGE, m.Y, m.S, m.T, rule=ramp_down_discharging_rule)

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
            # storage_units = [gen for gen, zone in self.data.battery_properties_dict['NEM_ZONE'].items() if zone == z]

            return (m.DEMAND[z, y, s, t]
                    - sum(m.p[g, y, s, t] for g in generators)
                    + sum(m.INCIDENCE_MATRIX[l, z] * m.p_L[l, y, s, t] for l in m.L)
                    # - sum(m.p_out[g, y, s, t] - m.p_in[g, y, s, t] for g in storage_units)
                    - m.p_V[z, y, s, t]
                    == 0)

        # Power balance constraint for each zone and time period
        m.POWER_BALANCE = Constraint(m.Z, m.Y, m.S, m.T, rule=power_balance_rule)

        # def generator_energy_output_rule(_m, g, y, s, t):
        #     """Energy output from generators (excluding storage)"""
        #
        #     if t == m.T.first():
        #         return m.e[g, y, s, t] - ((m.P0[g, y, s] + m.p[g, y, s, t]) / 2) == 0
        #
        #     else:
        #         return m.e[g, y, s, t] - ((m.p[g, y, s, t - 1] + m.p[g, y, s, t]) / 2) == 0
        #
        # # Generator energy output
        # m.GENERATOR_ENERGY_OUT = Constraint(m.G.difference(m.G_STORAGE), m.Y, m.S, m.T,
        #                                     rule=generator_energy_output_rule)

        # def storage_energy_output_rule(_m, g, y, s, t):
        #     """Energy output from storage units"""
        #
        #     if t == m.T.first():
        #         return m.e[g, y, s, t] - ((m.P_OUT_0[g, y, s] + m.p_out[g, y, s, t]) / 2) == 0
        #     else:
        #         return m.e[g, y, s, t] - ((m.p_out[g, y, s, t - 1] + m.p_out[g, y, s, t]) / 2) == 0
        #
        # # Storage energy output
        # m.STORAGE_ENERGY_OUT = Constraint(m.G_STORAGE, m.Y, m.S, m.T, rule=storage_energy_output_rule)

        # def lost_load_energy_rule(_m, z, y, s, t):
        #     """Lost load energy"""
        #
        #     if t == m.T.first():
        #         return m.e_V[z, y, s, t] - ((m.P_V0[z, y, s] + m.p_V[z, y, s, t]) / 2) == 0
        #     else:
        #         return m.e_V[z, y, s, t] - ((m.p_V[z, y, s, t - 1] + m.p_V[z, y, s, t]) / 2) == 0
        #
        # # Lost load energy
        # m.LOST_LOAD_ENERGY = Constraint(m.Z, m.Y, m.S, m.T, rule=lost_load_energy_rule)

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

        # Add component allowing dual variables to be imported
        m.dual = Suffix(direction=Suffix.IMPORT)

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
    def __init__(self, final_year, scenarios_per_year):
        self.data = ModelData()
        self.components = CommonComponents(final_year, scenarios_per_year)

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

    @staticmethod
    def define_variables(m):
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
        # m.nu_1 = Var(m.G_C, m.Y, initialize=0)

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

        # Max charging power - existing storage
        m.sigma_11 = Var(m.G_E_STORAGE, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Max charging power - candidate storage
        m.sigma_12 = Var(m.G_C_STORAGE, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Max discharging power - existing storage
        m.sigma_13 = Var(m.G_E_STORAGE, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Max discharging power - candidate storage
        m.sigma_14 = Var(m.G_C_STORAGE, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Min energy - storage units
        m.sigma_15 = Var(m.G_STORAGE, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Max energy - existing storage units
        m.sigma_16 = Var(m.G_E_STORAGE, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Max energy - candidate storage
        m.sigma_17 = Var(m.G_C_STORAGE, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Min energy - interval end
        m.sigma_18 = Var(m.G_STORAGE, m.Y, m.S, within=NonNegativeReals, initialize=0)

        # Max energy - interval end
        m.sigma_19 = Var(m.G_STORAGE, m.Y, m.S, within=NonNegativeReals, initialize=0)

        # Ramp-rate up (thermal and hydro generators)
        m.sigma_20 = Var(m.G_THERM.union(m.G_E_HYDRO), m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # # Ramp-rate up - charging power - storage
        # m.sigma_21 = Var(m.G_STORAGE, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)
        #
        # # Ramp-rate up - discharging power - storage
        # m.sigma_22 = Var(m.G_STORAGE, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Ramp-rate down (thermal and hydro generators)
        m.sigma_23 = Var(m.G_THERM.union(m.G_E_HYDRO), m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # # Ramp-rate down - charging power - storage
        # m.sigma_24 = Var(m.G_STORAGE, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)
        #
        # # Ramp-rate down - discharging power - storage
        # m.sigma_25 = Var(m.G_STORAGE, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Non-negative lost load power
        m.sigma_26 = Var(m.Z, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Min powerflow
        m.sigma_27 = Var(m.L, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Max powerflow
        m.sigma_28 = Var(m.L, m.Y, m.S, m.T, within=NonNegativeReals, initialize=0)

        # Storage energy transition
        m.zeta_1 = Var(m.G_STORAGE, m.Y, m.S, m.T, initialize=0)

        # Energy output (all generators excluding storage)
        # m.zeta_2 = Var(m.G.difference(m.G_STORAGE), m.Y, m.S, m.T, initialize=0)

        # Energy output (storage units)
        # m.zeta_3 = Var(m.G_STORAGE, m.Y, m.S, m.T, initialize=0)

        # Energy - lost load
        # m.zeta_4 = Var(m.Z, m.Y, m.S, m.T, initialize=0)

        # Power balance (locational marginal price)
        m.lamb = Var(m.Z, m.Y, m.S, m.T, initialize=0)

        return m

    @staticmethod
    def define_expressions(m):
        """Define dual problem expressions"""

        def dual_objective_expression_rule(_m):
            """Expression for dual objective function"""

            # Build limits
            t_1 = sum(- (m.mu_2[z, y] * m.SOLAR_BUILD_LIMITS[z]) - (m.mu_3[z, y] * m.WIND_BUILD_LIMITS[z]) - (
                    m.mu_4[z, y] * m.STORAGE_BUILD_LIMITS[z]) for z in m.Z for y in m.Y)

            # Min power output
            t_2 = sum(
                m.sigma_1[g, y, s, t] * m.P_MIN[g] for g in m.G.difference(m.G_STORAGE) for y in m.Y for s in m.S for t
                in m.T)

            # Max power - existing generators
            t_3 = sum(
                - m.sigma_2[g, y, s, t] * m.P_MAX[g] * (1 - m.F[g, y]) for g in m.G_E_THERM for y in m.Y for s in m.S
                for t in m.T)

            # Max power - existing wind
            t_4 = sum(
                - m.sigma_4[g, y, s, t] * m.Q_W[g, y, s, t] * m.P_MAX[g] * (1 - m.F[g, y]) for g in m.G_E_WIND for y in
                m.Y for s in m.S for t in m.T)

            # Max power - existing solar
            t_5 = sum(
                - m.sigma_6[g, y, s, t] * m.Q_S[g, y, s, t] * m.P_MAX[g] * (1 - m.F[g, y]) for g in m.G_E_SOLAR for y in
                m.Y for s in m.S for t in m.T)

            # Max power - existing hydro
            t_6 = sum(
                - m.sigma_8[g, y, s, t] * m.P_H[g, y, s, t] * (1 - m.F[g, y]) for g in m.G_E_HYDRO for y in m.Y for s in
                m.S for t in m.T)

            # Max charging power - existing storage
            t_7 = sum(
                - m.sigma_11[g, y, s, t] * m.P_IN_MAX[g] * (1 - m.F[g, y]) for g in m.G_E_STORAGE for y in m.Y for s in
                m.S for t in m.T)

            # Max discharging power - existing storage
            t_8 = sum(
                - m.sigma_13[g, y, s, t] * m.P_OUT_MAX[g] * (1 - m.F[g, y]) for g in m.G_E_STORAGE for y in m.Y for s in
                m.S for t in m.T)

            # Max energy - existing storage units
            t_9 = sum(
                - m.sigma_16[g, y, s, t] * m.Q_MAX[g] for g in m.G_E_STORAGE for y in m.Y for s in m.S for t in m.T)

            # Min energy - interval end
            t_10 = sum(m.sigma_18[g, y, s] * m.Q_END_MIN[g] for g in m.G_STORAGE for y in m.Y for s in m.S)

            # Max energy - interval end
            t_11 = sum(- m.sigma_19[g, y, s] * m.Q_END_MAX[g] for g in m.G_STORAGE for y in m.Y for s in m.S)

            # Ramp-up constraint - generators
            t_12 = sum(
                - m.sigma_20[g, y, s, t] * m.RR_UP[g] for g in m.G_THERM.union(m.G_E_HYDRO) for y in m.Y for s in m.S
                for t in m.T)

            # Ramp-up constraint - initial power output - generators
            t_13 = sum(
                - m.sigma_20[g, y, s, m.T.first()] * m.P0[g, y, s] for g in m.G_THERM.union(m.G_E_HYDRO) for y in m.Y
                for s in m.S)

            # # Ramp-up constraint - storage charging
            # t_14 = sum(- m.sigma_21[g, y, s, t] * m.RR_UP[g] for g in m.G_STORAGE for y in m.Y for s in m.S for t in m.T)

            # # Ramp-up constraint - initial charging power - storage
            # t_15 = sum(- m.sigma_21[g, y, s, m.T.first()] * m.P_IN_0[g, y, s] for g in m.G_STORAGE for y in m.Y for s in m.S)

            # # Ramp-up constraint - storage discharging
            # t_16 = sum(- m.sigma_22[g, y, s, t] * m.RR_UP[g] for g in m.G_STORAGE for y in m.Y for s in m.S for t in m.T)
            #
            # # Ramp-up constraint - initial discharging power - storage
            # t_17 = sum(- m.sigma_22[g, y, s, m.T.first()] * m.P_OUT_0[g, y, s] for g in m.G_STORAGE for y in m.Y for s in m.S)

            # Ramp-down constraint - generators
            t_18 = sum(
                - m.sigma_23[g, y, s, t] * m.RR_DOWN[g] for g in m.G_THERM.union(m.G_E_HYDRO) for y in m.Y for s in m.S
                for t in m.T)

            # Ramp-down constraint - initial power output - generators
            t_19 = sum(
                m.sigma_23[g, y, s, m.T.first()] * m.P0[g, y, s] for g in m.G_THERM.union(m.G_E_HYDRO) for y in m.Y for
                s in m.S)

            # # Ramp-down constraint - storage charging
            # t_20 = sum(- m.sigma_24[g, y, s, t] * m.RR_DOWN[g] for g in m.G_STORAGE for y in m.Y for s in m.S for t in m.T)

            # # Ramp-down constraint - initial charging power - storage
            # t_21 = sum(m.sigma_24[g, y, s, m.T.first()] * m.P_IN_0[g, y, s] for g in m.G_STORAGE for y in m.Y for s in m.S)
            #
            # # Ramp-down constraint - storage discharging
            # t_22 = sum(- m.sigma_25[g, y, s, t] * m.RR_DOWN[g] for g in m.G_STORAGE for y in m.Y for s in m.S for t in m.T)
            #
            # # Ramp-down constraint - initial discharging power - storage
            # t_23 = sum(m.sigma_25[g, y, s, m.T.first()] * m.P_OUT_0[g, y, s] for g in m.G_STORAGE for y in m.Y for s in m.S)

            # Min powerflow
            t_24 = sum(m.sigma_27[l, y, s, t] * m.POWERFLOW_MIN[l] for l in m.L for y in m.Y for s in m.S for t in m.T)

            # Max powerflow
            t_25 = sum(
                - m.sigma_28[l, y, s, t] * m.POWERFLOW_MAX[l] for l in m.L for y in m.Y for s in m.S for t in m.T)

            # Demand
            t_26 = sum(m.lamb[z, y, s, t] * m.DEMAND[z, y, s, t] for z in m.Z for y in m.Y for s in m.S for t in m.T)

            # Initial storage unit energy
            t_27 = sum(m.zeta_1[g, y, s, m.T.first()] * m.Q0[g, y, s] for g in m.G_STORAGE for y in m.Y for s in m.S)

            # Initial generator power output
            # t_28 = sum(- m.zeta_2[g, y, s, m.T.first()] * (m.P0[g, y, s] / 2) for g in m.G.difference(m.G_STORAGE) for y in m.Y for s in m.S)

            # # Initial storage unit power output
            # t_29 = sum(- m.zeta_3[g, y, s, m.T.first()] * (m.P_OUT_0[g, y, s] / 2) for g in m.G_STORAGE for y in m.Y for s in m.S)

            # Initial lost-load power
            # t_30 = sum(- m.zeta_4[z, y, s, m.T.first()] * (m.P_V0[z, y, s] / 2) for z in m.Z for y in m.Y for s in m.S)

            # Fixed operations and maintenance cost - existing generators
            # t_31 = sum(m.DELTA[y] * m.C_FOM[g] * m.P_MAX[g] * (1 - m.F[g, y]) for g in m.G_E for y in m.Y)

            # # Fixed operations and maintenance cost - existing generators - end of horizon cost
            # t_32 = (m.DELTA[m.Y.last()] / m.INTEREST_RATE) * sum(m.C_FOM[g] * m.P_MAX[g] * (1 - m.F[g, m.Y.last()]) for g in m.G_E)

            return t_1 + t_2 + t_3 + t_4 + t_5 + t_6 + t_7 + t_8 + t_9 + t_10 + t_11 + t_12 + t_13 + t_18 + t_19 + t_24 + t_25 + t_26 + t_27

        # Dual objective expression
        m.DUAL_OBJECTIVE_EXPRESSION = Expression(rule=dual_objective_expression_rule)

        def scenario_average_price(_m, y, s):
            """Average price for a given scenario"""

            # Total demand
            demand = sum(m.DEMAND[z, y, s, t] for z in m.Z for t in m.T)

            if y != m.Y.last():
                # Revenue from electricity sales (wholesale)
                revenue = sum((m.lamb[z, y, s, t] / m.RHO[y, s]) * m.DEMAND[z, y, s, t] for z in m.Z for t in m.T)

            else:
                # Revenue from electricity sales (wholesale)
                revenue = sum(
                    (m.lamb[z, y, s, t] / m.RHO[y, s]) * (1 / (1 + (1 / m.INTEREST_RATE))) * m.DEMAND[z, y, s, t] for z
                    in m.Z for t in m.T)

            # Average price
            average_price = revenue / demand

            return average_price

        # Scenario demand weighted average wholesale price
        m.SCENARIO_AVERAGE_PRICE = Expression(m.Y, m.S, rule=scenario_average_price)

        def year_average_price(_m, y):
            """Average price for a given year"""

            # Days in year - accounting for leap-years
            days = sum(m.RHO[y, s] for s in m.S)

            return sum((m.RHO[y, s] / days) * m.SCENARIO_AVERAGE_PRICE[y, s] for s in m.S)

        # Year demand weighted average wholesale price
        m.YEAR_AVERAGE_PRICE = Expression(m.Y, rule=year_average_price)

        return m

    def define_constraints(self, m):
        """Define dual problem constraints"""

        # def investment_decisions_rule(_m, g, y):
        #     """Investment decision constraint (x_c)"""
        #
        #     return ((m.DELTA[y] / m.INTEREST_RATE) * m.GAMMA[g] * m.I_C[g, y]) - m.mu_1[g, y] + sum(-m.nu_1[g, j] for j in m.Y if j >= y) == 0
        #
        # # Investment decision in each year
        # m.INVESTMENT_DECISION = Constraint(m.G_C, m.Y, rule=investment_decisions_rule)

        def investment_decision_thermal_rule(_m, g, y):
            """Investment decision constraints for candidate thermal generators (x_c)"""

            return (((m.DELTA[y] / m.INTEREST_RATE) * m.GAMMA[g] * m.I_C[g, y])
                    + sum(m.DELTA[j] * m.C_FOM[g] for j in m.Y if j >= y)
                    - m.mu_1[g, y]
                    + ((m.DELTA[m.Y.last()] / m.INTEREST_RATE) * m.C_FOM[g])
                    + sum(- m.sigma_3[g, j, s, t] for s in m.S for t in m.T for j in m.Y if j >= y)
                    == 0)

        # Investment decision for thermal plant
        m.INVESTMENT_DECISION_THERMAL = Constraint(m.G_C_THERM, m.Y, rule=investment_decision_thermal_rule)

        def investment_decision_wind_rule(_m, g, y):
            """Investment decision constraints for candidate wind generators (x_c)"""

            return (((m.DELTA[y] / m.INTEREST_RATE) * m.GAMMA[g] * m.I_C[g, y])
                    + sum(m.DELTA[j] * m.C_FOM[g] for j in m.Y if j >= y)
                    + sum(m.mu_3[self.k(m, g), j] for j in m.Y if j >= y)
                    - m.mu_1[g, y]
                    + ((m.DELTA[m.Y.last()] / m.INTEREST_RATE) * m.C_FOM[g])
                    # + sum(m.mu_3[self.k(m, g), j] for j in m.Y if j >= y)
                    + sum(- m.Q_W[g, j, s, t] * m.sigma_5[g, j, s, t] for s in m.S for t in m.T for j in m.Y if j >= y)
                    == 0)

        # Investment decision for wind plant
        m.INVESTMENT_DECISION_WIND = Constraint(m.G_C_WIND, m.Y, rule=investment_decision_wind_rule)

        def investment_decision_solar_rule(_m, g, y):
            """Investment decision constraints for candidate solar generators (x_c)"""

            return (((m.DELTA[y] / m.INTEREST_RATE) * m.GAMMA[g] * m.I_C[g, y])
                    + sum(m.DELTA[j] * m.C_FOM[g] for j in m.Y if j >= y)
                    + sum(m.mu_2[self.k(m, g), j] for j in m.Y if j >= y)
                    - m.mu_1[g, y]
                    + ((m.DELTA[m.Y.last()] / m.INTEREST_RATE) * m.C_FOM[g])
                    # + sum(m.mu_2[self.k(m, g), j] for j in m.Y if j >= y)
                    + sum(- m.Q_S[g, j, s, t] * m.sigma_7[g, j, s, t] for s in m.S for t in m.T for j in m.Y if j >= y)
                    == 0)

        # Investment decision for solar plant
        m.INVESTMENT_DECISION_SOLAR = Constraint(m.G_C_SOLAR, m.Y, rule=investment_decision_solar_rule)

        def investment_decision_storage_rule(_m, g, y):
            """Investment decision constraints for candidate thermal generators (x_c)"""

            return (((m.DELTA[y] / m.INTEREST_RATE) * m.GAMMA[g] * m.I_C[g, y])
                    + sum(m.DELTA[j] * m.C_FOM[g] for j in m.Y if j >= y)
                    + sum(m.mu_4[self.k(m, g), j] for j in m.Y if j >= y)
                    - m.mu_1[g, y]
                    + ((m.DELTA[m.Y.last()] / m.INTEREST_RATE) * m.C_FOM[g])
                    + sum(
                        (- m.sigma_12[g, j, s, t] - m.sigma_14[g, j, s, t] - m.sigma_17[g, j, s, t]) for s in m.S for t
                        in m.T for j in m.Y if j >= y)
                    == 0)

        # Investment decision for storage units
        m.INVESTMENT_DECISION_STORAGE = Constraint(m.G_C_STORAGE, m.Y, rule=investment_decision_storage_rule)

        # def total_capacity_thermal_rule(_m, g, y):
        #     """Total thermal generator installed capacity"""
        #
        #     # if y != m.Y.last():
        #     return m.nu_1[g, y] + (m.DELTA[y] * m.C_FOM[g]) + sum(- m.sigma_3[g, y, s, t] for s in m.S for t in m.T) == 0
        #     # else:
        #     #     return m.nu_1[g, y] + (m.DELTA[y] * (1 + (1 / m.INTEREST_RATE)) * m.C_FOM[g]) + sum(- m.sigma_3[g, y, s, t] for s in m.S for t in m.T) == 0
        #
        # # Total installed capacity
        # m.TOTAL_THERMAL_CAPACITY = Constraint(m.G_C_THERM, m.Y, rule=total_capacity_thermal_rule)
        #
        # def total_capacity_solar_rule(_m, g, y):
        #     """Total solar generator installed capacity"""
        #
        #     # if y != m.Y.last():
        #     return m.mu_2[self.k(m, g), y] + m.nu_1[g, y] + (m.DELTA[y] * m.C_FOM[g]) + sum(- m.Q_S[g, y, s, t] * m.sigma_7[g, y, s, t] for s in m.S for t in m.T) == 0
        #     # else:
        #     #     return m.mu_2[self.k(m, g), y] + m.nu_1[g, y] + (m.DELTA[y] * (1 + (1 / m.INTEREST_RATE)) * m.C_FOM[g]) + sum(- m.Q_S[g, y, s, t] * m.sigma_7[g, y, s, t] for s in m.S for t in m.T) == 0
        #
        # # Total installed capacity
        # m.TOTAL_SOLAR_CAPACITY = Constraint(m.G_C_SOLAR, m.Y, rule=total_capacity_solar_rule)
        #
        # def total_capacity_wind_rule(_m, g, y):
        #     """Total wind generator installed capacity"""
        #
        #     # if y != m.Y.last():
        #     return m.mu_3[self.k(m, g), y] + m.nu_1[g, y] + (m.DELTA[y] * m.C_FOM[g]) + sum(- m.Q_W[g, y, s, t] * m.sigma_5[g, y, s, t] for s in m.S for t in m.T) == 0
        #     # else:
        #     #     return m.mu_3[self.k(m, g), y] + m.nu_1[g, y] + (m.DELTA[y] * (1 + (1 / m.INTEREST_RATE)) * m.C_FOM[g]) + sum(- m.Q_W[g, y, s, t] * m.sigma_5[g, y, s, t] for s in m.S for t in m.T) == 0
        #
        # # Total installed capacity
        # m.TOTAL_WIND_CAPACITY = Constraint(m.G_C_WIND, m.Y, rule=total_capacity_wind_rule)

        # def total_capacity_storage_rule(_m, g, y):
        #     """Total storage installed capacity"""
        #
        #     if y != m.Y.last():
        #         return m.mu_4[self.k(m, g), y] + m.nu_1[g, y] + (m.DELTA[y] * m.C_FOM[g]) + sum(- m.sigma_14[g, y, s, t] for s in m.S for t in m.T) == 0
        #     else:
        #         return m.mu_4[self.k(m, g), y] + m.nu_1[g, m.Y.last()] + (m.DELTA[m.Y.last()] * (1 + (1 / m.INTEREST_RATE)) * m.C_FOM[g]) + sum(- m.sigma_14[g, m.Y.last(), s, t] for s in m.S for t in m.T) == 0
        #
        # # Total installed capacity
        # m.TOTAL_STORAGE_CAPACITY = Constraint(m.G_C_STORAGE, m.Y, rule=total_capacity_storage_rule)

        def power_output_existing_thermal_rule(_m, g, y, s, t):
            """Power output from existing thermal generators"""

            if y != m.Y.last() and t != m.T.last():
                return (- m.sigma_1[g, y, s, t] + m.sigma_2[g, y, s, t]
                        + m.sigma_20[g, y, s, t] - m.sigma_20[g, y, s, t + 1]
                        - m.sigma_23[g, y, s, t] + m.sigma_23[g, y, s, t + 1]
                        - m.lamb[self.k(m, g), y, s, t]
                        # - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        + ((m.DELTA[y] * m.RHO[y, s]) * (
                                m.C_MC[g, y] + ((m.EMISSIONS_RATE[g] - m.baseline[y]) * m.permit_price[y])))
                        == 0)

            elif y != m.Y.last() and t == m.T.last():
                return (- m.sigma_1[g, y, s, t] + m.sigma_2[g, y, s, t]
                        + m.sigma_20[g, y, s, t]
                        - m.sigma_23[g, y, s, t]
                        - m.lamb[self.k(m, g), y, s, t]
                        # - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        + ((m.DELTA[y] * m.RHO[y, s]) * (
                                m.C_MC[g, y] + ((m.EMISSIONS_RATE[g] - m.baseline[y]) * m.permit_price[y])))
                        == 0)

            elif y == m.Y.last() and t != m.T.last():
                return (- m.sigma_1[g, y, s, t] + m.sigma_2[g, y, s, t]
                        + m.sigma_20[g, y, s, t] - m.sigma_20[g, y, s, t + 1]
                        - m.sigma_23[g, y, s, t] + m.sigma_23[g, y, s, t + 1]
                        - m.lamb[self.k(m, g), y, s, t]
                        # - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        + ((m.DELTA[y] * m.RHO[y, s]) * (1 + (1 / m.INTEREST_RATE)) * (
                                m.C_MC[g, y] + ((m.EMISSIONS_RATE[g] - m.baseline[y]) * m.permit_price[y])))
                        == 0)

            elif y == m.Y.last() and t == m.T.last():
                return (- m.sigma_1[g, y, s, t] + m.sigma_2[g, y, s, t]
                        + m.sigma_20[g, y, s, t]
                        - m.sigma_23[g, y, s, t]
                        - m.lamb[self.k(m, g), y, s, t]
                        # - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        + ((m.DELTA[y] * m.RHO[y, s]) * (1 + (1 / m.INTEREST_RATE)) * (
                                m.C_MC[g, y] + ((m.EMISSIONS_RATE[g] - m.baseline[y]) * m.permit_price[y])))
                        == 0)

            else:
                raise Exception(f'Unhandled case: {g, y, s, t}')

        # Total power output from thermal generators
        m.POWER_OUTPUT_EXISTING_THERMAL = Constraint(m.G_E_THERM, m.Y, m.S, m.T,
                                                     rule=power_output_existing_thermal_rule)

        def power_output_candidate_thermal_rule(_m, g, y, s, t):
            """Power output from candidate thermal generators"""

            if y != m.Y.last() and t != m.T.last():
                return (- m.sigma_1[g, y, s, t] + m.sigma_3[g, y, s, t]
                        + m.sigma_20[g, y, s, t] - m.sigma_20[g, y, s, t + 1]
                        - m.sigma_23[g, y, s, t] + m.sigma_23[g, y, s, t + 1]
                        - m.lamb[self.k(m, g), y, s, t]
                        # - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        + ((m.DELTA[y] * m.RHO[y, s]) * (
                                m.C_MC[g, y] + ((m.EMISSIONS_RATE[g] - m.baseline[y]) * m.permit_price[y])))
                        == 0)

            elif y != m.Y.last() and t == m.T.last():
                return (- m.sigma_1[g, y, s, t] + m.sigma_3[g, y, s, t]
                        + m.sigma_20[g, y, s, t]
                        - m.sigma_23[g, y, s, t]
                        - m.lamb[self.k(m, g), y, s, t]
                        # - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        + ((m.DELTA[y] * m.RHO[y, s]) * (
                                m.C_MC[g, y] + ((m.EMISSIONS_RATE[g] - m.baseline[y]) * m.permit_price[y])))
                        == 0)

            elif y == m.Y.last() and t != m.T.last():
                return (- m.sigma_1[g, y, s, t] + m.sigma_3[g, y, s, t]
                        + m.sigma_20[g, y, s, t] - m.sigma_20[g, y, s, t + 1]
                        - m.sigma_23[g, y, s, t] + m.sigma_23[g, y, s, t + 1]
                        - m.lamb[self.k(m, g), y, s, t]
                        # - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        + ((m.DELTA[y] * m.RHO[y, s]) * (1 + (1 / m.INTEREST_RATE)) * (
                                m.C_MC[g, y] + ((m.EMISSIONS_RATE[g] - m.baseline[y]) * m.permit_price[y])))
                        == 0)

            elif y == m.Y.last() and t == m.T.last():
                return (- m.sigma_1[g, y, s, t] + m.sigma_3[g, y, s, t]
                        + m.sigma_20[g, y, s, t]
                        - m.sigma_23[g, y, s, t]
                        - m.lamb[self.k(m, g), y, s, t]
                        # - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        + ((m.DELTA[y] * m.RHO[y, s]) * (1 + (1 / m.INTEREST_RATE)) * (
                                m.C_MC[g, y] + ((m.EMISSIONS_RATE[g] - m.baseline[y]) * m.permit_price[y])))
                        == 0)

            else:
                raise Exception(f'Unhandled case: {g, y, s, t}')

            # if y != m.Y.last():
            #     return (- m.sigma_1[g, y, s, t] + m.sigma_3[g, y, s, t]
            #             # + m.sigma_20[g, y, s, t] - m.sigma_20[g, y, s, t+1]
            #             # - m.sigma_23[g, y, s, t] + m.sigma_23[g, y, s, t+1]
            #             - m.lamb[self.k(m, g), y, s, t]
            #             # - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
            #             + ((m.DELTA[y] * m.RHO[y, s]) * (m.C_MC[g, y] + ((m.EMISSIONS_RATE[g] - m.baseline[y]) * m.permit_price[y])))
            #             == 0)
            # else:
            #     return (- m.sigma_1[g, y, s, t] + m.sigma_3[g, y, s, t]
            #             # + m.sigma_20[g, y, s, t] - m.sigma_20[g, y, s, t+1]
            #             # - m.sigma_23[g, y, s, t] + m.sigma_23[g, y, s, t+1]
            #             - m.lamb[self.k(m, g), y, s, t]
            #             # - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
            #             + ((m.DELTA[y] * m.RHO[y, s]) * (1 + (1 / m.INTEREST_RATE)) * (m.C_MC[g, y] + ((m.EMISSIONS_RATE[g] - m.baseline[y]) * m.permit_price[y])))
            #             == 0)

        # Total power output from candidate thermal generators
        m.POWER_OUTPUT_CANDIDATE_THERMAL = Constraint(m.G_C_THERM, m.Y, m.S, m.T,
                                                      rule=power_output_candidate_thermal_rule)

        def power_output_existing_wind_rule(_m, g, y, s, t):
            """Power output from existing wind generators"""

            if y != m.Y.last():
                return (- m.sigma_1[g, y, s, t] + m.sigma_4[g, y, s, t]
                        # + m.sigma_20[g, y, s, t] - m.sigma_20[g, y, s, t+1]
                        # - m.sigma_23[g, y, s, t] + m.sigma_23[g, y, s, t+1]
                        - m.lamb[self.k(m, g), y, s, t]
                        # - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        + (m.DELTA[y] * m.RHO[y, s] * m.C_MC[g, y])
                        == 0)
            else:
                return (- m.sigma_1[g, y, s, t] + m.sigma_4[g, y, s, t]
                        # + m.sigma_20[g, y, s, t] - m.sigma_20[g, y, s, t+1]
                        # - m.sigma_23[g, y, s, t] + m.sigma_23[g, y, s, t+1]
                        - m.lamb[self.k(m, g), y, s, t]
                        # - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        + (m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * m.C_MC[g, y])
                        == 0)

        # Total power output from existing wind generators
        m.POWER_OUTPUT_EXISTING_WIND = Constraint(m.G_E_WIND, m.Y, m.S, m.T, rule=power_output_existing_wind_rule)

        def power_output_candidate_wind_rule(_m, g, y, s, t):
            """Power output from candidate wind generators"""

            if y != m.Y.last():
                return (- m.sigma_1[g, y, s, t] + m.sigma_5[g, y, s, t]
                        # + m.sigma_20[g, y, s, t] - m.sigma_20[g, y, s, t+1]
                        # - m.sigma_23[g, y, s, t] + m.sigma_23[g, y, s, t+1]
                        - m.lamb[self.k(m, g), y, s, t]
                        # - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        + ((m.DELTA[y] * m.RHO[y, s]) * (m.C_MC[g, y] - (m.baseline[y] * m.permit_price[y])))
                        == 0)
            else:
                return (- m.sigma_1[g, y, s, t] + m.sigma_5[g, y, s, t]
                        # + m.sigma_20[g, y, s, t] - m.sigma_20[g, y, s, t+1]
                        # - m.sigma_23[g, y, s, t] + m.sigma_23[g, y, s, t+1]
                        - m.lamb[self.k(m, g), y, s, t]
                        # - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        + ((m.DELTA[y] * m.RHO[y, s]) * (1 + (1 / m.INTEREST_RATE)) * (
                                m.C_MC[g, y] - (m.baseline[y] * m.permit_price[y])))
                        == 0)

        # Total power output from candidate wind generators
        m.POWER_OUTPUT_CANDIDATE_WIND = Constraint(m.G_C_WIND, m.Y, m.S, m.T, rule=power_output_candidate_wind_rule)

        def power_output_existing_solar_rule(_m, g, y, s, t):
            """Power output from existing solar generators"""

            if y != m.Y.last():
                return (- m.sigma_1[g, y, s, t] + m.sigma_6[g, y, s, t]
                        # + m.sigma_20[g, y, s, t] - m.sigma_20[g, y, s, t+1]
                        # - m.sigma_23[g, y, s, t] + m.sigma_23[g, y, s, t+1]
                        - m.lamb[self.k(m, g), y, s, t]
                        # - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        + (m.DELTA[y] * m.RHO[y, s] * m.C_MC[g, y])
                        == 0)

            else:
                return (- m.sigma_1[g, y, s, t] + m.sigma_6[g, y, s, t]
                        # + m.sigma_20[g, y, s, t] - m.sigma_20[g, y, s, t+1]
                        # - m.sigma_23[g, y, s, t] + m.sigma_23[g, y, s, t+1]
                        - m.lamb[self.k(m, g), y, s, t]
                        # - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        + (m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * m.C_MC[g, y])
                        == 0)

        # Total power output from existing solar generators
        m.POWER_OUTPUT_EXISTING_SOLAR = Constraint(m.G_E_SOLAR, m.Y, m.S, m.T, rule=power_output_existing_solar_rule)

        def power_output_candidate_solar_rule(_m, g, y, s, t):
            """Power output from candidate solar generators"""

            if y != m.Y.last():
                return (- m.sigma_1[g, y, s, t] + m.sigma_7[g, y, s, t]
                        # + m.sigma_20[g, y, s, t] - m.sigma_20[g, y, s, t+1]
                        # - m.sigma_23[g, y, s, t] + m.sigma_23[g, y, s, t+1]
                        - m.lamb[self.k(m, g), y, s, t]
                        # - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        + ((m.DELTA[y] * m.RHO[y, s]) * (m.C_MC[g, y] - (m.baseline[y] * m.permit_price[y])))
                        == 0)

            else:
                return (- m.sigma_1[g, y, s, t] + m.sigma_7[g, y, s, t]
                        # + m.sigma_20[g, y, s, t] - m.sigma_20[g, y, s, t+1]
                        # - m.sigma_23[g, y, s, t] + m.sigma_23[g, y, s, t+1]
                        - m.lamb[self.k(m, g), y, s, t]
                        # - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        + ((m.DELTA[y] * m.RHO[y, s]) * (1 + (1 / m.INTEREST_RATE)) * (
                                m.C_MC[g, y] - (m.baseline[y] * m.permit_price[y])))
                        == 0)

        # Total power output from candidate solar generators
        m.POWER_OUTPUT_CANDIDATE_SOLAR = Constraint(m.G_C_SOLAR, m.Y, m.S, m.T, rule=power_output_candidate_solar_rule)

        def power_output_hydro_rule(_m, g, y, s, t):
            """Power output from hydro generators"""

            if y != m.Y.last() and t != m.T.last():
                return (- m.sigma_1[g, y, s, t] + m.sigma_8[g, y, s, t]
                        + m.sigma_20[g, y, s, t] - m.sigma_20[g, y, s, t + 1]
                        - m.sigma_23[g, y, s, t] + m.sigma_23[g, y, s, t + 1]
                        - m.lamb[self.k(m, g), y, s, t]
                        # - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        + (m.DELTA[y] * m.RHO[y, s] * m.C_MC[g, y])
                        == 0)

            elif y != m.Y.last() and t == m.T.last():
                return (- m.sigma_1[g, y, s, t] + m.sigma_8[g, y, s, t]
                        + m.sigma_20[g, y, s, t]
                        - m.sigma_23[g, y, s, t]
                        - m.lamb[self.k(m, g), y, s, t]
                        # - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        + (m.DELTA[y] * m.RHO[y, s] * m.C_MC[g, y])
                        == 0)

            elif y == m.Y.last() and t != m.T.last():
                return (- m.sigma_1[g, y, s, t] + m.sigma_8[g, y, s, t]
                        + m.sigma_20[g, y, s, t] - m.sigma_20[g, y, s, t + 1]
                        - m.sigma_23[g, y, s, t] + m.sigma_23[g, y, s, t + 1]
                        - m.lamb[self.k(m, g), y, s, t]
                        # - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        + (m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * m.C_MC[g, y])
                        == 0)

            elif y == m.Y.last() and t == m.T.last():
                return (- m.sigma_1[g, y, s, t] + m.sigma_8[g, y, s, t]
                        + m.sigma_20[g, y, s, t]
                        - m.sigma_23[g, y, s, t]
                        - m.lamb[self.k(m, g), y, s, t]
                        # - ((m.zeta_2[g, y, s, t] + m.zeta_2[g, y, s, t+1]) / 2)
                        + (m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * m.C_MC[g, y])
                        == 0)

            else:
                raise Exception(f'Unexpected case: {g, y, s, t}')

        # Total power output from hydro generators
        m.POWER_OUTPUT_HYDRO = Constraint(m.G_E_HYDRO, m.Y, m.S, m.T, rule=power_output_hydro_rule)

        def charging_power_existing_storage_rule(_m, g, y, s, t):
            """Charging power for existing storage units"""

            # if t != m.T.last():
            return (- m.sigma_9[g, y, s, t] + m.sigma_11[g, y, s, t] + m.lamb[self.k(m, g), y, s, t]
                    + (m.ETA[g] * m.zeta_1[g, y, s, t])
                    # + m.sigma_21[g, y, s, t] - m.sigma_21[g, y, s, t + 1]
                    # - m.sigma_24[g, y, s, t] + m.sigma_24[g, y, s, t + 1]
                    == 0)
            #
            # else:
            #     return (- m.sigma_9[g, y, s, t] + m.sigma_11[g, y, s, t] + m.lamb[self.k(m, g), y, s, t]
            #             + (m.ETA[g] * m.zeta_1[g, y, s, t])
            #             + m.sigma_21[g, y, s, t]
            #             - m.sigma_24[g, y, s, t]
            #             == 0)

        # Existing storage unit charging power
        m.CHARGING_POWER_EXISTING_STORAGE = Constraint(m.G_E_STORAGE, m.Y, m.S, m.T,
                                                       rule=charging_power_existing_storage_rule)

        def charging_power_candidate_storage_rule(_m, g, y, s, t):
            """Charging power for candidate storage units"""

            # if t != m.T.last():
            return (- m.sigma_9[g, y, s, t] + m.sigma_12[g, y, s, t] + m.lamb[self.k(m, g), y, s, t]
                    + (m.ETA[g] * m.zeta_1[g, y, s, t])
                    # + m.sigma_21[g, y, s, t] - m.sigma_21[g, y, s, t + 1]
                    # - m.sigma_24[g, y, s, t] + m.sigma_24[g, y, s, t + 1]
                    == 0)
            #
            # else:
            #     return (- m.sigma_9[g, y, s, t] + m.sigma_12[g, y, s, t] + m.lamb[self.k(m, g), y, s, t]
            #             + (m.ETA[g] * m.zeta_1[g, y, s, t])
            #             + m.sigma_21[g, y, s, t]
            #             - m.sigma_24[g, y, s, t]
            #             == 0)

        # Existing storage unit charging power
        m.CHARGING_POWER_CANDIDATE_STORAGE = Constraint(m.G_C_STORAGE, m.Y, m.S, m.T,
                                                        rule=charging_power_candidate_storage_rule)

        def discharging_power_existing_storage_rule(_m, g, y, s, t):
            """Discharging power for existing storage units"""

            if y != m.Y.last():
                return (- m.sigma_10[g, y, s, t] + m.sigma_13[g, y, s, t] - m.lamb[self.k(m, g), y, s, t]
                        - ((1 / m.ETA[g]) * m.zeta_1[g, y, s, t])
                        + (m.DELTA[y] * m.RHO[y, s] * m.C_MC[g, y])
                        # - ((m.zeta_3[g, y, s, t] + m.zeta_3[g, y, s, t+1]) / 2)
                        # + m.sigma_22[g, y, s, t] - m.sigma_22[g, y, s, t + 1]
                        # - m.sigma_25[g, y, s, t] + m.sigma_25[g, y, s, t + 1]
                        == 0)
            else:
                return (- m.sigma_10[g, y, s, t] + m.sigma_13[g, y, s, t] - m.lamb[self.k(m, g), y, s, t]
                        - ((1 / m.ETA[g]) * m.zeta_1[g, y, s, t])
                        + (m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * m.C_MC[g, y])
                        # - ((m.zeta_3[g, y, s, t] + m.zeta_3[g, y, s, t+1]) / 2)
                        # + m.sigma_22[g, y, s, t] - m.sigma_22[g, y, s, t + 1]
                        # - m.sigma_25[g, y, s, t] + m.sigma_25[g, y, s, t + 1]
                        == 0)  # else:
            #     return (- m.sigma_10[g, y, s, t] + m.sigma_13[g, y, s, t] - m.lamb[self.k(m, g), y, s, t]
            #             - ((1 / m.ETA[g]) * m.zeta_1[g, y, s, t])
            #             - (m.zeta_3[g, y, s, t] / 2)
            #             + m.sigma_22[g, y, s, t]
            #             - m.sigma_25[g, y, s, t]
            #             == 0)

        # Existing storage unit discharging power
        m.DISCHARGING_POWER_EXISTING_STORAGE = Constraint(m.G_E_STORAGE, m.Y, m.S, m.T,
                                                          rule=discharging_power_existing_storage_rule)

        def discharging_power_candidate_storage_rule(_m, g, y, s, t):
            """Discharging power for candidate storage units"""

            if y != m.Y.last():
                return (- m.sigma_10[g, y, s, t] + m.sigma_14[g, y, s, t] - m.lamb[self.k(m, g), y, s, t]
                        - ((1 / m.ETA[g]) * m.zeta_1[g, y, s, t])
                        + (m.DELTA[y] * m.RHO[y, s] * m.C_MC[g, y])
                        # - ((m.zeta_3[g, y, s, t] + m.zeta_3[g, y, s, t + 1]) / 2)
                        # + m.sigma_22[g, y, s, t] - m.sigma_22[g, y, s, t + 1]
                        # - m.sigma_25[g, y, s, t] + m.sigma_25[g, y, s, t + 1]
                        == 0)
            else:
                return (- m.sigma_10[g, y, s, t] + m.sigma_14[g, y, s, t] - m.lamb[self.k(m, g), y, s, t]
                        - ((1 / m.ETA[g]) * m.zeta_1[g, y, s, t])
                        + (m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * m.C_MC[g, y])
                        # - ((m.zeta_3[g, y, s, t] + m.zeta_3[g, y, s, t + 1]) / 2)
                        # + m.sigma_22[g, y, s, t] - m.sigma_22[g, y, s, t + 1]
                        # - m.sigma_25[g, y, s, t] + m.sigma_25[g, y, s, t + 1]
                        == 0)
            # else:
            #     return (- m.sigma_10[g, y, s, t] + m.sigma_14[g, y, s, t] - m.lamb[self.k(m, g), y, s, t]
            #             - ((1 / m.ETA[g]) * m.zeta_1[g, y, s, t])
            #             - (m.zeta_3[g, y, s, t] / 2)
            #             + m.sigma_22[g, y, s, t]
            #             - m.sigma_25[g, y, s, t]
            #             == 0)

        # Candidate storage unit discharging power
        m.DISCHARGING_POWER_CANDIDATE_STORAGE = Constraint(m.G_C_STORAGE, m.Y, m.S, m.T,
                                                           rule=discharging_power_candidate_storage_rule)

        def energy_existing_storage_unit(_m, g, y, s, t):
            """Storage unit energy"""

            if t != m.T.last():
                return - m.sigma_15[g, y, s, t] + m.sigma_16[g, y, s, t] - m.zeta_1[g, y, s, t] + m.zeta_1[
                    g, y, s, t + 1] == 0

            else:
                return - m.sigma_15[g, y, s, t] + m.sigma_16[g, y, s, t] - m.zeta_1[g, y, s, t] - m.sigma_18[g, y, s] + \
                       m.sigma_19[g, y, s] == 0

        # Existing storage unit energy
        m.ENERGY_EXISTING_STORAGE = Constraint(m.G_E_STORAGE, m.Y, m.S, m.T, rule=energy_existing_storage_unit)

        def energy_candidate_storage_unit(_m, g, y, s, t):
            """Storage unit energy"""

            if t != m.T.last():
                return - m.sigma_15[g, y, s, t] + m.sigma_17[g, y, s, t] - m.zeta_1[g, y, s, t] + m.zeta_1[
                    g, y, s, t + 1] == 0

            else:
                return - m.sigma_15[g, y, s, t] + m.sigma_17[g, y, s, t] - m.zeta_1[g, y, s, t] - m.sigma_18[g, y, s] + \
                       m.sigma_19[g, y, s] == 0

        # Existing storage unit energy
        m.ENERGY_CANDIDATE_STORAGE = Constraint(m.G_C_STORAGE, m.Y, m.S, m.T, rule=energy_candidate_storage_unit)

        # def energy_output_existing_thermal_rule(_m, g, y, s, t):
        #     """Existing thermal generator energy output"""
        #
        #     # if y != m.Y.last():
        #     return m.DELTA[y] * m.RHO[y, s] * (m.C_MC[g, y] + ((m.EMISSIONS_RATE[g] - m.baseline[y]) * m.permit_price[y])) + m.zeta_2[g, y, s, t] == 0
        #     # else:
        #         # return m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * (m.C_MC[g, y] + ((m.EMISSIONS_RATE[g] - m.baseline[y]) * m.permit_price[y])) + m.zeta_2[g, y, s, t] == 0
        #
        # # Existing thermal generator energy output
        # m.ENERGY_EXISTING_THERMAL = Constraint(m.G_E_THERM, m.Y, m.S, m.T, rule=energy_output_existing_thermal_rule)

        # def energy_output_candidate_thermal_rule(_m, g, y, s, t):
        #     """Candidate thermal generator energy output"""
        #
        #     # if y != m.Y.last():
        #     return (m.DELTA[y] * m.RHO[y, s] * (m.C_MC[g, y] + ((m.EMISSIONS_RATE[g] - m.baseline[y]) * m.permit_price[y]))) + m.zeta_2[g, y, s, t] == 0
        #     # else:
        #     #     return (m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * (m.C_MC[g, y] + ((m.EMISSIONS_RATE[g] - m.baseline[y]) * m.permit_price[y]))) + m.zeta_2[g, y, s, t] == 0
        #
        # # Candidate thermal generator energy output
        # m.ENERGY_CANDIDATE_THERMAL = Constraint(m.G_C_THERM, m.Y, m.S, m.T, rule=energy_output_candidate_thermal_rule)
        #
        # def energy_output_existing_wind_rule(_m, g, y, s, t):
        #     """Existing wind generator energy output"""
        #
        #     # if y != m.Y.last():
        #     return (m.DELTA[y] * m.RHO[y, s] * m.C_MC[g, y]) + m.zeta_2[g, y, s, t] == 0
        #     # else:
        #     #     return (m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * m.C_MC[g, y]) + m.zeta_2[g, y, s, t] == 0
        #
        # # Existing wind generator energy output
        # m.ENERGY_EXISTING_WIND = Constraint(m.G_E_WIND, m.Y, m.S, m.T, rule=energy_output_existing_wind_rule)
        #
        # def energy_output_candidate_wind_rule(_m, g, y, s, t):
        #     """Candidate wind generator energy output"""
        #
        #     # if y != m.Y.last():
        #     return (m.DELTA[y] * m.RHO[y, s] * (m.C_MC[g, y] - (m.baseline[y] * m.permit_price[y]))) + m.zeta_2[g, y, s, t] == 0
        #     # else:
        #     #     return (m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * (m.C_MC[g, y] - (m.baseline[y] * m.permit_price[y]))) + m.zeta_2[g, y, s, t] == 0
        #
        # # Candidate wind generator energy output
        # m.ENERGY_CANDIDATE_WIND = Constraint(m.G_C_WIND, m.Y, m.S, m.T, rule=energy_output_candidate_wind_rule)
        #
        # def energy_output_existing_solar_rule(_m, g, y, s, t):
        #     """Existing solar generator energy output"""
        #
        #     # if y != m.Y.last():
        #     return (m.DELTA[y] * m.RHO[y, s] * m.C_MC[g, y]) + m.zeta_2[g, y, s, t] == 0
        #     # else:
        #     #     return (m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * m.C_MC[g, y]) + m.zeta_2[g, y, s, t] == 0
        #
        # # Existing wind generator energy output
        # m.ENERGY_EXISTING_SOLAR = Constraint(m.G_E_SOLAR, m.Y, m.S, m.T, rule=energy_output_existing_solar_rule)
        #
        # def energy_output_candidate_solar_rule(_m, g, y, s, t):
        #     """Candidate solar generator energy output"""
        #
        #     # if y != m.Y.last():
        #     return (m.DELTA[y] * m.RHO[y, s] * (m.C_MC[g, y] - (m.baseline[y] * m.permit_price[y]))) + m.zeta_2[g, y, s, t] == 0
        #     # else:
        #     #     return (m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * (m.C_MC[g, y] - (m.baseline[y] * m.permit_price[y]))) + m.zeta_2[g, y, s, t] == 0
        #
        # # Candidate wind generator energy output
        # m.ENERGY_CANDIDATE_SOLAR = Constraint(m.G_C_SOLAR, m.Y, m.S, m.T, rule=energy_output_candidate_solar_rule)

        # def energy_output_existing_storage_rule(_m, g, y, s, t):
        #     """Energy output for existing storage units"""
        #
        #     if y != m.Y.last():
        #         return (m.DELTA[y] * m.RHO[y, s] * m.C_MC[g, y]) + m.zeta_3[g, y, s, t] == 0
        #     else:
        #         return (m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * m.C_MC[g, y]) + m.zeta_3[g, y, s, t] == 0
        #
        # # Existing storage unit energy output
        # m.ENERGY_OUTPUT_EXISTING_STORAGE = Constraint(m.G_E_STORAGE, m.Y, m.S, m.T, rule=energy_output_existing_storage_rule)
        #
        # def energy_output_candidate_storage_rule(_m, g, y, s, t):
        #     """Energy output for candidate storage units"""
        #
        #     if y != m.Y.last():
        #         return (m.DELTA[y] * m.RHO[y, s] * m.C_MC[g, y]) + m.zeta_3[g, y, s, t] == 0
        #     else:
        #         return (m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * m.C_MC[g, y]) + m.zeta_3[g, y, s, t] == 0
        #
        # # Candidate storage unit energy output
        # m.ENERGY_OUTPUT_CANDIDATE_STORAGE = Constraint(m.G_C_STORAGE, m.Y, m.S, m.T, rule=energy_output_candidate_storage_rule)

        # def energy_output_hydro_rule(_m, g, y, s, t):
        #     """Energy output for hydro units"""
        #
        #     # if y != m.Y.last():
        #     return (m.DELTA[y] * m.RHO[y, s] * m.C_MC[g, y]) + m.zeta_2[g, y, s, t] == 0
        #     # else:
        #     #     return (m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * m.C_MC[g, y]) + m.zeta_2[g, y, s, t] == 0
        #
        # # Energy output from hydro units
        # m.ENERGY_HYDRO = Constraint(m.G_E_HYDRO, m.Y, m.S, m.T, rule=energy_output_hydro_rule)

        def load_shedding_power_rule(_m, z, y, s, t):
            """Load shedding power"""

            if y != m.Y.last():
                return - m.sigma_26[z, y, s, t] - m.lamb[z, y, s, t] + (m.DELTA[y] * m.RHO[y, s] * m.C_L[z]) == 0
            else:
                return - m.sigma_26[z, y, s, t] - m.lamb[z, y, s, t] + (
                        m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * m.C_L[z]) == 0

        # Load shedding power
        m.LOAD_SHEDDING_POWER = Constraint(m.Z, m.Y, m.S, m.T, rule=load_shedding_power_rule)

        # def load_shedding_energy_rule(_m, z, y, s, t):
        #     """Load shedding energy"""
        #
        #     # if y != m.Y.last():
        #     return (m.DELTA[y] * m.RHO[y, s] * m.C_L) + m.zeta_4[z, y, s, t] == 0
        #     # else:
        #     #     return (m.DELTA[y] * m.RHO[y, s] * (1 + (1 / m.INTEREST_RATE)) * m.C_L) + m.zeta_4[z, y, s, t] == 0
        #
        # # Load shedding energy
        # m.LOAD_SHEDDING_ENERGY = Constraint(m.Z, m.Y, m.S, m.T, rule=load_shedding_energy_rule)

        def powerflow_rule(_m, l, y, s, t):
            """Powerflow between adjacent NEM zones"""

            return (- m.sigma_27[l, y, s, t] + m.sigma_28[l, y, s, t]
                    + (m.INCIDENCE_MATRIX[l, self.g(l)] * m.lamb[self.g(l), y, s, t])
                    + (m.INCIDENCE_MATRIX[l, self.h(l)] * m.lamb[self.h(l), y, s, t])
                    == 0)

        # Powerflow
        m.POWERFLOW = Constraint(m.L, m.Y, m.S, m.T, rule=powerflow_rule)

        return m

    @staticmethod
    def define_objective(m):
        """Define dual problem objective"""

        # Dual objective function
        m.OBJECTIVE = Objective(expr=m.DUAL_OBJECTIVE_EXPRESSION, sense=maximize)

        return m

    def construct_model(self):
        """Construct dual program"""

        # Initialise model
        m = ConcreteModel()

        # Add component allowing dual variables to be imported
        m.dual = Suffix(direction=Suffix.IMPORT)

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


class MPPDCModel:
    def __init__(self, final_year, scenarios_per_year):
        self.components = CommonComponents(final_year, scenarios_per_year)
        self.primal = Primal(final_year, scenarios_per_year)
        self.dual = Dual(final_year, scenarios_per_year)

        # Solver options
        self.keepfiles = False
        self.solver_options = {}  # 'MIPGap': 0.0005, 'optimalitytarget': 2, 'simplex tolerances optimality': 1e-4
        self.opt = SolverFactory('cplex', solver_io='mps')

    @staticmethod
    def define_parameters(m):
        """Define MPPDC parameters"""

        def emissions_intensity_target_rule(_m, y):
            """Emissions intensity target for each year in model horizon"""

            # TODO: Update this with desired trajectory
            return float(0.7)

        # Emissions intensity target
        m.EMISSIONS_INTENSITY_TARGET = Param(m.Y, rule=emissions_intensity_target_rule)

        # Fixed emissions intensity baseline values
        m.FIXED_BASELINE = Param(m.Y, initialize=0, mutable=True)

        # Fixed permit price values
        m.FIXED_PERMIT_PRICE = Param(m.Y, initialize=0, mutable=True)

        # Average price in year prior to model start
        m.YEAR_AVERAGE_PRICE_0 = Param(initialize=40)

        # Strong duality constraint violation penalty
        m.STRONG_DUALITY_VIOLATION_PENALTY = Param(initialize=float(1e6))

        # Lower limit for scheme revenue in a given year
        m.SCHEME_REVENUE_LB = Param(initialize=float(-10e6))

        # Year at which yearly revenue neutrality constraint will be enforced
        m.TRANSITION_YEAR = Param(initialize=0, mutable=True)

        return m

    @staticmethod
    def define_variables(m):
        """Define MPPDC variables"""

        # Dummy objective function variable
        m.dummy = Var(within=NonNegativeReals, initialize=0)

        # Dummy variables used to minimise price differences between successive years
        m.z_p1 = Var(m.Y, within=NonNegativeReals, initialize=0)
        m.z_p2 = Var(m.Y, within=NonNegativeReals, initialize=0)

        # Dummy variables to ensure strong duality constraint feasibility
        m.sd_1 = Var(within=NonNegativeReals, initialize=0)
        m.sd_2 = Var(within=NonNegativeReals, initialize=0)

        return m

    @staticmethod
    def define_expressions(m):
        """Define MPPDC expressions"""

        # Change in price between successive intervals
        m.AVERAGE_PRICE_DIFFERENCE = Expression(expr=sum(m.z_p1[y] + m.z_p2[y] for y in m.Y))

        # Strong duality constraint violation
        m.STRONG_DUALITY_VIOLATION_COST = Expression(expr=(m.sd_1 + m.sd_2) * m.STRONG_DUALITY_VIOLATION_PENALTY)

        return m

    @staticmethod
    def define_constraints(m):
        """MPPDC constraints"""

        def strong_duality_rule(_m):
            """Strong duality constraint"""

            return m.TPV + m.sd_1 == m.DUAL_OBJECTIVE_EXPRESSION + m.sd_2

        # Strong duality constraint (primal objective = dual objective at optimality)
        m.STRONG_DUALITY = Constraint(rule=strong_duality_rule)

        def price_target_deviation_1_rule(_m, y):
            """Constraint computing absolute difference between prices in successive years"""

            if y == m.Y.first():
                return m.z_p1[y] >= m.YEAR_AVERAGE_PRICE[y] - m.YEAR_AVERAGE_PRICE_0
            else:
                return m.z_p1[y] >= m.YEAR_AVERAGE_PRICE[y] - m.YEAR_AVERAGE_PRICE[y - 1]

        # Emissions intensity deviation - 1
        m.PRICE_TARGET_DEV_1 = Constraint(m.Y, rule=price_target_deviation_1_rule)

        def price_target_deviation_2_rule(_m, y):
            """Constraint computing absolute difference between prices in successive years"""

            if y == m.Y.first():
                return m.z_p2[y] >= m.YEAR_AVERAGE_PRICE_0 - m.YEAR_AVERAGE_PRICE[y]
            else:
                return m.z_p2[y] >= m.YEAR_AVERAGE_PRICE[y - 1] - m.YEAR_AVERAGE_PRICE[y]

        # Emissions intensity deviation - 2
        m.PRICE_TARGET_DEV_2 = Constraint(m.Y, rule=price_target_deviation_2_rule)

        def total_scheme_revenue_non_negative_rule(_m):
            """Ensure that net scheme revenue is greater than 0 over model horizon"""

            return m.TOTAL_SCHEME_REVENUE >= 0

        # Total net scheme revenue non-negativity constraint
        m.TOTAL_SCHEME_REVENUE_NON_NEGATIVE_CONS = Constraint(rule=total_scheme_revenue_non_negative_rule)
        m.TOTAL_SCHEME_REVENUE_NON_NEGATIVE_CONS.deactivate()

        def total_scheme_revenue_neutral_rule(_m):
            """Ensure that net scheme revenue is equals 0 over model horizon"""

            return m.TOTAL_SCHEME_REVENUE == 0

        # Total net scheme revenue neutrality constraint
        m.TOTAL_NET_SCHEME_REVENUE_NEUTRAL_CONS = Constraint(rule=total_scheme_revenue_neutral_rule)
        m.TOTAL_NET_SCHEME_REVENUE_NEUTRAL_CONS.deactivate()

        def year_scheme_revenue_neutral_rule(_m, y):
            """Ensure that net scheme revenue in each year = 0 (equivalent to a REP scheme)"""

            return m.YEAR_SCHEME_REVENUE[y] == 0

        # Ensure scheme is revenue neutral in each year of model horizon (equivalent to a REP scheme)
        m.YEAR_NET_SCHEME_REVENUE_NEUTRAL_CONS = Constraint(m.Y, rule=year_scheme_revenue_neutral_rule)
        m.YEAR_NET_SCHEME_REVENUE_NEUTRAL_CONS.deactivate()

        def cumulative_scheme_revenue_lower_bound_rule(_m, y):
            """Ensure that cumulative scheme revenue each year is greater than some lower limit"""

            return sum(m.YEAR_SCHEME_REVENUE[j] for j in m.Y if j <= y) >= m.SCHEME_REVENUE_LB

        # Ensure cumulative scheme revenue is greater than or equal to some lower limit
        m.CUMULATIVE_NET_SCHEME_REVENUE_LB_CONS = Constraint(m.Y, rule=cumulative_scheme_revenue_lower_bound_rule)
        m.CUMULATIVE_NET_SCHEME_REVENUE_LB_CONS.deactivate()

        def transition_net_scheme_revenue_neutral_rule(_m):
            """Ensure revenue neutrality over transitional period"""

            if m.TRANSITION_YEAR.value >= m.Y.first():
                return sum(m.YEAR_SCHEME_REVENUE[y] for y in m.Y if y <= m.TRANSITION_YEAR.value) == 0

            else:
                return Constraint.Skip

        # Enforce net scheme revenue over transitional period = 0
        m.TRANSITION_NET_SCHEME_REVENUE_NEUTRAL_CONS = Constraint(rule=transition_net_scheme_revenue_neutral_rule)

        return m

    @staticmethod
    def define_objective(m):
        """MPPDC objective function"""

        # Price targeting objective
        m.OBJECTIVE = Objective(expr=m.AVERAGE_PRICE_DIFFERENCE + m.STRONG_DUALITY_VIOLATION_COST, sense=minimize)

        return m

    def construct_model(self):
        """Construct dual program"""

        # Initialise model
        m = ConcreteModel()

        # Add component allowing dual variables to be imported
        m.dual = Suffix(direction=Suffix.IMPORT)

        # Add common components
        m = self.components.define_sets(m)
        m = self.components.define_parameters(m)
        m = self.components.define_variables(m)

        # Primal problem elements
        m = self.primal.define_variables(m)
        m = self.primal.define_expressions(m)
        m = self.primal.define_constraints(m)

        # Dual problem elements
        m = self.dual.define_variables(m)
        m = self.dual.define_expressions(m)
        m = self.dual.define_constraints(m)

        # MPPDC problem parameters
        m = self.define_parameters(m)

        # MPPDC problem variables
        m = self.define_variables(m)

        # MPPDC problem expressions
        m = self.define_expressions(m)

        # MPPDC constraints
        m = self.define_constraints(m)

        # MPPDC problem objective
        m = self.define_objective(m)

        return m

    def solve_model(self, m):
        """Solve model"""

        # Solve model
        solve_status = self.opt.solve(m, tee=True, options=self.solver_options, keepfiles=self.keepfiles)

        # Log infeasible constraints if they exist
        log_infeasible_constraints(m)

        return m, solve_status


def setup_logger(name):
    """Setup logger to be used when running algorithm"""

    logging.basicConfig(filename=f'{name}.log', filemode='w',
                        format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)


def algorithm_logger(function, message, print_message=False):
    """Write message to logfile and optionally print output"""

    # Get logging object
    logger = logging.getLogger(__name__)

    # Message to write to logfile
    log_message = f'{function} - {message}'

    if print_message:
        print(log_message)
        logger.info(log_message)
    else:
        logger.info(log_message)


def get_max_absolute_difference(d1, d2):
    """Compute max absolute difference between two dictionaries"""

    # Check that keys are the same
    assert set(d1.keys()) == set(d2.keys()), f'Keys are not the same: {d1.keys()}, {d2.keys()}'

    # Absolute difference for each key
    abs_difference = {k: abs(d1[k] - d2[k]) for k in d1.keys()}

    # Max absolute difference
    max_abs_difference = max(abs_difference.values())

    return max_abs_difference


def extract_result(m, component_name):
    """Extract values associated with model components"""

    model_component = m.__getattribute__(component_name)

    if type(model_component) == pyomo.core.base.expression.IndexedExpression:
        return {k: model_component[k].expr() for k in model_component.keys()}

    elif type(model_component) == pyomo.core.base.expression.SimpleExpression:
        return model_component.expr()

    elif type(model_component) == pyomo.core.base.var.IndexedVar:
        return model_component.get_values()

    else:
        raise Exception(f'Unexpected model component: {component_name}')


def check_solution(m_p, m_d, elements):
    """Check solution

    Parameters
    ----------
    m_p : pyomo model
        Primal model

    m_d : pyomo model
        Dual model

    elements : dict
        primal and dual components to check
    """

    # Components
    p, d = m_p.__getattribute__(elements['primal']), m_d.__getattribute__(elements['dual'])

    # Common keys
    common_keys = set(p.keys()).intersection(set(d.keys()))

    # Absolute difference
    if elements['primal_var']:
        difference = {
            i: {'primal': p[i].value, 'dual': m_d.dual[d[i]], 'abs_diff': abs(abs(p[i].value) - abs(m_d.dual[d[i]]))}
            for i in common_keys}
    else:
        difference = {
            i: {'primal': m_p.dual[p[i]], 'dual': d[i].value, 'abs_diff': abs(abs(m_p.dual[p[i]]) - abs(d[i].value))}
            for i in common_keys}

    # Max difference over all keys
    max_difference = max([v['abs_diff'] for _, v in difference.items()])

    print(f"""Primal element: {elements['primal']}, Dual element: {elements[
        'dual']}, Max difference: {max_difference}, keys: {len(common_keys)}""")

    # Only retain elements where difference is non-negative
    non_negative_difference = {k: v for k, v in difference.items() if v['abs_diff'] > 0.1}

    return difference, max_difference, non_negative_difference


def transfer_primal_solution_to_mppdc(m_p, m_m, vars_to_fix):
    """Transfer solution from primal model to MPPDC model"""

    for v in vars_to_fix:
        # Get values obtained from primal model
        primal_values = m_p.__getattribute__(v).get_values()

        # Fix values in MPPDC model
        m_m.__getattribute__(v).set_values(primal_values)

        # Fix MPPDC values
        m_m.__getattribute__(v).fix()

    return m_m


def run_mppdc_fixed_policy(baselines, permit_prices, final_year, scenarios_per_year):
    """Run BAU case"""

    # Initialise object and model used to run MPPDC model
    mppdc = MPPDCModel(final_year, scenarios_per_year)
    m = mppdc.construct_model()

    # Fix permit prices and baselines
    for y in m.Y:
        m.permit_price[y].fix(permit_prices[y])
        m.baseline[y].fix(baselines[y])

    # Solve MPPDC model with fixed policy parameters
    m, status = mppdc.solve_model(m)

    # Results to extract
    result_keys = ['x_c', 'p', 'baseline', 'permit_price', 'lamb', 'YEAR_EMISSIONS', 'YEAR_EMISSIONS_INTENSITY',
                   'YEAR_SCHEME_REVENUE', 'TOTAL_SCHEME_REVENUE', 'YEAR_AVERAGE_PRICE']

    # Model results
    results = {k: extract_result(m, k) for k in result_keys}

    return results


def run_primal_fixed_policy(baselines, permit_prices, final_year, scenarios_per_year):
    """Run primal model with fixed policy parameters"""

    # Initialise object and model used to run primal model
    primal = Primal(final_year, scenarios_per_year)
    m = primal.construct_model()

    # Fix permit prices and baselines
    for y in m.Y:
        m.permit_price[y].fix(permit_prices[y])
        m.baseline[y].fix(baselines[y])

    # Solve primal model with fixed policy parameters
    m, status = primal.solve_model(m)

    # Results to extract
    result_keys = ['x_c', 'p', 'baseline', 'permit_price', 'YEAR_EMISSIONS', 'YEAR_EMISSIONS_INTENSITY',
                   'YEAR_SCHEME_REVENUE', 'TOTAL_SCHEME_REVENUE']

    # Model results
    results = {k: extract_result(m, k) for k in result_keys}

    # Add dual variable from power balance constraint
    results['PRICES'] = {k: m.dual[m.POWER_BALANCE[k]] for k in m.POWER_BALANCE.keys()}

    return results


def run_bau_case(output_dir, final_year, scenarios_per_year, mode='primal'):
    """Run business-as-usual case"""

    # Initialise object and model used to run primal model
    primal = Primal(final_year, scenarios_per_year)
    primal_dummy_model = primal.construct_model()

    # Baselines and permit prices = 0 for all years in model horizon
    baselines = {y: float(0) for y in primal_dummy_model.Y}
    permit_prices = {y: float(0) for y in primal_dummy_model.Y}

    if mode == 'primal':
        # Run primal BAU case
        results = run_primal_fixed_policy(baselines, permit_prices, final_year, scenarios_per_year)
        filename = 'primal_bau_results.pickle'

    elif mode == 'mppdc':
        # Run MPPDC BAU case
        results = run_mppdc_fixed_policy(baselines, permit_prices, final_year, scenarios_per_year)
        filename = 'mppdc_bau_results.pickle'

    else:
        raise Exception(f'Unexpected run mode: {mode}')

    # Save results
    with open(os.path.join(output_dir, filename), 'wb') as f:
        pickle.dump(results, f)

    return results


def get_permit_price_trajectory(primal, model, target_emissions_trajectory, baselines, initial_permit_prices,
                                permit_price_tol):
    """Run algorithm to identify sequence of permit prices that achieves a given emissions intensity trajectory"""

    # Name of function - used by logger
    function_name = 'get_permit_price_trajectory'

    # Initialise results container
    results = {'target_emissions_trajectory': target_emissions_trajectory, 'iteration_results': {}}

    # Initialise permit prices for each year in model horizon
    permit_prices = initial_permit_prices

    # Update emissions intensity baseline. Set baseline = target emissions intensity trajectory
    for y in model.Y:
        model.baseline[y].fix(baselines[y])

    # Iteration counter
    i = 1

    while True:
        algorithm_logger(function_name, f'Running iteration {i}', print_message=True)

        # Fix permit prices
        for y in model.Y:
            model.permit_price[y].fix(permit_prices[y])

        algorithm_logger(function_name, f'Model permit prices: {model.permit_price.get_values()}',
                         print_message=True)

        # Run model
        model, solver_status = primal.solve_model(model)
        algorithm_logger(function_name, 'Solved model')

        # Latest emissions intensities
        latest_emissions_intensities = extract_result(model, 'YEAR_EMISSIONS_INTENSITY')
        algorithm_logger(function_name, f'Emissions intensities: {latest_emissions_intensities}', print_message=True)

        # Compute difference between actual emissions intensity and target
        emissions_intensity_difference = {y: latest_emissions_intensities[y] - target_emissions_trajectory[y]
                                          for y in model.Y}
        algorithm_logger(function_name, f'Emissions intensity target difference: {emissions_intensity_difference}',
                         print_message=True)

        # Container for new permit prices
        new_permit_prices = {}

        # Update permit prices
        for y in model.Y:
            new_permit_prices[y] = permit_prices[y] + (emissions_intensity_difference[y] * 100)

            # Set equal to zero if permit price is less than 0
            if new_permit_prices[y] < 0:
                new_permit_prices[y] = 0

        algorithm_logger(function_name, f'Updated permit prices: {new_permit_prices}', print_message=True)

        # Compute max difference between new and old permit prices
        max_permit_price_difference = get_max_absolute_difference(new_permit_prices, permit_prices)

        # Copying results from model object into a dictionary
        result_keys = ['baseline', 'permit_price', 'YEAR_EMISSIONS', 'YEAR_EMISSIONS_INTENSITY', 'YEAR_SCHEME_REVENUE',
                       'TOTAL_SCHEME_REVENUE']

        # Store selected results
        results['iteration_results'][i] = {k: extract_result(model, k) for k in result_keys}

        # Check if the max difference is less than the tolerance
        if max_permit_price_difference < permit_price_tol:
            message = f"""
            Max permit price difference: {max_permit_price_difference}
            Tolerance: {permit_price_tol}
            Exiting algorithm
            """
            algorithm_logger(function_name, message, print_message=True)

            return model, results

        else:
            # update permit prices to use in next iteration
            permit_prices = new_permit_prices

            # Update iteration counter
            i += 1


def run_carbon_tax_case(output_dir, final_year, scenarios_per_year, target_emissions_trajectory, permit_price_tol):
    """Run carbon tax case (no refunding)"""

    # Initialise object and model used to run primal model
    primal = Primal(final_year, scenarios_per_year)
    primal_model = primal.construct_model()

    # Baselines = 0 for all years in model horizon
    baselines = {y: float(0) for y in primal_model.Y}

    # Initial permit prices (assume = 0 for all years)
    initial_permit_prices = {y: float(0) for y in primal_model.Y}

    # Run algorithm to identify permit price trajectory that achieve emissions trajectory target
    primal_model, permit_price_results = get_permit_price_trajectory(primal, primal_model, target_emissions_trajectory,
                                                                     baselines, initial_permit_prices, permit_price_tol)

    # Combine results into single dictionary
    results = {'permit_price_trajectory': permit_price_results}

    # Save results
    filename = 'carbon_tax_results.pickle'

    with open(os.path.join(output_dir, filename), 'wb') as f:
        pickle.dump(results, f)

    return results


def run_algorithm(output_dir, final_year, scenarios_per_year, target_emissions_trajectory, baseline_tol,
                  permit_price_tol, case):
    """Run price smoothing case where non-negative revenue is enforced"""

    # Name of function
    function_name = 'run_algorithm'

    # Model parameters
    parameters = {'final_year': final_year, 'scenarios_per_year': scenarios_per_year,
                  'target_emissions_trajectory': target_emissions_trajectory, 'baseline_tol': baseline_tol,
                  'permit_price_tol': permit_price_tol, 'case': case}

    algorithm_logger(function_name, f'Running algorithm with parameters: {parameters}', print_message=True)

    # Container for model results
    results = {'parameters': parameters, 'baseline_iteration': {}}

    # Construct primal model
    primal = Primal(final_year, scenarios_per_year)
    primal_model = primal.construct_model()
    algorithm_logger(function_name, f'Constructed primal model', print_message=True)

    # Construct MPPDC
    mppdc = MPPDCModel(final_year, scenarios_per_year)
    mppdc_model = mppdc.construct_model()
    algorithm_logger(function_name, f'Constructed MPPDC model', print_message=True)

    # Activate constraints required for case
    if case == 'price_smoothing_non_negative_revenue':
        mppdc_model.TOTAL_SCHEME_REVENUE_NON_NEGATIVE_CONS.activate()

    elif case == 'price_smoothing_neutral_revenue':
        mppdc_model.TOTAL_NET_SCHEME_REVENUE_NEUTRAL_CONS.activate()

    elif case == 'price_smoothing_neutral_revenue_lower_bound':
        mppdc_model.TOTAL_NET_SCHEME_REVENUE_NEUTRAL_CONS.activate()
        mppdc_model.CUMULATIVE_NET_SCHEME_REVENUE_LB_CONS.activate()

    elif case == 'rep':
        mppdc_model.YEAR_NET_SCHEME_REVENUE_NEUTRAL_CONS.activate()

    elif case == 'transition':
        # Set the transition year
        transition_year = 2017
        mppdc_model.TRANSITION_YEAR = 2017

        # Record transition year in model parameters dictionary
        results['parameters']['transition_year'] = transition_year

        # Activate year revenue neutrality requirement for all years
        mppdc_model.YEAR_NET_SCHEME_REVENUE_NEUTRAL_CONS.activate()

        # Deactivate year revenue neutrality requirement for years before transition year
        for y in range(2016, transition_year + 1):
            mppdc_model.YEAR_NET_SCHEME_REVENUE_NEUTRAL_CONS[y].deactivate()

        # Reconstruct and enforce revenue neutrality over the transition period
        mppdc_model.TRANSITION_NET_SCHEME_REVENUE_NEUTRAL_CONS.reconstruct()
        mppdc_model.TRANSITION_NET_SCHEME_REVENUE_NEUTRAL_CONS.activate()

    else:
        algorithm_logger(function_name, f'Unexpected case: {case}')
        raise Exception(f'Unexpected case: {case}')

    # Initialise baselines = target emissions intensity trajectory
    baselines = target_emissions_trajectory

    # Initialise permit prices = 0 for first iteration
    initial_permit_prices = {y: float(0) for y in primal_model.Y}

    # Iteration counter
    i = 1

    while True:
        algorithm_logger(function_name, f'Performing iteration {i}', print_message=True)

        # Initialise container for iteration results
        baseline_iteration = {'baselines': baselines}

        algorithm_logger(function_name, f'Fixing baselines {baselines}')
        for k, v in baselines.items():
            primal_model.baseline[k].fix(v)

        # Implement permit price trajectory algorithm
        algorithm_logger(function_name, f'Solving for permit price trajectory')
        primal_model, permit_price_results = get_permit_price_trajectory(primal, primal_model,
                                                                         target_emissions_trajectory,
                                                                         baselines, initial_permit_prices,
                                                                         permit_price_tol)

        algorithm_logger(function_name, f'Solved permit price trajectory: {permit_price_results}')

        # Save permit price trajectory results to dictionary
        baseline_iteration['permit_price_results'] = permit_price_results

        # Fix p, x_c, and baseline, and permit_price based on primal model solution
        mppdc_model = transfer_primal_solution_to_mppdc(primal_model, mppdc_model,
                                                        vars_to_fix=['p', 'x_c', 'permit_price', 'baseline'])
        algorithm_logger(function_name, f'Fixed primal variables')

        # Unfixing baseline
        mppdc_model.baseline.unfix()

        # Solving MPPDC
        mppdc_model, model_status = mppdc.solve_model(mppdc_model)
        algorithm_logger(function_name, f'Solved MPPDC model: {model_status}')

        # New baseline trajectory
        latest_baselines = mppdc_model.baseline.get_values()
        algorithm_logger(function_name, f'New baseline trajectory: {latest_baselines}')

        # Check max difference between baselines from this iteration and last
        max_baseline_difference = get_max_absolute_difference(latest_baselines, baselines)
        algorithm_logger(function_name,
                         f'Max baseline difference between successive iterations: {max_baseline_difference}')

        # Store baseline iteration results
        results['baseline_iteration'][i] = baseline_iteration

        # If max difference < tol stop; else update baseline and repeat procedure
        if max_baseline_difference < baseline_tol:

            message = f"""
            Max baseline difference between successive iterations: {max_baseline_difference}
            Difference less than tolerance: {baseline_tol}
            Terminating algorithm"""
            algorithm_logger(function_name, message, print_message=True)

            break

        else:
            # Update baselines
            baselines = latest_baselines

            # Update initial permit prices to be used in next iteration for permit price trajectory algorithm
            max_iteration = max(permit_price_results['iteration_results'].keys())
            initial_permit_prices = permit_price_results['iteration_results'][max_iteration]['permit_price']

            message = f"""
            Max baseline difference between successive iterations: {max_baseline_difference}
            Difference greater than tolerance: {baseline_tol}
            Updated baselines: {baselines}"""
            algorithm_logger(function_name, message, print_message=True)

            # Update iteration counter
            i += 1

    # Extract results from final MPPDC model
    result_keys = ['x_c', 'p', 'baseline', 'permit_price', 'lamb', 'YEAR_EMISSIONS', 'YEAR_EMISSIONS_INTENSITY',
                   'YEAR_SCHEME_REVENUE', 'TOTAL_SCHEME_REVENUE', 'YEAR_AVERAGE_PRICE']

    # Model results
    results['final_iteration'] = {k: extract_result(mppdc_model, k) for k in result_keys}

    # Filename
    filename = f'{case}_results.pickle'

    # Save results
    with open(os.path.join(output_dir, filename), 'wb') as f:
        pickle.dump(results, f)

    return results


if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(__file__))
    final_year = 2018
    scenarios_per_year = 2
    setup_logger('controller')

    # BAU case
    # primal_results = run_bau_case(output_dir, final_year, scenarios_per_year, mode='primal')
    # mppdc_results = run_bau_case(output_dir, final_year, scenarios_per_year, mode='mppdc')

    # Carbon tax
    target_emissions_intensities = {y: 0.7 for y in range(2016, final_year + 1)}
    # carbon_tax_results = run_carbon_tax_case(output_dir, final_year, scenarios_per_year, target_emissions_intensities,
    #                                          permit_price_tol=2)

    # # Non-negative scheme revenue
    # non_neg_revenue_results = run_algorithm(output_dir, final_year, scenarios_per_year, target_emissions_intensities,
    #                                         baseline_tol=0.05, permit_price_tol=10,
    #                                         case='price_smoothing_non_negative_revenue')
    #
    # # Revenue neutral scheme
    # neutral_revenue_results = run_algorithm(output_dir, final_year, scenarios_per_year, target_emissions_intensities,
    #                                         baseline_tol=0.05, permit_price_tol=10,
    #                                         case='price_smoothing_neutral_revenue')

    # Revenue neutral scheme with lower bound
    neutral_revenue_with_lb_results = run_algorithm(output_dir, final_year, scenarios_per_year,
                                                    target_emissions_intensities, baseline_tol=0.05,
                                                    permit_price_tol=10,
                                                    case='price_smoothing_neutral_revenue_lower_bound')

    # # Refunded emissions payment scheme
    # rep_results = run_algorithm(output_dir, final_year, scenarios_per_year, target_emissions_intensities,
    #                             baseline_tol=0.05, permit_price_tol=10,
    #                             case='rep')
    #
    # # Transitional scheme
    # transition_results = run_algorithm(output_dir, final_year, scenarios_per_year, target_emissions_intensities,
    #                                    baseline_tol=0.05, permit_price_tol=10,
    #                                    case='transition')
