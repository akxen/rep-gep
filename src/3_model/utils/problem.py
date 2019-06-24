import os
import time
from math import ceil
from collections import OrderedDict

import pandas as pd

from pyomo.environ import *
from pyomo.core.expr import current as EXPR
from pyomo.util.infeasible import log_infeasible_constraints

import matplotlib.pyplot as plt

from components import BaseComponents


class UnitCommitmentModel(BaseComponents):
    def __init__(self, raw_data_dir, data_dir, input_traces_dir):
        # Inherit model data and common model components
        super().__init__(raw_data_dir, data_dir, input_traces_dir)

    def define_parameters(self, m):
        """Define model parameters"""

        # Model year - must update each time model is run
        m.YEAR = Param(initialize=2016, mutable=True)

        # Year indicator - set = 1 for all years <= m.YEAR for years in model horizon, otherwise = 0
        m.YEAR_INDICATOR = Param(m.I, initialize=0, within=Binary, mutable=True)

        # Capacity factor wind (must be updated each time model is run)
        m.Q_WIND = Param(m.B, m.T, initialize=0, mutable=True)

        # Capacity factor solar, for given technology j (must be updated each time model is run)
        m.Q_SOLAR = Param(m.Z, m.G_C_SOLAR_TECHNOLOGIES, m.T, initialize=0, mutable=True)

        # Zone demand (must be updated each time model is run)
        m.DEMAND = Param(m.Z, m.T, initialize=0, mutable=True)

        # Duration of operating scenario (must be updated each time model is run)
        m.SCENARIO_DURATION = Param(initialize=0, mutable=True)

        # TODO: If unit retires p0 should be set to 0. Handle when updating parameters. Consider baseload state also.

        # Unit availability indicator (indicates if unit is available / not retired). Must be set each time model run.
        m.AVAILABILITY_INDICATOR = Param(m.G, initialize=1, within=Binary, mutable=True)

        # Power output in interval prior to model start (handle when updating parameters)
        m.P0 = Param(m.G, mutable=True, within=NonNegativeReals, initialize=0)

        # Energy in battering in interval prior to model start (assume battery initially completely discharged)
        m.Y0 = Param(m.G_C_STORAGE, initialize=0)

        # Historic output for existing hydro generators
        m.P_HYDRO_HISTORIC = Param(m.G_E_HYDRO, m.T, initialize=0, mutable=True)

        # Min state of charge for storage unit at end of operating scenario (assume = 0)
        m.STORAGE_INTERVAL_END_MIN_ENERGY = Param(m.G_C_STORAGE, initialize=0)

        # Value of lost-load [$/MWh]
        m.C_LOST_LOAD = Param(initialize=float(1e4), mutable=True)

        # Fixed emissions intensity baseline [tCO2/MWh]
        m.FIXED_BASELINE = Param(initialize=0, mutable=True)

        # Fixed permit price [$/tCO2]
        m.FIXED_PERMIT_PRICE = Param(initialize=0, mutable=True)

        # Fixed capacities for candidate units
        m.FIXED_X_C = Param(m.G_C_WIND.union(m.G_C_SOLAR).union(m.G_C_STORAGE), m.I, initialize=0, mutable=True)

        # Fixed integer variables for candidate thermal unit discrete sizing options
        m.FIXED_D = Param(m.G_C_THERM, m.I, m.G_C_THERM_SIZE_OPTIONS, initialize=0, mutable=True)

        def startup_cost_rule(m, g):
            """
            Startup costs for existing and candidate thermal units

            Note: costs are normalised by plant capacity e.g. $/MW
            """

            if g in m.G_E_THERM:
                # Startup cost for existing thermal units
                startup_cost = (self.existing_units.loc[g, ('PARAMETERS', 'SU_COST_WARM')]
                                / self.existing_units.loc[g, ('PARAMETERS', 'REG_CAP')])

            elif g in m.G_C_THERM:
                # Startup cost for candidate thermal units
                startup_cost = self.candidate_units.loc[g, ('PARAMETERS', 'SU_COST_WARM_MW')]

            else:
                # Assume startup cost = 0 for all solar, wind, hydro generators
                startup_cost = 0

            # Shutdown cost cannot be negative
            assert startup_cost >= 0, 'Negative startup cost'

            return float(startup_cost)

        # Generator startup costs - per MW
        m.C_SU_MW = Param(m.G, rule=startup_cost_rule)

        def shutdown_cost_rule(m, g):
            """
            Shutdown costs for existing and candidate thermal units

            Note: costs are normalised by plant capacity e.g. $/MW

            No data for shutdown costs, so assume it is half the value of
            startup cost for given generator.
            """

            if g in m.G_E_THERM:
                # Shutdown cost for existing thermal units
                shutdown_cost = (self.existing_units.loc[g, ('PARAMETERS', 'SU_COST_WARM')]
                                 / self.existing_units.loc[g, ('PARAMETERS', 'REG_CAP')])

            elif g in m.G_C_THERM:
                # Shutdown cost for candidate thermal units
                shutdown_cost = self.candidate_units.loc[g, ('PARAMETERS', 'SU_COST_WARM_MW')]

            else:
                # Assume shutdown cost = 0 for all solar, wind, hydro generators
                shutdown_cost = 0

            # Shutdown cost cannot be negative
            assert shutdown_cost >= 0, 'Negative shutdown cost'

            return float(shutdown_cost)

        # Generator shutdown costs - per MW
        m.C_SD_MW = Param(m.G, rule=shutdown_cost_rule)

        def startup_ramp_rate_rule(m, g):
            """Startup ramp-rate (MW)"""

            if g in m.G_E_THERM:
                # Startup ramp-rate for existing thermal generators
                startup_ramp = self.existing_units.loc[g, ('PARAMETERS', 'RR_STARTUP')]

            elif g in m.G_C_THERM:
                # Startup ramp-rate for candidate thermal generators
                startup_ramp = self.candidate_units.loc[g, ('PARAMETERS', 'RR_STARTUP')]

            else:
                raise Exception(f'Unexpected generator {g}')

            return float(startup_ramp)

        # Startup ramp-rate for existing and candidate thermal generators
        m.RR_SU = Param(m.G_E_THERM.union(m.G_C_THERM), rule=startup_ramp_rate_rule)

        def shutdown_ramp_rate_rule(m, g):
            """Shutdown ramp-rate (MW)"""

            if g in m.G_E_THERM:
                # Shutdown ramp-rate for existing thermal generators
                shutdown_ramp = self.existing_units.loc[g, ('PARAMETERS', 'RR_SHUTDOWN')]

            elif g in m.G_C_THERM:
                # Shutdown ramp-rate for candidate thermal generators
                shutdown_ramp = self.candidate_units.loc[g, ('PARAMETERS', 'RR_SHUTDOWN')]

            else:
                raise Exception(f'Unexpected generator {g}')

            return float(shutdown_ramp)

        # Shutdown ramp-rate for existing and candidate thermal generators
        m.RR_SD = Param(m.G_E_THERM.union(m.G_C_THERM), rule=shutdown_ramp_rate_rule)

        def ramp_rate_up_rule(m, g):
            """Ramp-rate up (MW/h) - when running"""

            if g in m.G_E_THERM:
                # Ramp-rate up for existing generators
                ramp_up = self.existing_units.loc[g, ('PARAMETERS', 'RR_UP')]

            elif g in m.G_C_THERM:
                # Ramp-rate up for candidate generators
                ramp_up = self.candidate_units.loc[g, ('PARAMETERS', 'RR_UP')]

            else:
                raise Exception(f'Unexpected generator {g}')

            return float(ramp_up)

        # Ramp-rate up (normal operation)
        m.RR_UP = Param(m.G_E_THERM.union(m.G_C_THERM), rule=ramp_rate_up_rule)

        def ramp_rate_down_rule(m, g):
            """Ramp-rate down (MW/h) - when running"""

            if g in m.G_E_THERM:
                # Ramp-rate down for existing generators
                ramp_down = self.existing_units.loc[g, ('PARAMETERS', 'RR_DOWN')]

            elif g in m.G_C_THERM:
                # Ramp-rate down for candidate generators
                ramp_down = self.candidate_units.loc[g, ('PARAMETERS', 'RR_DOWN')]

            else:
                raise Exception(f'Unexpected generator {g}')

            return float(ramp_down)

        # Ramp-rate down (normal operation)
        m.RR_DOWN = Param(m.G_E_THERM.union(m.G_C_THERM), rule=ramp_rate_down_rule)

        def existing_generator_registered_capacities_rule(m, g):
            """Registered capacities of existing generators"""

            return float(self.existing_units.loc[g, ('PARAMETERS', 'REG_CAP')])

        # Registered capacities of existing generators
        m.EXISTING_GEN_REG_CAP = Param(m.G_E, rule=existing_generator_registered_capacities_rule)

        def emissions_intensity_rule(m, g):
            """Emissions intensity (tCO2/MWh)"""

            if g in m.G_E_THERM:
                # Emissions intensity
                emissions = float(self.existing_units.loc[g, ('PARAMETERS', 'EMISSIONS')])

            elif g in m.G_C_THERM:
                # Emissions intensity
                emissions = float(self.candidate_units.loc[g, ('PARAMETERS', 'EMISSIONS')])

            else:
                # Set emissions intensity = 0 for all solar, wind, hydro, and storage units
                emissions = float(0)

            return emissions

        # Emissions intensities for all generators
        m.EMISSIONS_RATE = Param(m.G.union(m.G_C_STORAGE), rule=emissions_intensity_rule)

        def min_power_output_proportion_rule(m, g):
            """Minimum generator power output as a proportion of maximum output"""

            if g in m.G_E_THERM:
                # Minimum power output for existing generators
                min_output = (self.existing_units.loc[g, ('PARAMETERS', 'MIN_GEN')] /
                              self.existing_units.loc[g, ('PARAMETERS', 'REG_CAP')])

            elif g in m.G_E_HYDRO:
                # Set minimum power output for existing hydro generators = 0
                min_output = 0

            elif g in m.G_C_THERM:
                # Minimum power output for candidate thermal generators
                min_output = self.candidate_units.loc[g, ('PARAMETERS', 'MIN_GEN_PERCENT')] / 100

            else:
                # Minimum power output = 0
                min_output = 0

            return float(min_output)

        # Minimum power output (as a proportion of max capacity) for existing and candidate thermal generators
        m.P_MIN_PROP = Param(m.G, rule=min_power_output_proportion_rule)

        def thermal_unit_discrete_size_rule(m, g, n):
            """Possible discrete sizes for candidate thermal units"""

            # Discrete sizes available for candidate thermal unit investment
            options = {0: 0, 1: 100, 2: 200, 3: 400}

            return float(options[n])

        # Candidate thermal unit size options
        m.X_C_THERM_SIZE = Param(m.G_C_THERM, m.G_C_THERM_SIZE_OPTIONS, rule=thermal_unit_discrete_size_rule)

        def initial_on_state_rule(m, g):
            """Defines which units should be on in period preceding model start"""

            if g in m.G_THERM_SLOW:
                return float(1)
            else:
                return float(0)

        # Initial on-state rule - assumes slow-start (effectively baseload) units are on in period prior to model start
        m.U0 = Param(m.G_E_THERM.union(m.G_C_THERM), within=Binary, mutable=True, rule=initial_on_state_rule)

        def marginal_cost_rule(m, g):
            """Marginal costs for existing and candidate generators

            Note: Marginal costs for existing and candidate thermal plant depend on fuel costs, which are time
            varying. Therefore marginal costs for thermal plant must be updated each time the model is run.
            These costs have been initialised to zero.
            """

            if (g in m.G_E_THERM) or (g in m.G_C_THERM):
                #  Initialise marginal cost for existing and candidate thermal plant = 0
                marginal_cost = 1

            elif (g in m.G_E_WIND) or (g in m.G_E_SOLAR) or (g in m.G_E_HYDRO):
                # Marginal cost = VOM cost for wind and solar generators
                marginal_cost = self.existing_units.loc[g, ('PARAMETERS', 'VOM')]

            elif (g in m.G_C_WIND) or (g in m.G_C_SOLAR):
                # Marginal cost = VOM cost for wind and solar generators
                marginal_cost = self.candidate_units.loc[g, ('PARAMETERS', 'VOM')]

            elif g in m.G_C_STORAGE:
                # Assume marginal cost = VOM cost of typical hydro generator (7 $/MWh)
                marginal_cost = 7

            else:
                raise Exception(f'Unexpected generator: {g}')

            assert marginal_cost >= 0, 'Cannot have negative marginal cost'

            return float(marginal_cost)

        # Marginal costs for all generators (must be updated each time model is run)
        m.C_MC = Param(m.G.union(m.G_C_STORAGE), rule=marginal_cost_rule, mutable=True)

        def battery_efficiency_rule(m, g):
            """Battery efficiency"""

            return float(self.battery_properties.loc[g, 'CHARGE_EFFICIENCY'])

        # Battery efficiency
        m.BATTERY_EFFICIENCY = Param(m.G_C_STORAGE, rule=battery_efficiency_rule)

        def network_incidence_matrix_rule(m, l, z):
            """Incidence matrix describing connections between adjacent NEM zones"""

            # Network incidence matrix
            df = self._get_network_incidence_matrix()

            return float(df.loc[l, z])

        # Network incidence matrix
        m.INCIDENCE_MATRIX = Param(m.L, m.Z, rule=network_incidence_matrix_rule)

        def minimum_region_up_reserve_rule(m, r):
            """Minimum upward reserve rule"""

            # Minimum upward reserve for region
            up_reserve = self.minimum_reserve_levels.loc[r, 'MINIMUM_RESERVE_LEVEL']

            return float(up_reserve)

        # Minimum upward reserve
        m.RESERVE_UP = Param(m.R, rule=minimum_region_up_reserve_rule)

        def powerflow_min_rule(m, l):
            """Minimum powerflow over network link"""

            return float(-self.powerflow_limits[l]['reverse'] * 100)

        # Lower bound for powerflow over link
        m.POWERFLOW_MIN = Param(m.L_I, rule=powerflow_min_rule)

        def powerflow_max_rule(m, l):
            """Maximum powerflow over network link"""

            return float(self.powerflow_limits[l]['forward'] * 100)

        # Lower bound for powerflow over link
        m.POWERFLOW_MAX = Param(m.L_I, rule=powerflow_max_rule)

        return m

    @staticmethod
    def define_variables(m):
        """Define model variables"""

        # Power output above minimum dispatchable level of output [MW]
        m.p = Var(m.G, m.T, within=NonNegativeReals, initialize=0)

        # Startup state variable
        m.v = Var(m.G_E_THERM.union(m.G_C_THERM), m.T, within=Binary, initialize=0)

        # Shutdown state variable
        m.w = Var(m.G_E_THERM.union(m.G_C_THERM), m.T, within=Binary, initialize=0)

        # On-state variable
        m.u = Var(m.G_E_THERM.union(m.G_C_THERM), m.T, within=Binary, initialize=1)

        # Upward reserve allocation [MW]
        m.r_up = Var(m.G_E_THERM.union(m.G_C_THERM).union(m.G_C_STORAGE), m.T, within=NonNegativeReals, initialize=0)

        # Storage unit charging (power in) [MW]
        m.p_in = Var(m.G_C_STORAGE, m.T, within=NonNegativeReals, initialize=0)

        # Storage unit discharging (power out) [MW]
        m.p_out = Var(m.G_C_STORAGE, m.T, within=NonNegativeReals, initialize=0)

        # Energy in storage unit [MWh]
        m.y = Var(m.G_C_STORAGE, m.T, within=NonNegativeReals, initialize=0)

        # Powerflow between NEM zones [MW]
        m.p_flow = Var(m.L, m.T, initialize=0)

        # Lost load - up [MW]
        m.p_lost_up = Var(m.Z, m.T, within=NonNegativeReals, initialize=0)

        # Lost load - down [MW]
        m.p_lost_down = Var(m.Z, m.T, within=NonNegativeReals, initialize=0)

        # Variables to be determined in master program (will be fixed in sub-problems)
        # ----------------------------------------------------------------------------
        # Capacity of candidate units (defined for all years in model horizon)
        m.x_c = Var(m.G_C_WIND.union(m.G_C_SOLAR).union(m.G_C_STORAGE), m.I, within=NonNegativeReals, initialize=0)

        # Binary variable used to determine size of candidate thermal units
        m.d = Var(m.G_C_THERM, m.I, m.G_C_THERM_SIZE_OPTIONS, within=NonNegativeReals, initialize=0)

        # Emissions intensity baseline [tCO2/MWh] (must be fixed each time model is run)
        m.baseline = Var(initialize=0)

        # Permit price [$/tCO2] (must be fixed each time model is run)
        m.permit_price = Var(initialize=0)

        return m

    @staticmethod
    def define_expressions(m):
        """Define model expressions"""

        def capacity_sizing_rule(m, g, i):
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
        m.X_C = Expression(m.G_C, m.I, rule=capacity_sizing_rule)

        def max_generator_power_output_rule(m, g):
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
                return sum(m.X_C[g, i] * m.YEAR_INDICATOR[i] for i in m.I)

            else:
                raise Exception(f'Unexpected generator: {g}')

        # Maximum power output for existing and candidate units (must be updated each time model is run)
        m.P_MAX = Expression(m.G, rule=max_generator_power_output_rule)

        def min_power_output_rule(m, g):
            """
            Minimum generator output in MW

            Note: Min power output can be a function of max capacity which is only known ex-post for candidate
            generators.
            """

            return m.P_MIN_PROP[g] * m.P_MAX[g]

        # Expression for minimum power output
        m.P_MIN = Expression(m.G, rule=min_power_output_rule)

        def total_power_output_rule(m, g, t):
            """Total power output [MW]"""

            if g in m.G_THERM_QUICK:
                if t < m.T.last():
                    return ((m.P_MIN[g] * (m.u[g, t] + m.v[g, t + 1])) + m.p[g, t]) * m.AVAILABILITY_INDICATOR[g]
                else:
                    return ((m.P_MIN[g] * m.u[g, t]) + m.p[g, t]) * m.AVAILABILITY_INDICATOR[g]

            # Define startup and shutdown trajectories for 'slow' generators
            elif g in m.G_THERM_SLOW:
                # Startup duration
                SU_D = ceil(m.P_MIN[g].expr() / m.RR_SU[g])

                # Startup power output trajectory increment
                ramp_up_increment = m.P_MIN[g].expr() / SU_D

                # Startup power output trajectory
                P_SU = OrderedDict({i + 1: ramp_up_increment * i for i in range(0, SU_D + 1)})

                # Shutdown duration
                SD_D = ceil(m.P_MIN[g].expr() / m.RR_SD[g])

                # Shutdown power output trajectory increment
                ramp_down_increment = m.P_MIN[g].expr() / SD_D

                # Shutdown power output trajectory
                P_SD = OrderedDict({i + 1: m.P_MIN[g].expr() - (ramp_down_increment * i) for i in range(0, SD_D + 1)})

                if t < m.T.last():
                    return (((m.P_MIN[g] * (m.u[g, t] + m.v[g, t + 1])) + m.p[g, t]
                             + sum(P_SU[i] * m.v[g, t - i + SU_D + 2] if t - i + SU_D + 2 in m.T else 0 for i in
                                   range(1, SU_D + 1))
                             + sum(P_SD[i] * m.w[g, t - i + 2] if t - i + 2 in m.T else 0 for i in range(2, SD_D + 2))
                             ) * m.AVAILABILITY_INDICATOR[g])
                else:
                    return (((m.P_MIN[g] * m.u[g, t]) + m.p[g, t]
                             + sum(P_SU[i] * m.v[g, t - i + SU_D + 2] if t - i + SU_D + 2 in m.T else 0 for i in
                                   range(1, SU_D + 1))
                             + sum(P_SD[i] * m.w[g, t - i + 2] if t - i + 2 in m.T else 0 for i in range(2, SD_D + 2))
                             ) * m.AVAILABILITY_INDICATOR[g])

            # Remaining generators with no startup / shutdown directory defined
            else:
                return (m.P_MIN[g] + m.p[g, t]) * m.AVAILABILITY_INDICATOR[g]

        # Total power output [MW]
        m.P_TOTAL = Expression(m.G, m.T, rule=total_power_output_rule)

        def generator_energy_output_rule(m, g, t):
            """
            Total generator energy output [MWh]

            No information regarding generator dispatch level in period prior to
            t=0. Assume that all generators are at their minimum dispatch level.
            """

            if t != m.T.first():
                return (m.P_TOTAL[g, t - 1] + m.P_TOTAL[g, t]) / 2

            # Else, first interval (t-1 will be out of range)
            else:
                return (m.P0[g] + m.P_TOTAL[g, t]) / 2

        # Energy output for a given generator
        m.ENERGY = Expression(m.G, m.T, rule=generator_energy_output_rule)

        def storage_unit_energy_output_rule(m, g, t):
            """Energy output from storage units"""

            if t != m.T.first():
                return (m.p_out[g, t] + m.p_out[g, t - 1]) / 2

            else:
                return m.p_out[g, t] / 2

        # Energy output for a given storage unit
        m.ENERGY_OUT = Expression(m.G_C_STORAGE, m.T, rule=storage_unit_energy_output_rule)

        def storage_unit_energy_input_rule(m, g, t):
            """Energy input (charging) from storage units"""

            if t != m.T.first():
                return (m.p_in[g, t] + m.p_in[g, t - 1]) / 2

            else:
                return m.p_in[g, t] / 2

        # Energy input for a given storage unit
        m.ENERGY_IN = Expression(m.G_C_STORAGE, m.T, rule=storage_unit_energy_input_rule)

        def thermal_startup_cost_rule(m, g):
            """Startup cost for existing and candidate thermal generators"""

            return m.C_SU_MW[g] * m.P_MAX[g]

        # Startup cost - absolute cost [$]
        m.C_SU = Expression(m.G_E_THERM.union(m.G_C_THERM), rule=thermal_startup_cost_rule)

        def thermal_shutdown_cost_rule(m, g):
            """Startup cost for existing and candidate thermal generators"""
            # TODO: For now set shutdown cost = 0
            return m.C_SD_MW[g] * 0

        # Startup cost - absolute cost [$]
        m.C_SD = Expression(m.G_E_THERM.union(m.G_C_THERM), rule=thermal_shutdown_cost_rule)

        def thermal_operating_costs_rule(m):
            """Cost to operate existing and candidate thermal units"""

            return (sum((m.C_MC[g] + (m.EMISSIONS_RATE[g] - m.baseline) * m.permit_price) * m.ENERGY[g, t]
                        + (m.C_SU[g] * m.v[g, t]) + (m.C_SD[g] * m.w[g, t])
                        for g in m.G_E_THERM.union(m.G_C_THERM) for t in m.T))

        # Existing and candidate thermal unit operating costs
        m.C_OP_THERM = Expression(rule=thermal_operating_costs_rule)

        def hydro_operating_costs_rule(m):
            """Cost to operate existing hydro generators"""

            return sum(m.C_MC[g] * m.ENERGY[g, t] for g in m.G_E_HYDRO for t in m.T)

        # Existing hydro unit operating costs (no candidate hydro generators)
        m.C_OP_HYDRO = Expression(rule=hydro_operating_costs_rule)

        def solar_operating_costs_rule(m):
            """Cost to operate existing and candidate solar units"""

            return (sum(m.C_MC[g] * m.ENERGY[g, t] for g in m.G_E_SOLAR for t in m.T)
                    + sum((m.C_MC[g] - m.baseline * m.permit_price) * m.ENERGY[g, t] for g in m.G_C_SOLAR for t in m.T))

        # Existing and candidate solar unit operating costs (only candidate solar eligible for credits)
        m.C_OP_SOLAR = Expression(rule=solar_operating_costs_rule)

        def wind_operating_costs_rule(m):
            """Cost to operate existing and candidate wind generators"""

            return (sum(m.C_MC[g] * m.ENERGY[g, t] for g in m.G_E_WIND for t in m.T)
                    + sum((m.C_MC[g] - m.baseline * m.permit_price) * m.ENERGY[g, t] for g in m.G_C_WIND for t in m.T))

        # Existing and candidate solar unit operating costs (only candidate solar eligible for credits)
        m.C_OP_WIND = Expression(rule=wind_operating_costs_rule)

        def storage_unit_charging_cost_rule(m):
            """Cost to charge storage unit"""

            return sum(m.C_MC[g] * m.ENERGY_IN[g, t] for g in m.G_C_STORAGE for t in m.T)

        # Charging cost rule - no subsidy received when purchasing energy
        m.C_OP_STORAGE_CHARGING = Expression(rule=storage_unit_charging_cost_rule)

        def storage_unit_discharging_cost_rule(m):
            """
            Cost to charge storage unit

            Note: If storage units are included in the scheme this could create an undesirable outcome. Units
            would be subsidised for each MWh they generate. Therefore they could be incentivised to continually charge
            and then immediately discharge in order to receive the subsidy. For now assume the storage units are not
            eligible to receive a subsidy for each MWh under the policy.
            """
            # - (m.baseline * m.permit_price))
            return sum(m.C_MC[g] * m.ENERGY_OUT[g, t] for g in m.G_C_STORAGE for t in m.T)

        # Discharging cost rule - assumes storage units are eligible under REP scheme
        m.C_OP_STORAGE_DISCHARGING = Expression(rule=storage_unit_discharging_cost_rule)

        # Candidate storage unit operating costs
        m.C_OP_STORAGE = Expression(expr=m.C_OP_STORAGE_CHARGING + m.C_OP_STORAGE_DISCHARGING)

        def lost_load_cost_rule(m):
            """Value of lost-load"""

            return sum((m.p_lost_up[z, t] + m.p_lost_down[z, t]) * m.C_LOST_LOAD for z in m.Z for t in m.T)

        # Total cost of lost-load
        m.C_OP_LOST_LOAD = Expression(rule=lost_load_cost_rule)

        def total_operating_cost_rule(m):
            """Total operating cost"""

            return m.C_OP_THERM + m.C_OP_HYDRO + m.C_OP_SOLAR + m.C_OP_WIND + m.C_OP_STORAGE + m.C_OP_LOST_LOAD

        # Total operating cost
        m.C_OP_TOTAL = Expression(rule=total_operating_cost_rule)

        def storage_unit_energy_capacity_rule(m, g):
            """Energy capacity depends on installed capacity (variable in master problem)"""

            return sum(m.x_c[g, i] * m.YEAR_INDICATOR[i] for i in m.I)

        # Capacity of storage unit [MWh]
        m.STORAGE_UNIT_ENERGY_CAPACITY = Expression(m.G_C_STORAGE, rule=storage_unit_energy_capacity_rule)

        def storage_unit_max_energy_interval_end_rule(m, g):
            """Maximum energy at end of storage interval. Assume equal to unit capacity."""

            return m.STORAGE_UNIT_ENERGY_CAPACITY[g]

        # Max state of charge for storage unit at end of operating scenario (assume = unit capacity)
        m.STORAGE_INTERVAL_END_MAX_ENERGY = Expression(m.G_C_STORAGE, rule=storage_unit_max_energy_interval_end_rule)

        def storage_unit_max_power_out_rule(m, g):
            """
            Maximum discharging power of storage unit - set equal to energy capacity. Assumes
            storage unit can completely discharge in 1 hour
            """

            return m.STORAGE_UNIT_ENERGY_CAPACITY[g]

        # Max MW out of storage device - discharging
        m.P_STORAGE_MAX_OUT = Expression(m.G_C_STORAGE, rule=storage_unit_max_power_out_rule)

        def storage_unit_max_power_in_rule(m, g):
            """
            Maximum charging power of storage unit - set equal to energy capacity. Assumes
            storage unit can completely charge in 1 hour
            """

            return m.STORAGE_UNIT_ENERGY_CAPACITY[g]

        # Max MW into storage device - charging
        m.P_STORAGE_MAX_IN = Expression(m.G_C_STORAGE, rule=storage_unit_max_power_in_rule)

        return m

    def define_constraints(self, m):
        """Define model constraints"""

        # Fix emissions intensity baseline to given value in sub-problems
        m.FIXED_BASELINE_CONS = Constraint(expr=m.baseline == m.FIXED_BASELINE)

        # Fix permit price to given value in sub-problems
        m.FIXED_PERMIT_PRICE_CONS = Constraint(expr=m.permit_price == m.FIXED_PERMIT_PRICE)

        def fixed_capacity_continuous_rule(m, g, i):
            """Fixed installed capacities"""

            return m.FIXED_X_C[g, i] == m.x_c[g, i]

        # Fixed installed capacity for candidate units - continuous sizing
        m.FIXED_CAPACITY_CONT = Constraint(m.G_C_SOLAR.union(m.G_C_WIND).union(m.G_C_STORAGE), m.I,
                                           rule=fixed_capacity_continuous_rule)

        def fixed_capacity_discrete_rule(m, g, i, n):
            """Fix installed capacity for discrete unit sizing"""
            # TODO: Need to fix how binary variables for discrete sizing are updated in subproblem
            return m.FIXED_D[g, i, n] == m.d[g, i, n]

        # Fixed installed capacity for candidate units - discrete sizing
        m.FIXED_CAPACITY_DISC = Constraint(m.G_C_THERM, m.I, m.G_C_THERM_SIZE_OPTIONS,
                                           rule=fixed_capacity_discrete_rule)

        def upward_power_reserve_rule(m, r, t):
            """Upward reserve constraint"""

            # NEM region-zone map
            region_zone_map = self._get_nem_region_zone_map()

            # Mapping describing the zone to which each generator is assigned
            gen_zone_map = self._get_generator_zone_map()

            # Existing and candidate thermal gens + candidate storage units
            gens = m.G_E_THERM.union(m.G_C_THERM).union(m.G_C_STORAGE)

            return sum(m.r_up[g, t] for g in gens if gen_zone_map.loc[g] in region_zone_map.loc[r]) >= m.RESERVE_UP[r]

        # Upward power reserve rule for each NEM region
        m.UPWARD_POWER_RESERVE = Constraint(m.R, m.T, rule=upward_power_reserve_rule)

        def operating_state_logic_rule(m, g, t):
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
        m.OPERATING_STATE = Constraint(m.G_E_THERM.union(m.G_C_THERM), m.T, rule=operating_state_logic_rule)

        def minimum_on_time_rule(m, g, t):
            """Minimum number of hours generator must be on"""

            # Hours for existing units
            if g in self.existing_units.index:
                hours = self.existing_units.loc[g, ('PARAMETERS', 'MIN_ON_TIME')]

            # Hours for candidate units
            elif g in self.candidate_units.index:
                hours = self.candidate_units.loc[g, ('PARAMETERS', 'MIN_ON_TIME')]

            else:
                raise Exception(f'Min on time hours not found for generator: {g}')

            # Time index used in summation
            time_index = [i for i in range(t - int(hours), t) if i >= 0]

            # Constraint only defined over subset of timestamps
            if t >= hours:
                return sum(m.v[g, j] for j in time_index) <= m.u[g, t]
            else:
                return Constraint.Skip

        # Minimum on time constraint
        m.MINIMUM_ON_TIME = Constraint(m.G_E_THERM.union(m.G_C_THERM), m.T, rule=minimum_on_time_rule)

        def minimum_off_time_rule(m, g, t):
            """Minimum number of hours generator must be off"""

            # Hours for existing units
            if g in self.existing_units.index:
                hours = self.existing_units.loc[g, ('PARAMETERS', 'MIN_OFF_TIME')]

            # Hours for candidate units
            elif g in self.candidate_units.index:
                hours = self.candidate_units.loc[g, ('PARAMETERS', 'MIN_OFF_TIME')]

            else:
                raise Exception(f'Min off time hours not found for generator: {g}')

            # Time index used in summation
            time_index = [i for i in range(t - int(hours) + 1, t) if i >= 0]

            # Constraint only defined over subset of timestamps
            if t >= hours:
                return sum(m.w[g, j] for j in time_index) <= 1 - m.u[g, t]
            else:
                return Constraint.Skip

        # Minimum off time constraint
        m.MINIMUM_OFF_TIME = Constraint(m.G_E_THERM.union(m.G_C_THERM), m.T, rule=minimum_off_time_rule)

        def power_production_rule(m, g, t):
            """Power production rule"""

            if t != m.T.last():
                return (m.p[g, t] + m.r_up[g, t]
                        <= ((m.P_MAX[g] - m.P_MIN[g]) * m.u[g, t])
                        - ((m.P_MAX[g] - m.RR_SD[g]) * m.w[g, t + 1])
                        + ((m.RR_SU[g] - m.P_MIN[g]) * m.v[g, t + 1]))
            else:
                return m.p[g, t] + m.r_up[g, t] <= (m.P_MAX[g] - m.P_MIN[g]) * m.u[g, t]

        # Power production
        m.POWER_PRODUCTION = Constraint(m.G_E_THERM.union(m.G_C_THERM), m.T, rule=power_production_rule)

        def ramp_rate_up_rule(m, g, t):
            """Ramp-rate up constraint"""

            if t > m.T.first():
                return (m.p[g, t] + m.r_up[g, t]) - m.p[g, t - 1] <= m.RR_UP[g]

            else:
                # Ramp-rate for first interval
                return m.p[g, t] - m.P0[g] <= m.RR_UP[g]

        # Ramp-rate up limit
        m.RAMP_RATE_UP = Constraint(m.G_E_THERM.union(m.G_C_THERM), m.T, rule=ramp_rate_up_rule)

        def ramp_rate_down_rule(m, g, t):
            """Ramp-rate down constraint"""

            if t > m.T.first():
                return - m.p[g, t] + m.p[g, t - 1] <= m.RR_DOWN[g]

            else:
                # Ramp-rate for first interval
                return - m.p[g, t] + m.P0[g] <= m.RR_DOWN[g]

        # Ramp-rate up limit
        m.RAMP_RATE_DOWN = Constraint(m.G_E_THERM.union(m.G_C_THERM), m.T, rule=ramp_rate_down_rule)

        def existing_wind_output_min_rule(m, g, t):
            """Constrain minimum output for existing wind generators"""

            return m.P_TOTAL[g, t] >= 0

        # Minimum wind output for existing generators
        m.EXISTING_WIND_MIN_OUTPUT = Constraint(m.G_E_WIND, m.T, rule=existing_wind_output_min_rule)

        def wind_output_max_rule(m, g, t):
            """
            Constrain maximum output for wind generators

            Note: Candidate unit output depends on investment decisions
            """

            # Get wind bubble to which generator belongs
            if g in m.G_E_WIND:

                # If an existing generator
                bubble = self.existing_wind_bubble_map.loc[g, 'BUBBLE']

            elif g in m.G_C_WIND:

                # If a candidate generator
                bubble = self.candidate_units.loc[g, ('PARAMETERS', 'WIND_BUBBLE')]

            else:
                raise Exception(f'Unexpected generator: {g}')

            return m.P_TOTAL[g, t] <= m.Q_WIND[bubble, t] * m.P_MAX[g]

        # Max output from existing wind generators
        m.EXISTING_WIND_MAX_OUTPUT = Constraint(m.G_E_WIND.union(m.G_C_WIND), m.T, rule=wind_output_max_rule)

        def solar_output_max_rule(m, g, t):
            """
            Constrain maximum output for solar generators

            Note: Candidate unit output depends on investment decisions
            """

            # Get NEM zone
            if g in m.G_E_SOLAR:
                # If an existing generator
                zone = self.existing_units.loc[g, ('PARAMETERS', 'NEM_ZONE')]

                # Assume existing arrays are single-axis tracking arrays
                technology = 'SAT'

            elif g in m.G_C_SOLAR:
                # If a candidate generator
                zone = self.candidate_units.loc[g, ('PARAMETERS', 'ZONE')]

                # Extract technology type from unit ID
                technology = g.split('-')[-1]

            else:
                raise Exception(f'Unexpected generator: {g}')

            return m.P_TOTAL[g, t] <= m.Q_SOLAR[zone, technology, t] * m.P_MAX[g]

        # Max output from existing wind generators
        m.EXISTING_SOLAR_MAX_OUTPUT = Constraint(m.G_E_SOLAR.union(m.G_C_SOLAR), m.T, rule=solar_output_max_rule)

        def hydro_output_max_rule(m, g, t):
            """
            Constrain hydro output to registered capacity of existing plant

            Note: Assume no investment in hydro over model horizon (only consider existing units)
            """

            return m.P_TOTAL[g, t] <= m.P_HYDRO_HISTORIC[g, t]

        # Max output from existing hydro generators
        m.EXISTING_HYDRO_MAX_OUTPUT = Constraint(m.G_E_HYDRO, m.T, rule=hydro_output_max_rule)

        # TODO: May want to add energy constraint for hydro units

        def thermal_generator_max_output_rule(m, g, t):
            """Max MW output for thermal generators"""

            return m.P_TOTAL[g, t] <= m.P_MAX[g]

        # Max output for existing and candidate thermal generators
        m.EXISTING_THERMAL_MAX_OUTPUT = Constraint(m.G_E_THERM.union(m.G_C_THERM), m.T,
                                                   rule=thermal_generator_max_output_rule)

        def storage_max_charge_rate_rule(m, g, t):
            """Maximum charging power for storage units [MW]"""

            return m.p_in[g, t] <= m.P_STORAGE_MAX_IN[g]

        # Storage unit max charging power
        m.STORAGE_MAX_CHARGE_RATE = Constraint(m.G_C_STORAGE, m.T, rule=storage_max_charge_rate_rule)

        def storage_max_discharge_rate_rule(m, g, t):
            """Maximum discharging power for storage units [MW]"""

            return m.p_out[g, t] + m.r_up[g, t] <= m.P_STORAGE_MAX_OUT[g]

        # Storage unit max charging power
        m.STORAGE_MAX_DISCHARGE_RATE = Constraint(m.G_C_STORAGE, m.T, rule=storage_max_discharge_rate_rule)

        def state_of_charge_rule(m, g, t):
            """Energy within a given storage unit [MWh]"""

            return m.y[g, t] <= m.STORAGE_UNIT_ENERGY_CAPACITY[g]

        # Energy within storage unit [MWh]
        m.STATE_OF_CHARGE = Constraint(m.G_C_STORAGE, m.T, rule=state_of_charge_rule)

        def storage_energy_transition_rule(m, g, t):
            """Constraint that couples energy + power between periods for storage units"""

            if t != m.T.first():
                return m.y[g, t] == m.y[g, t - 1] + (m.BATTERY_EFFICIENCY[g] * m.p_in[g, t]) - (
                        m.p_out[g, t] / m.BATTERY_EFFICIENCY[g])
            else:
                # Assume battery completely discharged in first period
                return m.y[g, t] == m.Y0[g] + (m.BATTERY_EFFICIENCY[g] * m.p_in[g, t]) - (
                        m.p_out[g, t] / m.BATTERY_EFFICIENCY[g])

        # Account for inter-temporal energy transition within storage units
        m.STORAGE_ENERGY_TRANSITION = Constraint(m.G_C_STORAGE, m.T, rule=storage_energy_transition_rule)

        def storage_interval_end_min_energy_rule(m, g):
            """Lower bound on permissible amount of energy in storage unit at end of operating scenario"""

            return m.STORAGE_INTERVAL_END_MIN_ENERGY[g] <= m.y[g, m.T.last()]

        # Minimum amount of energy that must be in storage unit at end of operating scenario
        m.STORAGE_INTERVAL_END_MIN_ENERGY_CONS = Constraint(m.G_C_STORAGE, rule=storage_interval_end_min_energy_rule)

        def storage_interval_end_max_energy_rule(m, g):
            """Upper bound on permissible amount of energy in storage unit at end of operating scenario"""

            return m.y[g, m.T.last()] <= m.STORAGE_INTERVAL_END_MAX_ENERGY[g]

        # Maximum amount of energy that must be in storage unit at end of operating scenario
        m.STORAGE_INTERVAL_END_MAX_ENERGY_CONS = Constraint(m.G_C_STORAGE, rule=storage_interval_end_max_energy_rule)

        def power_balance_rule(m, z, t):
            """Power balance for each NEM zone"""

            # Existing units within zone
            existing_units = self.existing_units[self.existing_units[('PARAMETERS', 'NEM_ZONE')] == z].index.tolist()

            # Candidate units within zone
            candidate_units = self.candidate_units[self.candidate_units[('PARAMETERS', 'ZONE')] == z].index.tolist()

            # All generators within a given zone
            generators = existing_units + candidate_units

            # Storage units within a given zone
            storage_units = (self.battery_properties.loc[self.battery_properties['NEM_ZONE'] == z, 'NEM_ZONE']
                             .index.tolist())

            return (sum(m.P_TOTAL[g, t] for g in generators) - m.DEMAND[z, t]
                    - sum(m.INCIDENCE_MATRIX[l, z] * m.p_flow[l, t] for l in m.L)
                    + sum(m.p_out[g, t] - m.p_in[g, t] for g in storage_units)
                    + m.p_lost_up[z, t] - m.p_lost_down[z, t] == 0)

        # Power balance constraint for each zone and time period
        m.POWER_BALANCE = Constraint(m.Z, m.T, rule=power_balance_rule)

        def powerflow_min_constraint_rule(m, l, t):
            """Minimum powerflow over a link connecting adjacent NEM zones"""

            return m.p_flow[l, t] >= m.POWERFLOW_MIN[l]

        # Constrain max power flow over given network link
        m.POWERFLOW_MIN_CONS = Constraint(m.L_I, m.T, rule=powerflow_min_constraint_rule)

        def powerflow_max_constraint_rule(m, l, t):
            """Maximum powerflow over a link connecting adjacent NEM zones"""

            return m.p_flow[l, t] <= m.POWERFLOW_MAX[l]

        # Constrain max power flow over given network link
        m.POWERFLOW_MAX_CONS = Constraint(m.L_I, m.T, rule=powerflow_max_constraint_rule)

        return m

    @staticmethod
    def define_objective(m):
        """Objective function - cost minimisation for each scenario"""

        # Minimise total operating cost
        m.OBJECTIVE = Objective(expr=m.C_OP_TOTAL, sense=minimize)

        return m

    def construct_model(self):
        """Assemble model components"""

        # Initialise model object
        m = ConcreteModel()

        # Define sets
        m = self.define_sets(m)

        # Define parameters
        m = self.define_parameters(m)

        # Define variables
        m = self.define_variables(m)

        # Construct expressions
        m = self.define_expressions(m)

        # # Construct constraints
        m = self.define_constraints(m)

        # Construct objective function
        m = self.define_objective(m)

        return m

    @staticmethod
    def update_iteration_parameters(m, fixed_baseline, fixed_permit_price, capacity_continuous, capacity_discrete):
        """
        Update variables determined in master problem which will be fixed for entire iteration

        Note: Only needs to be done once per iteration
        """

        def _update_fixed_baseline(m, fixed_baseline):
            """Fix emissions intensity baseline to given value in sub-problems"""

            # Fixed emissions intensity baseline
            m.FIXED_BASELINE = float(fixed_baseline)

        def _update_fixed_permit_price(m, fixed_permit_price):
            """Fix emissions intensity baseline to given value in sub-problems"""

            # Fixed emissions intensity baseline
            m.FIXED_PERMIT_PRICE = float(fixed_permit_price)

        def _update_candidate_unit_capacity_continuous(m, capacity_continuous):
            """Update candidate unit capacity for units with continuous sizing options"""

            # For each candidate generator with a continuous sizing option
            for g in m.G_C_WIND.union(m.G_C_SOLAR).union(m.G_C_STORAGE):

                # Each year in the model horizon
                for i in m.I:
                    # Fix capacity of candidate generator
                    m.FIXED_X_C[g, i] = float(capacity_continuous[g][i])

        def _update_candidate_unit_capacity_discrete(m, capacity_discrete):
            """Update binary variables used in discrete candidate capacity sizing"""

            # For each candidate generator with a discrete sizing option
            for g in m.G_C_THERM:

                # Each year in the model horizon
                for i in m.I:

                    # Each sizing option
                    for n in m.G_C_THERM_SIZE_OPTIONS:
                        # Fix integer variables used to determine discrete sizing options
                        m.FIXED_D[g, i, n] = float(capacity_discrete[g][i][n])

        def _update_all_parameters(m, fixed_baseline, fixed_permit_price, capacity_continuous, capacity_discrete):
            """Run each function used to update model parameters"""

            _update_fixed_baseline(m, fixed_baseline)
            _update_fixed_permit_price(m, fixed_permit_price)
            _update_candidate_unit_capacity_continuous(m, capacity_continuous)
            _update_candidate_unit_capacity_discrete(m, capacity_discrete)

        # Update model parameters
        _update_all_parameters(m, fixed_baseline, fixed_permit_price, capacity_continuous, capacity_discrete)

        return m

    def update_year_parameters(self, m, year):
        """
        Update model parameters for a given year

        Note: This only needs to be called once per year e.g. before the operating
        scenarios for a given year are run
        """

        def _update_model_year(m, year):
            """Update year for which model is to be run"""

            # Update model year
            m.YEAR = year

        def _update_year_indicator(m, year):
            """Update year indicator parameter - used to control available capacity of candidate units"""

            for i in m.I:
                if i <= year:
                    m.YEAR_INDICATOR[i] = 1
                else:
                    m.YEAR_INDICATOR[i] = 0

        def _update_short_run_marginal_costs(m, year):
            """Update short-run marginal costs for generators - based on fuel-cost profiles"""

            for g in m.G_E_THERM:

                # Last year in the dataset for which fuel cost information exists
                max_year = self.existing_units.loc[g, 'FUEL_COST'].index.max()

                # If year in model horizon exceeds max year for which data are available use values for last
                # available year
                if year > max_year:
                    # Use final year in dataset to max year
                    year = max_year

                m.C_MC[g] = float(self.existing_units.loc[g, ('FUEL_COST', year)]
                                  * self.existing_units.loc[g, ('PARAMETERS', 'HEAT_RATE')])

            for g in m.G_C_THERM:
                # Last year in the dataset for which fuel cost information exists
                max_year = self.candidate_units.loc[g, 'FUEL_COST'].index.max()

                # If year in model horizon exceeds max year for which data are available use values for last
                # available year
                if year > max_year:
                    # Use final year in dataset to max year
                    year = max_year

                m.C_MC[g] = float(self.candidate_units.loc[g, ('FUEL_COST', year)])

        def _update_u0(m, year):
            """Update initial on-state for given generators"""

            for g in m.G_E_THERM.union(m.G_C_THERM):
                m.U0[g] = float(1)

        def _update_availability_indicator(m, year):
            """Update generator availability status (set = 0 if unit retires)"""

            for g in m.G:
                m.AVAILABILITY_INDICATOR[g] = float(1)

        def _update_all_parameters(m, year):
            """Run each function used to update model parameters"""

            _update_model_year(m, year)
            _update_year_indicator(m, year)
            _update_short_run_marginal_costs(m, year)
            _update_u0(m, year)
            _update_availability_indicator(m, year)

        # Update model parameters
        _update_all_parameters(m, year)

        return m

    def update_scenario_parameters(self, m, year, scenario):
        """
        Update parameters for a given operating scenario

        Note: should be called prior to running each operating scenario
        """

        def _update_wind_capacity_factors(m, year, scenario):
            """Update capacity factors for wind generators"""

            # For each wind bubble
            for b in m.B:

                # For each timestamp of a given operating scenario
                for t in m.T:

                    # Capacity factor
                    cf = float(self.input_traces.loc[(year, scenario), ('WIND', b, t)])

                    # Set = 0 if smaller than threshold (don't want numerical instability due to small values)
                    if cf < 0.01:
                        cf = float(0)

                    # Update wind capacity factor
                    m.Q_WIND[b, t] = cf

        def _update_solar_capacity_factors(m, year, scenario):
            """Update capacity factors for solar generators"""

            # For each zone
            for z in m.Z:

                # For each solar technology
                for tech in m.G_C_SOLAR_TECHNOLOGIES:

                    # For each timestamp
                    for t in m.T:

                        # FFP in NTNDP modelling assumptions dataset is FFP2 in trace dataset. Use 'replace' to ensure
                        # consistency
                        tech_name = tech.replace('FFP', 'FFP2')

                        # Capacity factor
                        cf = float(self.input_traces.loc[(year, scenario), ('SOLAR', f"{z}|{tech_name}", t)])

                        # Set = 0 if less than some threshold (don't want very small numbers in model)
                        if cf < 0.01:
                            cf = float(0)

                        # Update solar capacity factor for given zone and technology
                        m.Q_SOLAR[z, tech, t] = cf

        def _update_zone_demand(m, year, scenario):
            """Update zonal demand"""

            # For each zone
            for z in m.Z:

                # For each hour in the given operating scenario
                for t in m.T:
                    # Update zone demand
                    m.DEMAND[z, t] = float(self.input_traces.loc[(year, scenario), ('DEMAND', z, t)])

        def _update_hydro_output(m, year, scenario):
            """Update hydro output based on historic values"""

            # Update historic hydro output for each operating scenario
            for g in m.G_E_HYDRO:

                for t in m.T:
                    output = self.input_traces.loc[(year, scenario), ('HYDRO', g, t)]

                    # Set output = 0 if value too small (prevent numerical instability if numbers are too small)
                    if output < 0.01:
                        output = 0

                    m.P_HYDRO_HISTORIC[g, t] = output

        def _update_scenario_duration(m, year, scenario):
            """Update duration of operating scenario"""

            # TODO: Must figure out how scenario duration is handled in master and sub-problems
            m.SCENARIO_DURATION = float(1)

            return m

        def _update_all_parameters(m, year, scenario):
            """Run each function used to update model parameters"""

            _update_wind_capacity_factors(m, year, scenario)
            _update_solar_capacity_factors(m, year, scenario)
            _update_zone_demand(m, year, scenario)
            _update_hydro_output(m, year, scenario)
            _update_scenario_duration(m, year, scenario)

            return m

        # Update model parameters
        _update_all_parameters(m, year, scenario)

        return m

    @staticmethod
    def fix_baseline(m):
        """Fix emissions intensity baseline"""

        # Fix emissions intensity baseline
        m.baseline.fix(m.FIXED_BASELINE.value)

        return m

    @staticmethod
    def unfix_baseline(m):
        """Unfix emissions intensity baseline"""

        # Fix emissions intensity baseline
        m.baseline.unfix()

        return m

    @staticmethod
    def fix_permit_price(m):
        """Fix permit price"""

        # Fix permit price
        m.permit_price.fix(m.FIXED_PERMIT_PRICE.value)

        return m

    @staticmethod
    def unfix_permit_price(m):
        """Unfix permit price"""

        # Fix permit price
        m.permit_price.unfix()

        return m

    def fix_policy_variables(self, m):
        """Fix policy variables"""

        # Fix baseline
        m = self.fix_baseline(m)

        # Fix permit price
        m = self.fix_permit_price(m)

        return m

    def unfix_policy_variables(self, m):
        """Unfix policy variables"""

        # Unfix baseline
        m = self.unfix_baseline(m)

        # Fix permit price
        m = self.unfix_permit_price(m)

        return m

    @staticmethod
    def fix_candidate_unit_capacity_variables(m):
        """Fix variables in the sub-problem"""

        # For each candidate generator with continuous sizing options
        for g in m.G_C_SOLAR.union(m.G_C_WIND).union(m.G_C_STORAGE):

            # Each year in the model horizon
            for i in m.I:
                # Fix capacity of candidate generator
                m.x_c[g, i].fix(m.FIXED_X_C[g, i].value)

        # For each candidate generator with discrete sizing options
        for g in m.G_C_THERM:

            # Each year in the model horizon
            for i in m.I:

                # Each size option
                for n in m.G_C_THERM_SIZE_OPTIONS:
                    # Fix integer variables
                    m.d[g, i, n].fix(m.FIXED_D[g, i, n].value)

        return m

    @staticmethod
    def unfix_candidate_unit_capacity_variables(m):
        """Fix variables in the sub-problem"""

        # For each candidate generator with continuous sizing options
        for g in m.G_C_SOLAR.union(m.G_C_WIND).union(m.G_C_STORAGE):

            # Each year in the model horizon
            for i in m.I:
                # Fix capacity of candidate generator
                m.x_c[g, i].unfix()

        # For each candidate generator with discrete sizing options
        for g in m.G_C_THERM:

            # Each year in model horizon
            for i in m.I:

                # Each sizing option
                for n in m.G_C_THERM_SIZE_OPTIONS:
                    # Unfix integer variables used to make discrete size selection
                    m.d[g, i, n].unfix()

        return m

    @staticmethod
    def fix_uc_integer_variables(m):
        """
        Fix integer variables

        Note: These values should be fixed to those obtained from the MILP solution.
        This effectively means a linear program is being solved, allowing dual variables
        (marginal prices) to be extracted.
        """

        for g in m.G_E_THERM.union(m.G_C_THERM):
            for t in m.T:
                # Fix integer variables
                m.u[g, t].fix()
                m.v[g, t].fix()
                m.w[g, t].fix()

        return m

    @staticmethod
    def unfix_uc_integer_variables(m):
        """
        Unfix integer variables

        Note: This must be done prior to solving the MILP for different parameters
        """

        for g in m.G_E_THERM.union(m.G_C_THERM):
            for t in m.T:
                # Fix integer variables
                m.u[g, t].unfix()
                m.v[g, t].unfix()
                m.w[g, t].unfix()

        return m

    @staticmethod
    def fix_uc_continuous_variables(m):
        """Fix subproblem primal variables"""

        for t in m.T:

            for g in m.G:
                # Power output above minimum dispatchable level of output [MW]
                m.p[g, t].fix()

            for g in m.G_E_THERM.union(m.G_C_THERM):
                # Upward reserve allocation [MW]
                m.r_up[g, t].fix()

            for g in m.G_C_STORAGE:
                # Storage unit charging (power in) [MW]
                m.p_in[g, t].fix()

                # Storage unit discharging (power out) [MW]
                m.p_out[g, t].fix()

                # Energy in storage unit [MWh]
                m.y[g, t].fix()

            for l in m.L:
                # Power flow between NEM zones
                m.p_flow[l, t].fix()

            for z in m.Z:
                # Lost load - up
                m.p_lost_up[z, t].fix()

                # Lost load - down
                m.p_lost_down[z, t].fix()

        return m

    def construct_model_iteration(self, m, fixed_baseline, fixed_permit_price, capacity_continuous, capacity_discrete):
        """Run model for a given operating scenario"""

        # Update parameters
        m = self.update_iteration_parameters(m, fixed_baseline, fixed_permit_price, capacity_continuous,
                                             capacity_discrete)

        return m

    def construct_model_year(self, m, year):
        """Update model parameters that will entire all operating scenarios for a given year"""

        # Update parameters
        m = self.update_year_parameters(m, year)

        return m

    def construct_model_scenario(self, m, year, scenario):
        """Update model parameters applying to a given operating scenario"""

        # Update parameters
        m = self.update_scenario_parameters(m, year, scenario)

        return m

    def solve_model(self, m):
        """Solve model for a given operating scenario"""

        # Solve model
        self.opt.solve(m, tee=True, options=self.solver_options, keepfiles=self.keepfiles)

        # Log infeasible constraints if they exist
        log_infeasible_constraints(m)


if __name__ == '__main__':
    # Directory containing files from which dataset is derived
    raw_data_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'data')

    # Directory containing core data files
    data_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '1_collect_data', 'output')

    # Directory containing input traces
    input_traces_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '2_input_traces', 'output')

    # Instantiate object used to construct unit commitment model
    uc = UnitCommitmentModel(raw_data_directory, data_directory, input_traces_directory)

    # Construct model
    model = uc.construct_model()

    # Initialise dictionary of installed capacities for candidate technologies
    capacities_continuous = {g: {i: 0 for i in range(2016, 2051)} for g in
                             model.G_C_SOLAR.union(model.G_C_WIND).union(model.G_C_STORAGE)}
    capacities_continuous['TAS-WIND'][2020] = 100

    # Initialise dictionary of integer variables used to make discrete unit sizing decisions
    capacities_discrete = {
        g: {i: {n: 1 if n == 0 else 0 for n in model.G_C_THERM_SIZE_OPTIONS} for i in range(2016, 2051)}
        for g in model.G_C_THERM}

    # Construct model objects
    # -----------------------
    # Clone the base model
    m_base = model.clone()

    # Construct base model object used in each iteration
    m_iteration = uc.construct_model_iteration(m_base, 0.8, 50, capacities_continuous, capacities_discrete)

    # Construct base model used to solve a given year
    m_year = uc.construct_model_year(m_iteration, 2020)

    # Construct model used to solve a given scenario
    m_scenario = uc.construct_model_scenario(m_year, 2020, 1)

    # Import dual variables when obtaining solution to second stage linear program
    m_scenario.dual = Suffix(direction=Suffix.IMPORT)

    # Stage 1 - Solve MILP
    # --------------------
    start = time.time()

    # Fix master problem variables
    m_scenario = uc.fix_baseline(m_scenario)
    m_scenario = uc.fix_permit_price(m_scenario)
    m_scenario = uc.fix_candidate_unit_capacity_variables(m_scenario)

    # Solve model
    uc.solve_model(m_scenario)

    # Get sensitivities for unit capacities and marginal prices
    # ---------------------------------------------------------
    # Fix integer variables for unit commitment problem
    m_scenario = uc.fix_uc_integer_variables(m_scenario)

    # Unfix capacity variables
    m_scenario = uc.unfix_candidate_unit_capacity_variables(m_scenario)

    # Solve model
    uc.solve_model(m_scenario)

    # Get sensitivities for permit price and baseline
    # -----------------------------------------------
    # Fix capacity variables
    m_scenario = uc.fix_candidate_unit_capacity_variables(m_scenario)

    # Fix all unit commitment primal variables
    m_scenario = uc.fix_uc_continuous_variables(m_scenario)

    # Unfix permit price and baseline
    m_scenario = uc.unfix_permit_price(m_scenario)
    m_scenario = uc.unfix_baseline(m_scenario)

    # Solve model
    uc.solve_model(m_scenario)

    print(f'Solved in: {time.time() - start}s')
