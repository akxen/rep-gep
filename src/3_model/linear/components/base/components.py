"""Model components common to both subproblem and master problem"""

import random

from pyomo.environ import *

from data import ModelData


class CommonComponents:

    def __init__(self, first_year, final_year, scenarios_per_year):
        # Model data
        self.data = ModelData()

        # First year in model horizon (min is 2016)
        self.first_year = first_year

        # Default final year (max is 2050)
        self.final_year = final_year

        # Default number of scenarios per year (max is 10)
        self.scenarios_per_year = scenarios_per_year

    def define_sets(self, m):
        """Define sets to be used in model"""

        # NEM regions
        m.R = Set(initialize=self.data.nem_regions)

        # NEM zones
        m.Z = Set(initialize=self.data.nem_zones)

        # Links between NEM zones
        m.L = Set(initialize=self.data.network_links)

        # Interconnectors for which flow limits are defined
        m.L_I = Set(initialize=list(self.data.powerflow_limits.keys()))

        # NEM wind bubbles
        m.B = Set(initialize=self.data.wind_bubbles)

        # Existing thermal units
        m.G_E_THERM = Set(initialize=self.data.existing_thermal_unit_ids)

        # Candidate thermal units
        m.G_C_THERM = Set(initialize=self.data.candidate_thermal_unit_ids)

        # All existing and candidate thermal generators
        m.G_THERM = Set(initialize=m.G_E_THERM.union(m.G_C_THERM))

        # Index for candidate thermal unit size options
        m.G_C_THERM_SIZE_OPTIONS = RangeSet(0, 3, ordered=True)

        # Existing wind units
        m.G_E_WIND = Set(initialize=self.data.existing_wind_unit_ids)

        # Candidate wind units
        m.G_C_WIND = Set(initialize=self.data.candidate_wind_unit_ids)

        # Existing solar units
        m.G_E_SOLAR = Set(initialize=self.data.existing_solar_unit_ids)

        # Candidate solar units
        m.G_C_SOLAR = Set(initialize=self.data.candidate_solar_unit_ids)

        # Available technologies
        m.G_C_SOLAR_TECHNOLOGIES = Set(initialize=list(set(y.split('-')[-1] for y in m.G_C_SOLAR)))

        # Existing hydro units
        m.G_E_HYDRO = Set(initialize=self.data.existing_hydro_unit_ids)

        # Existing storage units
        m.G_E_STORAGE = Set(initialize=self.data.existing_storage_unit_ids)

        # Candidate storage units
        m.G_C_STORAGE = Set(initialize=self.data.candidate_storage_units)

        # Existing + candidate storage units
        m.G_STORAGE = m.G_E_STORAGE.union(m.G_C_STORAGE)

        # Slow start thermal generators (existing and candidate)
        m.G_THERM_SLOW = Set(initialize=self.data.slow_start_thermal_generator_ids)

        # Quick start thermal generators (existing and candidate)
        m.G_THERM_QUICK = Set(initialize=self.data.quick_start_thermal_generator_ids)

        # All existing generators
        m.G_E = m.G_E_THERM.union(m.G_E_WIND).union(m.G_E_SOLAR).union(m.G_E_HYDRO)

        # All candidate generators
        m.G_C = m.G_C_THERM.union(m.G_C_WIND).union(m.G_C_SOLAR).union(m.G_C_STORAGE)

        # All generators
        m.G = m.G_E.union(m.G_C)

        # All years in model horizon
        m.Y = RangeSet(self.first_year, self.final_year)

        # Operating scenarios for each year
        m.S = RangeSet(1, self.scenarios_per_year)

        # Operating scenario hour
        m.T = RangeSet(1, 24, ordered=True)

        # Build limit technology types
        m.BUILD_LIMIT_TECHNOLOGIES = Set(initialize=self.data.candidate_unit_build_limits.index)

        return m

    def define_parameters(self, m):
        """Define common parameters"""

        def minimum_power_output_rule(_m, g):
            """Minimum power output for candidate and existing generators (excluding storage)"""

            return float(0)

        # Min power output
        m.P_MIN = Param(m.G.difference(m.G_STORAGE), rule=minimum_power_output_rule)

        def existing_units_max_output_rule(_m, g):
            """Max power output for existing units"""

            return float(self.data.existing_units_dict[('PARAMETERS', 'REG_CAP')][g])

        # Max output for existing units
        m.P_MAX = Param(m.G_E, rule=existing_units_max_output_rule)

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

        def asset_life_rule(_m, g):
            """Assumed lifetime (years) for candidate generators"""

            # TODO: Update this assumption with better data
            return float(25)

        # Asset lifetime for candidate generators
        m.ASSET_LIFE = Param(m.G_C, rule=asset_life_rule)

        # Interest rate (weighted average cost of capital)
        m.INTEREST_RATE = Param(initialize=float(self.data.WACC))

        def wind_capacity_factors_rule(_m, g, y, s, t):
            """Get wind capacity factors"""

            # If an existing wind generator
            if g in m.G_E_WIND:
                # Wind bubble to which existing wind generator belongs
                bubble = self.data.existing_wind_bubble_map_dict['BUBBLE'][g]

                # Check that value is sufficiently large (prevent numerical ill conditioning)
                val = self.data.input_traces_dict[('WIND', bubble, t)][y, s]

            # If a candidate generator
            elif g in m.G_C_WIND:
                # Wind bubble to which candidate wind generator belongs
                bubble = self.data.candidate_units_dict[('PARAMETERS', 'WIND_BUBBLE')][g]

                # Check that value is sufficiently large (prevent numerical ill conditioning)
                val = self.data.input_traces_dict[('WIND', bubble, t)][y, s]

            else:
                raise Exception(f'Unexpected wind generator encountered: {g}')

            # Set to zero if value is too small (prevent numerical ill conditioning)
            if val < 0.1:
                val = 0

            # Some capacity factors are > 1 in AEMO dataset. Set = 0.99 to prevent constraint violation
            elif val >= 1:
                val = 0.99

            return float(val)

        # Wind capacity factors
        m.Q_W = Param(m.G_E_WIND.union(m.G_C_WIND), m.Y, m.S, m.T, initialize=wind_capacity_factors_rule)

        def solar_capacity_factors_rule(_m, g, y, s, t):
            """Get solar capacity factors"""

            # Map between existing solar DUIDs and labels in input_traces database
            solar_map = {'BROKENH1': 'BROKENHILL', 'MOREESF1': 'MOREE', 'NYNGAN1': 'NYNGAN'}

            if g in m.G_C_SOLAR:
                # Get zone and technology
                zone, tech = g.split('-')[0], g.split('-')[-1]

                # FFP used in model, but but FFP2 in traces database - convert so both are the same
                if tech == 'FFP':
                    tech = 'FFP2'

                # Solar capacity factors - candidate units
                val = self.data.input_traces_dict[('SOLAR', f'{zone}|{tech}', t)][(y, s)]

            elif g in m.G_E_SOLAR:
                # Solar capacity factor - existing units
                val = self.data.input_traces_dict[('SOLAR', solar_map[g], t)][(y, s)]

            else:
                raise Exception(f'Unexpected solar unit: {g}')

            # Check that value is sufficiently large (prevent numerical ill conditioning)
            if val < 0.1:
                val = 0

            # Some capacity factors are > 1 in AEMO dataset. Set = 0.99 to prevent constraint violation
            elif val >= 1:
                val = 0.99

            return float(val)

        # Wind capacity factors
        m.Q_S = Param(m.G_E_SOLAR.union(m.G_C_SOLAR), m.Y, m.S, m.T, initialize=solar_capacity_factors_rule)

        def retirement_indicator_rule(_m, g, y):
            """Indicates if unit is retired. 1=unit is retired, 0=unit operational"""

            # Year in which unit retires
            year = self.data.unit_retirement.get(g)

            if year is not None:

                # Unit is retired
                if y >= year:
                    return float(1)

                # Unit is still operational
                else:
                    return float(0)

            else:
                return float(0)

        # Retirement indicator
        m.F = Param(m.G_E.union(m.G_E_STORAGE), m.Y, rule=retirement_indicator_rule)

        def hydro_output_rule(_m, g, y, s, t):
            """Get hydro output for a given scenario"""

            # Historic hydro output - for operating scenario
            val = self.data.input_traces_dict[('HYDRO', g, t)][(y, s)]

            # Set output to zero if negative or very small (SCADA data collection may cause -ve values)
            if val < 0.1:
                val = 0

            return float(val)

        # Max hydro output
        m.P_H = Param(m.G_E_HYDRO, m.Y, m.S, m.T, rule=hydro_output_rule)

        def max_power_in_existing_storage_rule(_m, g):
            """Max charging power for existing storage units"""

            return self.data.existing_storage_units_dict[g]['REG_CAP']

        # Max charging power for existing storage units
        m.P_IN_MAX = Param(m.G_E_STORAGE, rule=max_power_in_existing_storage_rule)

        def max_power_out_existing_storage_rule(_m, g):
            """Max discharging power for existing storage units"""

            return self.data.existing_storage_units_dict[g]['REG_CAP']

        # Max discharging power for existing storage units
        m.P_OUT_MAX = Param(m.G_E_STORAGE, rule=max_power_out_existing_storage_rule)

        def max_energy_storage_rule(_m, g):
            """Max energy level for existing storage units"""

            return self.data.existing_storage_units_dict[g]['ENERGY']

        # Max energy level for existing storage units
        m.Q_MAX = Param(m.G_E_STORAGE, rule=max_energy_storage_rule)

        def min_energy_interval_end_rule(_m, g):
            """Minimum energy level at end of scenario"""

            return float(0)

        # Minimum energy level at end of scenario
        m.Q_END_MIN = Param(m.G_STORAGE, rule=min_energy_interval_end_rule)

        def max_energy_interval_end_rule(_m, g):
            """Max energy level at end of scenario"""

            # Arbitrarily high upper-limit (note: energy will also be constrained by transition function)
            return float(500)

        # Max energy level at end of scenario
        m.Q_END_MAX = Param(m.G_STORAGE, rule=max_energy_interval_end_rule)

        def initial_energy_rule(_m, g, y, s):
            """Initial energy in storage units"""

            # Assumes units are completely discharged at beginning of scenario
            return float(0)

        # Initial energy level for storage units
        m.Q0 = Param(m.G_STORAGE, m.Y, m.S, rule=initial_energy_rule)

        def battery_efficiency_rule(_m, g):
            """Battery efficiency"""

            if g in m.G_E_STORAGE:
                # Assumed efficiency based on candidate unit battery efficiency
                return float(0.92)

            elif g in m.G_C_STORAGE:
                return float(self.data.battery_properties.loc[g, 'CHARGE_EFFICIENCY'])

            else:
                raise Exception(f'Unknown storage ID encountered: {g}')

        # Battery efficiency
        m.ETA = Param(m.G_STORAGE, rule=battery_efficiency_rule)

        def initial_power_output(_m, g, y, s):
            """Initial power output prior to model start"""

            # Power output = 0 if unit is retired
            if (g in m.G_E_THERM) and (m.F[g, y] == 1):
                return float(0)

            # If an existing slow-start generator which is not retired - most likely baseload generator
            elif g in m.G_THERM_SLOW:
                return float(self.data.existing_units_dict[('PARAMETERS', 'MIN_GEN')][g])

            # Assume generator is off
            else:
                return float(0)

        # Initial power output
        m.P0 = Param(m.G.difference(m.G_STORAGE), m.Y, m.S, rule=initial_power_output)

        def ramp_rate_up_rule(_m, g):
            """Ramp-rate up (MW/h) - when running"""

            if g in m.G_STORAGE:
                # Default ramp-rate for storage units (arbitrarily large)
                ramp_up = 9000

            elif g in m.G_E.difference(m.G_STORAGE):
                # Ramp-rate up for existing generators
                ramp_up = self.data.existing_units.loc[g, ('PARAMETERS', 'RR_UP')]

            elif g in m.G_C.difference(m.G_STORAGE):
                # Ramp-rate up for candidate generators
                ramp_up = self.data.candidate_units.loc[g, ('PARAMETERS', 'RR_UP')]

            else:
                raise Exception(f'Unexpected generator {g}')

            return float(ramp_up)

        # Ramp-rate up (normal operation)
        m.RR_UP = Param(m.G, rule=ramp_rate_up_rule)

        def ramp_rate_down_rule(_m, g):
            """Ramp-rate down (MW/h) - when running"""

            if g in m.G_STORAGE:
                # Default ramp-rate for storage units (arbitrarily large)
                ramp_down = 9000

            elif g in m.G_E.difference(m.G_STORAGE):
                # Ramp-rate down for existing generators
                ramp_down = self.data.existing_units.loc[g, ('PARAMETERS', 'RR_DOWN')]

            elif g in m.G_C.difference(m.G_STORAGE):
                # Ramp-rate down for candidate generators
                ramp_down = self.data.candidate_units.loc[g, ('PARAMETERS', 'RR_DOWN')]

            else:
                raise Exception(f'Unexpected generator {g}')

            return float(ramp_down)

        # Ramp-rate down (normal operation)
        m.RR_DOWN = Param(m.G, rule=ramp_rate_down_rule)

        def powerflow_min_rule(_m, l):
            """Minimum powerflow over network link"""

            if l in m.L_I:
                return float(-self.data.powerflow_limits[l]['reverse'])
            else:
                # Set arbitrarily loose bound if not an interconnector
                return float(-1e5)

        # Lower bound for powerflow over link
        m.POWERFLOW_MIN = Param(m.L, rule=powerflow_min_rule)

        def powerflow_max_rule(_m, l):
            """Maximum powerflow over network link"""

            if l in m.L_I:
                return float(self.data.powerflow_limits[l]['forward'])
            else:
                # Set arbitrarily loose bound if not an interconnector
                return float(1e5)

        # Lower bound for powerflow over link
        m.POWERFLOW_MAX = Param(m.L, rule=powerflow_max_rule)

        def network_incidence_matrix_rule(_m, l, z):
            """Incidence matrix describing connections between adjacent NEM zones"""

            return float(self.data.network_incidence_matrix.loc[l, z])

        # Network incidence matrix
        m.INCIDENCE_MATRIX = Param(m.L, m.Z, rule=network_incidence_matrix_rule)

        def demand_rule(_m, z, y, s, t):
            """Get demand for a given scenario"""

            # Demand in each NEM zone for a given operating scenario
            val = self.data.input_traces_dict[('DEMAND', z, t)][(y, s)]

            # Check that value is sufficiently large (prevent numerical ill conditioning)
            if val < 0.1:
                val = 0

            return float(val)

        # Zone demand
        m.DEMAND = Param(m.Z, m.Y, m.S, m.T, rule=demand_rule)

        def initial_charging_power_rule(_m, g, y, s):
            """Initial charging power for storage units in interval prior to model start"""

            # Assume power in preceding interval = 0 MW
            return float(0)

        # Max charging power for existing storage units
        m.P_IN_0 = Param(m.G_STORAGE, m.Y, m.S, rule=initial_charging_power_rule)

        def initial_discharging_power_rule(_m, g, y, s):
            """Power output in interval prior to model start"""

            # Assume = 0 MW
            return float(0)

        # Power output in interval preceding model start
        m.P_OUT_0 = Param(m.G_STORAGE, m.Y, m.S, rule=initial_discharging_power_rule)

        def initial_power_lost_load_rule(_m, z, y, s):
            """Load lost power in interval prior to model start"""

            # Assume = 0 MW
            return float(0)

        # Lost lost power in interval preceding model start
        m.P_V0 = Param(m.Z, m.Y, m.S, rule=initial_power_lost_load_rule)

        def marginal_cost_rule(_m, g, y):
            """Marginal costs for existing and candidate generators

            Note: Marginal costs for existing and candidate thermal plant depend on fuel costs, which are time
            varying. Therefore marginal costs for thermal plant must be updated for each year in model horizon.
            """

            if g in m.G_THERM:

                # Existing generators
                if g in m.G_E_THERM:

                    # Last year in the dataset for which fuel cost information exists
                    max_year = max([i[1] for i in self.data.existing_units_dict.keys() if 'FUEL_COST' in i])

                    # If year in model horizon exceeds max year for which data are available use values for last
                    # available year
                    if y > max_year:
                        # Use final year in dataset to max year
                        y = max_year

                    # Fuel cost
                    fuel_cost = self.data.existing_units_dict[('FUEL_COST', y)][g]

                    # Variable operations and maintenance cost
                    vom = self.data.existing_units_dict[('PARAMETERS', 'VOM')][g]

                    # Heat rate
                    heat_rate = self.data.existing_units_dict[('PARAMETERS', 'HEAT_RATE')][g]

                # Candidate generators
                elif g in m.G_C_THERM:

                    # Last year in the dataset for which fuel cost information exists
                    max_year = max([i[1] for i in self.data.existing_units_dict.keys() if 'FUEL_COST' in i])

                    # If year in model horizon exceeds max year for which data are available use values for last
                    # available year
                    if y > max_year:
                        # Use final year in dataset to max year
                        y = max_year

                    # Fuel cost
                    fuel_cost = self.data.candidate_units_dict[('FUEL_COST', y)][g]

                    # Variable operations and maintenance cost
                    vom = self.data.candidate_units_dict[('PARAMETERS', 'VOM')][g]

                    # Heat rate
                    heat_rate = self.data.candidate_units_dict[('PARAMETERS', 'HEAT_RATE')][g]

                else:
                    raise Exception(f'Unexpected generator encountered: {g}')

                # Compute marginal cost for thermal units
                marginal_cost = float((fuel_cost * heat_rate) + vom)

            elif (g in m.G_E_WIND) or (g in m.G_E_SOLAR) or (g in m.G_E_HYDRO):
                # Marginal cost = VOM cost for wind and solar generators
                marginal_cost = self.data.existing_units.loc[g, ('PARAMETERS', 'VOM')]

            elif (g in m.G_C_WIND) or (g in m.G_C_SOLAR):
                # Marginal cost = VOM cost for wind and solar generators
                marginal_cost = self.data.candidate_units.loc[g, ('PARAMETERS', 'VOM')]

            elif g in m.G_STORAGE:
                # Assume marginal cost = VOM cost of typical hydro generator (7 $/MWh)
                marginal_cost = 7

            else:
                raise Exception(f'Unexpected generator: {g}')

            assert marginal_cost >= 0, 'Cannot have negative marginal cost'

            return float(marginal_cost + random.uniform(0, 2))

        # Marginal costs for all generators and time periods
        random.seed(10)
        m.C_MC = Param(m.G.union(m.G_E_STORAGE), m.Y, rule=marginal_cost_rule)

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

        def amortisation_rate_rule(_m, g):
            """Amortisation rate for a given investment"""

            # Numerator for amortisation rate expression
            num = self.data.WACC * ((1 + self.data.WACC) ** m.ASSET_LIFE[g])

            # Denominator for amortisation rate expression
            den = ((1 + self.data.WACC) ** m.ASSET_LIFE[g]) - 1

            # Amortisation rate
            amortisation_rate = num / den

            return amortisation_rate

        # Amortisation rate for a given investment
        m.GAMMA = Param(m.G_C, rule=amortisation_rate_rule)

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
        random.seed(10)
        m.I_C = Param(m.G_C, m.Y, rule=candidate_unit_build_costs_rule)

        # Lost load value [$/MWh]
        def lost_load_cost_rule(_m, z):
            """Return cost for lost-load power in each NEM zone"""

            return float(14700 + random.uniform(0, 100))

        # Lost load cost for each NEM zone
        random.seed(10)
        m.C_L = Param(m.Z, rule=lost_load_cost_rule)

        def discount_rate_rule(_m, y):
            """Discount factor"""

            return float(1 / (1 + m.INTEREST_RATE) ** (y - m.Y.first()))

        # Discount rate
        m.DELTA = Param(m.Y, rule=discount_rate_rule)

        def scenario_duration_rule(_m, y, s):
            """Get duration of operating scenarios (days)"""

            # Account for leap-years
            if y % 4 == 0:
                days_in_year = 366
            else:
                days_in_year = 365

            # Total days represented by scenario
            days = float(self.data.input_traces_dict[('K_MEANS', 'METRIC', 'NORMALISED_DURATION')][(y, s)]
                         * days_in_year)

            return days

        # Scenario duration
        m.RHO = Param(m.Y, m.S, rule=scenario_duration_rule)

        # Cumulative emissions cap - default value is 0 - should be updated when running specific case
        m.CUMULATIVE_EMISSIONS_CAP = Param(initialize=0, mutable=True)

        # Interim emissions cap - default value is 0 - should be updated when running specific case
        m.INTERIM_EMISSIONS_CAP = Param(m.Y, initialize=0, mutable=True)

        # Scheme revenue upper envelope
        m.SCHEME_REVENUE_ENVELOPE_UP = Param(m.Y, initialize=0, mutable=True)

        # Scheme revenue lower envelope
        m.SCHEME_REVENUE_ENVELOPE_LO = Param(m.Y, initialize=0, mutable=True)

        # Price weights
        m.PRICE_WEIGHTS = Param(m.Y, initialize=0, mutable=True)

        return m

    @staticmethod
    def define_variables(m):
        """Define variables common to both primal and dual models (policy variables)"""

        # Emissions intensity baseline [tCO2/MWh]
        m.baseline = Var(m.Y, within=NonNegativeReals, initialize=0)

        # Permit price [$/tCO2]
        m.permit_price = Var(m.Y, within=NonNegativeReals, initialize=0)

        return m

    @staticmethod
    def define_expressions(m):
        """Define expressions common to both primal and dual models"""

        def scenario_demand_rule(_m, y, s):
            """Total demand for a given scenario (MWh)"""

            return sum(m.RHO[y, s] * m.DEMAND[z, y, s, t] for z in m.Z for t in m.T)

        # Scenario demand
        m.SCENARIO_DEMAND = Expression(m.Y, m.S, rule=scenario_demand_rule)

        return m
