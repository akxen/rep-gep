"""Model components common to both subproblem and master problem"""

from pyomo.environ import *

from data import ModelData


class CommonComponents:
    # Model data
    data = ModelData()

    def __init__(self):
        pass

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
        m.G_C_SOLAR_TECHNOLOGIES = Set(initialize=list(set(i.split('-')[-1] for i in m.G_C_SOLAR)))

        # Existing hydro units
        m.G_E_HYDRO = Set(initialize=self.data.existing_hydro_unit_ids)

        # Candidate storage units
        m.G_C_STORAGE = Set(initialize=self.data.candidate_storage_units)

        # Slow start thermal generators (existing and candidate)
        m.G_THERM_SLOW = Set(initialize=self.data.slow_start_thermal_generator_ids)

        # Quick start thermal generators (existing and candidate)
        m.G_THERM_QUICK = Set(initialize=self.data.quick_start_thermal_generator_ids)

        # All existing generators
        m.G_E = m.G_E_THERM.union(m.G_E_WIND).union(m.G_E_SOLAR).union(m.G_E_HYDRO)

        # All candidate generators
        m.G_C = m.G_C_THERM.union(m.G_C_WIND).union(m.G_C_SOLAR)

        # All generators
        m.G = m.G_E.union(m.G_C)

        # All years in model horizon
        m.I = RangeSet(2016, 2017)

        # Operating scenarios for each year
        m.O = RangeSet(0, 9)

        # Operating scenario hour
        m.T = RangeSet(0, 23, ordered=True)

        # Build limit technology types
        m.BUILD_LIMIT_TECHNOLOGIES = Set(initialize=self.data.candidate_unit_build_limits.index)

        return m

    def define_parameters(self, m):
        """Define model parameters - these are common to all blocks"""

        # Cumulative revenue target (for entire model horizon)
        m.REVENUE_TARGET = Param(initialize=0, mutable=True)

        # Penalty imposed for each dollar scheme revenue falls short of target revenue
        m.REVENUE_SHORTFALL_PENALTY = Param(initialize=1000)

        # Emissions target
        m.EMISSIONS_TARGET = Param(initialize=9999999, mutable=True)

        # Penalty imposed for each tCO2 by which emissions target is exceeded
        m.EMISSIONS_EXCEEDED_PENALTY = Param(initialize=1000)

        # Min state of charge for storage unit at end of operating scenario (assume = 0)
        m.STORAGE_INTERVAL_END_MIN_ENERGY = Param(m.G_C_STORAGE, initialize=0)

        # Value of lost-load [$/MWh]
        m.C_LOST_LOAD = Param(initialize=float(1e4))

        # Fixed emissions intensity baseline [tCO2/MWh]
        m.FIXED_BASELINE = Param(m.I, initialize=0, mutable=True)

        # Fixed permit price [$/tCO2]
        m.FIXED_PERMIT_PRICE = Param(m.I, initialize=0, mutable=True)

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
                startup_cost = (self.data.existing_units.loc[g, ('PARAMETERS', 'SU_COST_WARM')]
                                / self.data.existing_units.loc[g, ('PARAMETERS', 'REG_CAP')])

            elif g in m.G_C_THERM:
                # Startup cost for candidate thermal units
                startup_cost = self.data.candidate_units.loc[g, ('PARAMETERS', 'SU_COST_WARM_MW')]

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
                shutdown_cost = (self.data.existing_units.loc[g, ('PARAMETERS', 'SU_COST_WARM')]
                                 / self.data.existing_units.loc[g, ('PARAMETERS', 'REG_CAP')])

            elif g in m.G_C_THERM:
                # Shutdown cost for candidate thermal units
                shutdown_cost = self.data.candidate_units.loc[g, ('PARAMETERS', 'SU_COST_WARM_MW')]

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
                startup_ramp = self.data.existing_units.loc[g, ('PARAMETERS', 'RR_STARTUP')]

            elif g in m.G_C_THERM:
                # Startup ramp-rate for candidate thermal generators
                startup_ramp = self.data.candidate_units.loc[g, ('PARAMETERS', 'RR_STARTUP')]

            else:
                raise Exception(f'Unexpected generator {g}')

            return float(startup_ramp)

        # Startup ramp-rate for existing and candidate thermal generators
        m.RR_SU = Param(m.G_E_THERM.union(m.G_C_THERM), rule=startup_ramp_rate_rule)

        def shutdown_ramp_rate_rule(m, g):
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
        m.RR_SD = Param(m.G_E_THERM.union(m.G_C_THERM), rule=shutdown_ramp_rate_rule)

        def ramp_rate_up_rule(m, g):
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
        m.RR_UP = Param(m.G_E_THERM.union(m.G_C_THERM), rule=ramp_rate_up_rule)

        def ramp_rate_down_rule(m, g):
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
        m.RR_DOWN = Param(m.G_E_THERM.union(m.G_C_THERM), rule=ramp_rate_down_rule)

        def existing_generator_registered_capacities_rule(m, g):
            """Registered capacities of existing generators"""

            return float(self.data.existing_units.loc[g, ('PARAMETERS', 'REG_CAP')])

        # Registered capacities of existing generators
        m.EXISTING_GEN_REG_CAP = Param(m.G_E, rule=existing_generator_registered_capacities_rule)

        def emissions_intensity_rule(m, g):
            """Emissions intensity (tCO2/MWh)"""

            if g in m.G_E_THERM:
                # Emissions intensity
                emissions = float(self.data.existing_units.loc[g, ('PARAMETERS', 'EMISSIONS')])

            elif g in m.G_C_THERM:
                # Emissions intensity
                emissions = float(self.data.candidate_units.loc[g, ('PARAMETERS', 'EMISSIONS')])

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
                min_output = (self.data.existing_units.loc[g, ('PARAMETERS', 'MIN_GEN')] /
                              self.data.existing_units.loc[g, ('PARAMETERS', 'REG_CAP')])

            elif g in m.G_E_HYDRO:
                # Set minimum power output for existing hydro generators = 0
                min_output = 0

            elif g in m.G_C_THERM:
                # Minimum power output for candidate thermal generators
                min_output = self.data.candidate_units.loc[g, ('PARAMETERS', 'MIN_GEN_PERCENT')] / 100

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

        def battery_efficiency_rule(m, g):
            """Battery efficiency"""

            return float(self.data.battery_properties.loc[g, 'CHARGE_EFFICIENCY'])

        # Battery efficiency
        m.BATTERY_EFFICIENCY = Param(m.G_C_STORAGE, rule=battery_efficiency_rule)

        def network_incidence_matrix_rule(m, l, z):
            """Incidence matrix describing connections between adjacent NEM zones"""

            # Network incidence matrix
            df = self.data.network_incidence_matrix.copy()

            return float(df.loc[l, z])

        # Network incidence matrix
        m.INCIDENCE_MATRIX = Param(m.L, m.Z, rule=network_incidence_matrix_rule)

        def candidate_unit_build_limits_rule(m, g, z):
            """Build limits in each zone for each candidate technology"""
            return float(self.data.candidate_unit_build_limits.loc[g, z])

        # Build limits for each candidate technology
        m.BUILD_LIMITS = Param(m.BUILD_LIMIT_TECHNOLOGIES, m.Z, rule=candidate_unit_build_limits_rule)

        def minimum_region_up_reserve_rule(m, r):
            """Minimum upward reserve rule"""

            # Minimum upward reserve for region
            up_reserve = self.data.minimum_reserve_levels.loc[r, 'MINIMUM_RESERVE_LEVEL']

            return float(up_reserve)

        # Minimum upward reserve
        m.RESERVE_UP = Param(m.R, rule=minimum_region_up_reserve_rule)

        def powerflow_min_rule(m, l):
            """Minimum powerflow over network link"""

            return float(-self.data.powerflow_limits[l]['reverse'] * 100)

        # Lower bound for powerflow over link
        m.POWERFLOW_MIN = Param(m.L_I, rule=powerflow_min_rule)

        def powerflow_max_rule(m, l):
            """Maximum powerflow over network link"""

            return float(self.data.powerflow_limits[l]['forward'] * 100)

        # Lower bound for powerflow over link
        m.POWERFLOW_MAX = Param(m.L_I, rule=powerflow_max_rule)

        def marginal_cost_rule(_s, g, i):
            """Marginal costs for existing and candidate generators

            Note: Marginal costs for existing and candidate thermal plant depend on fuel costs, which are time
            varying. Therefore marginal costs for thermal plant must be define for each year in model horizon.
            """

            if g in m.G_E_THERM:

                # Last year in the dataset for which fuel cost information exists
                max_year = self.data.existing_units.loc[g, 'FUEL_COST'].index.max()

                # If year in model horizon exceeds max year for which data are available use values for last
                # available year
                if i > max_year:
                    # Use final year in dataset to max year
                    i = max_year

                marginal_cost = float(self.data.existing_units.loc[g, ('FUEL_COST', i)]
                                      * self.data.existing_units.loc[g, ('PARAMETERS', 'HEAT_RATE')])

            elif g in m.G_C_THERM:
                # Last year in the dataset for which fuel cost information exists
                max_year = self.data.candidate_units.loc[g, 'FUEL_COST'].index.max()

                # If year in model horizon exceeds max year for which data are available use values for last
                # available year
                if i > max_year:
                    # Use final year in dataset to max year
                    i = max_year

                marginal_cost = float(self.data.candidate_units.loc[g, ('FUEL_COST', i)])

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
        m.C_MC = Param(m.G.union(m.G_C_STORAGE), m.I, rule=marginal_cost_rule)

        def fom_cost_rule(m, g):
            """Fixed operating and maintenance cost for all candidate generators"""

            if g in m.G_C_STORAGE:
                # TODO: Need to find reasonable FOM cost for storage units - setting = to MEL-WIND for now
                return float(self.data.candidate_units_dict[('PARAMETERS', 'FOM')]['MEL-WIND'])
            else:
                return float(self.data.candidate_units_dict[('PARAMETERS', 'FOM')][g])

        # Fixed operating and maintenance cost for candidate generators (assuming no unit retirement for now, so can
        # excluded consideration of FOM costs for existing units)
        m.C_FOM = Param(m.G_C_THERM.union(m.G_C_WIND).union(m.G_C_SOLAR).union(m.G_C_STORAGE), rule=fom_cost_rule)

        def investment_cost_rule(m, g, i):
            """Cost to build 1MW of the candidate technology"""

            if g in m.G_C_STORAGE:
                # Build costs for batteries
                return float(self.data.battery_build_costs_dict[i][g] * 1000)
            else:
                # Build costs for other candidate units
                return float(self.data.candidate_units_dict[('BUILD_COST', i)][g] * 1000)

        # Investment / build cost per MW for candidate technologies
        m.C_INV = Param(m.G_C.union(m.G_C_STORAGE), m.I, rule=investment_cost_rule)

        return m

    @staticmethod
    def define_variables(m):
        """Define model variables common to all blocks"""

        # Capacity of candidate units (defined for all years in model horizon)
        m.x_c = Var(m.G_C_WIND.union(m.G_C_SOLAR).union(m.G_C_STORAGE), m.I, within=NonNegativeReals, initialize=0)

        # Binary variable used to determine size of candidate thermal units
        m.d = Var(m.G_C_THERM, m.I, m.G_C_THERM_SIZE_OPTIONS, within=NonNegativeReals, initialize=0)

        # Emissions intensity baseline [tCO2/MWh] (must be fixed each time model is run)
        m.baseline = Var(m.I, initialize=0)

        # Permit price [$/tCO2] (must be fixed each time model is run)
        m.permit_price = Var(m.I, initialize=0)

        # Amount by which emissions target is exceeded
        m.emissions_target_exceeded = Var(within=NonNegativeReals, initialize=0)

        # Amount by which scheme revenue falls short of target
        m.revenue_shortfall = Var(within=NonNegativeReals, initialize=0)

        return m

    def define_expressions(self, m):
        """Define expressions common to both master and subproblem"""

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

        # Penalty imposed for violating emissions constraint
        m.C_EMISSIONS_VIOLATION = Expression(expr=m.emissions_target_exceeded * m.EMISSIONS_EXCEEDED_PENALTY)

        # Penalty imposed for violating revenue constraint
        m.C_REVENUE_VIOLATION = Expression(expr=m.revenue_shortfall * m.REVENUE_SHORTFALL_PENALTY)

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

        return m