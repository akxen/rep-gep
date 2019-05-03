import os
from math import ceil

import pandas as pd
from pyomo.environ import *


# Class to instantiate UC model, with methods to solve the model, update
# parameters, and fix variables.

# Class to run the UC model using a rolling window approach. Accepts as
# arguments data dictionaries describing demand traces, solar traces, wind
# traces, and hydro output.


class UnitCommitmentModel:
    def __init__(self, data_dir, input_traces_dir):
        # Directory containing core data files
        self.data_dir = data_dir

        # Directory containing input solar, wind, hydro, and demand traces
        self.input_traces_dir = input_traces_dir

        # Input traces
        self.input_traces = pd.read_pickle(os.path.join(input_traces_dir, 'centroids.pickle'))

        # Candidate units
        self.candidate_units = self._load_candidate_units()

        # Existing units
        self.existing_units = self._load_existing_units()

        # Battery build costs
        self.battery_build_costs = self._load_battery_build_costs()

        # Battery properties
        self.battery_properties = pd.read_csv(os.path.join(data_dir, 'battery_properties.csv'), header=0, index_col=0)

        # Battery build costs
        self.candidate_unit_build_limits = pd.read_csv(os.path.join(data_dir, 'candidate_unit_build_limits.csv'),
                                                       header=0, index_col=0)

        # Region minimum reserve levels
        self.minimum_reserve_levels = pd.read_csv(os.path.join(data_dir, 'minimum_reserve_levels.csv'),
                                                  header=0, index_col=0)

        # Initialise unit commitment model
        self.model = None

    @staticmethod
    def _col_mapper(text):
        """Try and convert text in columns to int if possible"""
        try:
            output = int(text)
        except ValueError:
            output = text
        return output

    def _load_existing_units(self):
        """Load existing unit information"""

        # Load CSV
        df = pd.read_csv(os.path.join(self.data_dir, 'existing_units.csv'), header=[0, 1], index_col=0)

        # Rename columns
        df = df.rename(self._col_mapper, axis='columns', level=1)

        return df

    def _load_candidate_units(self):
        """Load candidate unit information"""

        # Load CSV
        df = pd.read_csv(os.path.join(self.data_dir, 'candidate_units.csv'), header=[0, 1], index_col=0)

        # Rename columns
        df = df.rename(self._col_mapper, axis='columns', level=1)

        return df

    def _load_battery_build_costs(self):
        """Load battery build costs"""

        # Load CSV
        df = pd.read_csv(os.path.join(self.data_dir, 'battery_build_costs.csv'), header=0, index_col=0)

        # Rename columns
        df = df.rename(self._col_mapper, axis='columns')

        return df

    def _get_nem_zones(self):
        """Get list of unique NEM zones"""

        # Extract nem zones from existing generators dataset
        zones = self.existing_units.loc[:, ('PARAMETERS', 'NEM_ZONE')].unique()

        # There should be 16 zones
        assert len(zones) == 16, 'Unexpected number of NEM zones'

        return zones

    def _get_nem_regions(self):
        """Get list of unique NEM regions"""

        # Extract nem regions from existing generators dataset
        regions = self.existing_units.loc[:, ('PARAMETERS', 'NEM_REGION')].unique()

        # There should be 16 zones
        assert len(regions) == 5, 'Unexpected number of NEM regions'

        return regions

    def _get_wind_bubbles(self):
        """Get unique wind bubbles"""

        # Extract wind bubbles from input data traces file
        df = self.input_traces.copy()

        # Unique wind bubbles
        bubbles = list(set([i[0] for i in df.loc[:, 'WIND'].columns.values]))

        return bubbles

    def _get_candidate_thermal_unit_ids(self):
        """Get all IDs for candidate thermal units"""

        # Get candidate thermal units
        mask_candidate_thermal = self.candidate_units[('PARAMETERS', 'TECHNOLOGY_PRIMARY')].isin(['GAS', 'COAL'])

        # IDs for candidate thermal units
        candidate_thermal_ids = self.candidate_units[mask_candidate_thermal].index

        return candidate_thermal_ids

    def _get_candidate_solar_unit_ids(self):
        """Get IDs for candidate solar units"""

        # Filter candidate solar units
        mask_candidate_solar = self.candidate_units[('PARAMETERS', 'TECHNOLOGY_PRIMARY')].isin(['SOLAR'])

        # IDs for existing thermal units
        candidate_solar_ids = self.candidate_units[mask_candidate_solar].index

        return candidate_solar_ids

    def _get_candidate_wind_unit_ids(self):
        """Get IDs for candidate wind units"""

        # Filter candidate wind units
        mask_candidate_wind = self.candidate_units[('PARAMETERS', 'TECHNOLOGY_PRIMARY')].isin(['WIND'])

        # IDs for existing thermal units
        candidate_wind_ids = self.candidate_units[mask_candidate_wind].index

        return candidate_wind_ids

    def _get_existing_thermal_unit_ids(self):
        """Get IDs for existing thermal units"""

        # Filter existing thermal units
        mask_existing_thermal = self.existing_units[('PARAMETERS', 'TECHNOLOGY_CAT_PRIMARY')].isin(
            ['GAS', 'COAL', 'LIQUID'])

        # IDs for existing thermal units
        existing_thermal_ids = self.existing_units[mask_existing_thermal].index

        return existing_thermal_ids

    def _get_existing_solar_unit_ids(self):
        """Get IDs for existing solar units"""

        # Filter existing solar units
        mask_existing_solar = self.existing_units[('PARAMETERS', 'TECHNOLOGY_CAT_PRIMARY')].isin(['SOLAR'])

        # IDs for existing solar units
        existing_solar_ids = self.existing_units[mask_existing_solar].index

        return existing_solar_ids

    def _get_existing_wind_unit_ids(self):
        """Get IDs for existing wind units"""

        # Filter existing wind units
        mask_existing_wind = self.existing_units[('PARAMETERS', 'TECHNOLOGY_CAT_PRIMARY')].isin(['WIND'])

        # IDs for existing wind units
        existing_wind_ids = self.existing_units[mask_existing_wind].index

        return existing_wind_ids

    def _get_existing_hydro_unit_ids(self):
        """Get IDs for existing hydro units"""

        # Filter existing hydro units
        mask_existing_hydro = self.existing_units[('PARAMETERS', 'TECHNOLOGY_CAT_PRIMARY')].isin(['HYDRO'])

        # IDs for existing hydro units
        existing_hydro_ids = self.existing_units[mask_existing_hydro].index

        return existing_hydro_ids

    def _get_candidate_storage_units(self):
        """Get IDs for candidate storage units"""

        # IDs for candidate storage units
        candidate_storage_ids = self.battery_properties.index

        return candidate_storage_ids

    def _get_slow_start_thermal_generator_ids(self):
        """
        Get IDs for existing and candidate slow start unit

        A generator is classified as 'slow' if it cannot reach its
        minimum dispatchable power output in one interval (e.g. 1 hour).

        Note: A generator's classification of 'quick' or 'slow' depends on its
        minimum dispatchable output level and ramp-rate. For candidate units
        the minimum dispatchable output level is a function of the maximum
        output level, and so is variable. As this level is not known ex ante,
        all candidate thermal generators are assumed to operate the same way
        as quick start units (i.e. they can reach their minimum dispatchable
        output level in 1 trading interval (hour)).
        """

        # True if number of hours to ramp to min generator output > 1
        mask_slow_start = (self.existing_units[('PARAMETERS', 'MIN_GEN')]
                           .div(self.existing_units[('PARAMETERS', 'RR_STARTUP')])
                           .gt(1))

        # Only consider coal and gas units
        mask_technology = self.existing_units[('PARAMETERS', 'TECHNOLOGY_CAT_PRIMARY')].isin(['COAL', 'GAS', 'LIQUID'])

        # Get IDs for slow start generators
        gen_ids = self.existing_units.loc[mask_slow_start & mask_technology, :].index

        return gen_ids

    def _get_quick_start_thermal_generator_ids(self):
        """
        Get IDs for existing and candidate slow start unit

        Note: A generator is classified as 'quick' if it can reach its
        minimum dispatchable power output in one interval (e.g. 1 hour).
        """

        # Slow start unit IDs - previously identified
        slow_gen_ids = self._get_slow_start_thermal_generator_ids()

        # Filter for slow generator IDs (existing units)
        mask_slow_gen_ids = self.existing_units.index.isin(slow_gen_ids)

        # Only consider coal and gas units
        mask_existing_technology = (self.existing_units[('PARAMETERS', 'TECHNOLOGY_CAT_PRIMARY')]
                                    .isin(['COAL', 'GAS', 'LIQUID']))

        # Get IDs for quick start generators
        existing_quick_gen_ids = self.existing_units.loc[~mask_slow_gen_ids & mask_existing_technology, :].index

        # Get IDs for quick start candidate generators (all candidate coal / gas generators)
        mask_candidate_technology = (self.candidate_units[('PARAMETERS', 'TECHNOLOGY_PRIMARY')].isin(['COAL', 'GAS']))

        # Quick start candidate generator IDs
        candidate_quick_gen_ids = self.candidate_units.loc[mask_candidate_technology].index

        # All quick start generator IDs (existing and candidate)
        gen_ids = existing_quick_gen_ids.union(candidate_quick_gen_ids)

        return gen_ids

    def _get_network_incidence_matrix(self):
        """Construct network incidence matrix"""

        # All NEM zones:
        zones = self._get_nem_zones()

        # Links connecting different zones. First zone is 'from' zone second is 'to' zone
        links = ['NQ-CQ', 'CQ-SEQ', 'CQ-SWQ', 'SWQ-SEQ', 'SEQ-NNS',
                 'SWQ-NNS', 'NNS-NCEN', 'NCEN-CAN', 'CAN-SWNSW',
                 'CAN-NVIC', 'SWNSW-NVIC', 'LV-MEL', 'NVIC-MEL',
                 'TAS-LV', 'MEL-CVIC', 'SWNSW-CVIC', 'CVIC-NSA',
                 'MEL-SESA', 'SESA-ADE', 'NSA-ADE']

        # Initialise empty matrix with NEM zones as row and column labels
        incidence_matrix = pd.DataFrame(index=links, columns=zones, data=0)

        # Assign values to 'from' and 'to' zones. +1 is a 'from' zone, -1 is a 'to' zone
        for link in links:
            # Get from and to zones
            from_zone, to_zone = link.split('-')

            # Set from zone element to 1
            incidence_matrix.loc[link, from_zone] = 1

            # Set to zone element to -1
            incidence_matrix.loc[link, to_zone] = -1

        return incidence_matrix

    def _get_network_links(self):
        """Links connecting adjacent NEM zones"""

        return self._get_network_incidence_matrix().index

    @staticmethod
    def _get_link_power_flow_limits():
        """Max forward and reverse power flow over links between zones"""

        # Limits for interconnectors composed of single branches
        interconnector_limits = {'SEQ-NNS': {'forward': 210, 'reverse': 107},  # Terranora
                                 'SWQ-NNS': {'forward': 1078, 'reverse': 600},  # QNI
                                 'TAS-LV': {'forward': 594, 'reverse': 478},  # Basslink
                                 'MEL-SESA': {'forward': 600, 'reverse': 500},  # Heywood
                                 'CVIC-NSA': {'forward': 220, 'reverse': 200},  # Murraylink
                                 }

        return interconnector_limits
    #
    # def _get_build_limit_technologies(self):
    #     """"""

    def construct_model(self, year=2016):
        """
        Initialise unit commitment model
        """
        # Create concrete model
        m = ConcreteModel()

        # Sets
        # ----
        # NEM regions
        m.R = Set(initialize=self._get_nem_regions())

        # NEM zones
        m.Z = Set(initialize=self._get_nem_zones())

        # Links between NEM zones
        m.L = Set(initialize=self._get_network_links())

        # NEM wind bubbles
        m.B = Set(initialize=self._get_wind_bubbles())

        # Existing thermal units
        m.G_E_THERM = Set(initialize=self._get_existing_thermal_unit_ids())

        # Candidate thermal units
        m.G_C_THERM = Set(initialize=self._get_candidate_thermal_unit_ids())

        # Index for candidate thermal unit size options
        m.G_C_THERM_SIZE_OPTIONS = RangeSet(0, 3, ordered=True)

        # Existing wind units
        m.G_E_WIND = Set(initialize=self._get_existing_wind_unit_ids())

        # Candidate wind units
        m.G_C_WIND = Set(initialize=self._get_candidate_wind_unit_ids())

        # Existing solar units
        m.G_E_SOLAR = Set(initialize=self._get_existing_solar_unit_ids())

        # Candidate solar units
        m.G_C_SOLAR = Set(initialize=self._get_candidate_solar_unit_ids())

        # Existing hydro units
        m.G_E_HYDRO = Set(initialize=self._get_existing_hydro_unit_ids())

        # Existing storage units
        # m.S_E = Set()

        # Candidate storage units
        m.G_C_STORAGE = Set(initialize=self._get_candidate_storage_units())

        # Slow start thermal generators (existing and candidate)
        m.G_THERM_SLOW = Set(initialize=self._get_slow_start_thermal_generator_ids())

        # Quick start thermal generators (existing and candidate)
        m.G_THERM_QUICK = Set(initialize=self._get_quick_start_thermal_generator_ids())

        # All existing generators
        m.G_E = m.G_E_THERM.union(m.G_E_WIND).union(m.G_E_SOLAR).union(m.G_E_HYDRO)

        # All candidate generators + storage units
        m.G_C = m.G_C_THERM.union(m.G_C_WIND).union(m.G_C_SOLAR).union(m.G_C_STORAGE)

        # All generators + storage units
        m.G = m.G_E.union(m.G_C)

        # Operating scenario hour
        m.T = RangeSet(0, 23, ordered=True)

        # Years since start of model horizon and year for which model is being run
        m.Y = RangeSet(2016, year)

        # All years in model horizon
        m.I = RangeSet(2016, 2049)

        # Build limit technology types
        m.BUILD_LIMIT_TECHNOLOGIES = Set(initialize=self.candidate_unit_build_limits.index)

        # Parameters
        # ----------
        def candidate_thermal_size_options_rule(m, g, n):
            """Candidate thermal unit discrete size options"""

            if self.candidate_units.loc[g, ('PARAMETERS', 'TECHNOLOGY_PRIMARY')] == 'GAS':

                if n == 0:
                    return float(0)
                elif n == 1:
                    return float(100)
                elif n == 2:
                    return float(200)
                elif n == 3:
                    return float(300)
                else:
                    raise Exception('Unexpected size index')

            elif self.candidate_units.loc[g, ('PARAMETERS', 'TECHNOLOGY_PRIMARY')] == 'COAL':

                if n == 0:
                    return float(0)
                elif n == 1:
                    return float(300)
                elif n == 2:
                    return float(500)
                elif n == 3:
                    return float(700)
                else:
                    raise Exception('Unexpected size index')

            else:
                raise Exception('Unexpected technology type')

        # Possible size for thermal generators
        m.X_C_THERM = Param(m.G_C_THERM, m.G_C_THERM_SIZE_OPTIONS, rule=candidate_thermal_size_options_rule)

        def emissions_intensity_rule(m, g):
            """Emissions intensity (tCO2/MWh)"""

            if g in m.G_E_THERM:
                # Emissions intensity
                emissions = self.existing_units.loc[g, ('PARAMETERS', 'EMISSIONS')].astype(float)

            elif g in m.G_C_THERM:
                # Emissions intensity
                emissions = self.candidate_units.loc[g, ('PARAMETERS', 'EMISSIONS')].astype(float)

            else:
                # Set emissions intensity = 0 for all solar, wind, hydro, and storage units
                emissions = float(0)

            return emissions

        # Emissions intensities for all generators
        m.E = Param(m.G, rule=emissions_intensity_rule)

        def startup_cost_rule(m, g):
            """
            Startup costs for existing and candidate thermal units

            Note: costs are normalised by plant capacity e.g. $/MW
            """

            if g in m.G_E_THERM:
                # Startup cost for existing thermal units
                startup_cost = (self.existing_units.loc[g, ('PARAMETERS', 'SU_COST_WARM')] /
                                self.existing_units.loc[g, ('PARAMETERS', 'REG_CAP')])

                # Convert to float
                startup_cost = float(startup_cost)

            elif g in m.G_C_THERM:
                # Startup cost for candidate thermal units
                startup_cost = self.candidate_units.loc[g, ('PARAMETERS', 'SU_COST_WARM_MW')].astype(float)

            else:
                # Assume startup cost = 0 for all solar, wind, hydro generators
                startup_cost = float(0)

            # Shutdown cost cannot be negative
            assert startup_cost >= 0, 'Negative startup cost'

            return startup_cost

        # Generator startup costs
        m.C_SU = Param(m.G, rule=startup_cost_rule)

        def shutdown_cost_rule(m, g):
            """
            Shutdown costs for existing and candidate thermal units

            Note: costs are normalised by plant capacity e.g. $/MW

            No data for shutdown costs, so assume it is half the value of
            startup cost for given generator.
            """

            if g in m.G_E_THERM:
                # Shutdown cost for existing thermal units
                shutdown_cost = (self.existing_units.loc[g, ('PARAMETERS', 'SU_COST_WARM')] /
                                 self.existing_units.loc[g, ('PARAMETERS', 'REG_CAP')]) / 2

                # Convert to float
                shutdown_cost = float(shutdown_cost)

            elif g in m.G_C_THERM:
                # Shutdown cost for candidate thermal units
                shutdown_cost = self.candidate_units.loc[g, ('PARAMETERS', 'SU_COST_WARM_MW')] / 2

                # Convert to float
                shutdown_cost = float(shutdown_cost)

            else:
                # Assume shutdown cost = 0 for all solar, wind, hydro generators
                shutdown_cost = float(0)

            # Shutdown cost cannot be negative
            assert shutdown_cost >= 0, 'Negative shutdown cost'

            return shutdown_cost

        # Generator shutdown costs
        m.C_SD = Param(m.G, rule=shutdown_cost_rule)

        def minimum_region_up_reserve_rule(m, r):
            """Minimum upward reserve rule"""

            # Minimum upward reserve for region
            up_reserve = self.minimum_reserve_levels.loc[r, 'MINIMUM_RESERVE_LEVEL'].astype(float)

            return up_reserve

        # Minimum upward reserve
        m.D_UP = Param(m.R, rule=minimum_region_up_reserve_rule)

        def startup_ramp_rate_rule(m, g):
            """Startup ramp-rate (MW)"""

            if g in m.G_E_THERM:
                # Startup ramp-rate for existing thermal generators
                startup_ramp = self.existing_units.loc[g, ('PARAMETERS', 'RR_STARTUP')].astype(float)

            elif g in m.G_C_THERM:
                # Startup ramp-rate for candidate thermal generators
                startup_ramp = self.candidate_units.loc[g, ('PARAMETERS', 'RR_STARTUP')].astype(float)

            else:
                raise Exception(f'Unexpected generator {g}')

            return startup_ramp

        # Startup ramp-rate for existing and candidate thermal generators
        m.SU_RAMP = Param(m.G_E_THERM.union(m.G_C_THERM), rule=startup_ramp_rate_rule)

        def shutdown_ramp_rate_rule(m, g):
            """Shutdown ramp-rate (MW)"""

            if g in m.G_E_THERM:
                # Shutdown ramp-rate for existing thermal generators
                shutdown_ramp = self.existing_units.loc[g, ('PARAMETERS', 'RR_SHUTDOWN')].astype(float)

            elif g in m.G_C_THERM:
                # Shutdown ramp-rate for candidate thermal generators
                shutdown_ramp = self.candidate_units.loc[g, ('PARAMETERS', 'RR_SHUTDOWN')].astype(float)

            else:
                raise Exception(f'Unexpected generator {g}')

            return shutdown_ramp

        # Shutdown ramp-rate for existing and candidate thermal generators
        m.SD_RAMP = Param(m.G_E_THERM.union(m.G_C_THERM), rule=shutdown_ramp_rate_rule)

        def ramp_rate_up_rule(m, g):
            """Ramp-rate up (MW/h) - when running"""

            if g in m.G_E_THERM:
                # Ramp-rate up for existing generators
                ramp_up = self.existing_units.loc[g, ('PARAMETERS', 'RR_UP')].astype(float)

            elif g in m.G_C_THERM:
                # Ramp-rate up for candidate generators
                ramp_up = self.candidate_units.loc[g, ('PARAMETERS', 'RR_UP')].astype(float)

            else:
                raise Exception(f'Unexpected generator {g}')

            return ramp_up

        # Ramp-rate up (normal operation)
        m.RU = Param(m.G_E_THERM.union(m.G_C_THERM), rule=ramp_rate_up_rule)

        def ramp_rate_down_rule(m, g):
            """Ramp-rate down (MW/h) - when running"""

            if g in m.G_E_THERM:
                # Ramp-rate down for existing generators
                ramp_down = self.existing_units.loc[g, ('PARAMETERS', 'RR_DOWN')].astype(float)

            elif g in m.G_C_THERM:
                # Ramp-rate down for candidate generators
                ramp_down = self.candidate_units.loc[g, ('PARAMETERS', 'RR_DOWN')].astype(float)

            else:
                raise Exception(f'Unexpected generator {g}')

            return ramp_down

        # Ramp-rate down (normal operation)
        m.RD = Param(m.G_E_THERM.union(m.G_C_THERM), rule=ramp_rate_down_rule)

        def max_generator_power_output_rule(m, g):
            """
            Maximum power output from existing and candidate generators

            Note: candidate units will have their max power output determined by investment decisions which
            are made known in the master problem. Need to update these values each time model is run. Initialise
            max power output = 0.
            """

            if g in m.G_E:
                max_power = self.existing_units.loc[g, ('PARAMETERS', 'REG_CAP')].astype(float)
            else:
                max_power = float(0)
            return max_power

        # Maximum power output for existing and candidate units (must be updated each time model is run)
        m.P_MAX = Param(m.G, rule=max_generator_power_output_rule)

        def min_power_output_proportion_rule(m, g):
            """Minimum generator power output as a proportion of maximum output"""

            if g in m.G_E:
                # Minimum power output for existing generators
                min_output = (self.existing_units.loc[g, ('PARAMETERS', 'MIN_GEN')] /
                              self.existing_units.loc[g, ('PARAMETERS', 'REG_CAP')])

            elif g in m.G_C_THERM:
                # Minimum power output for candidate thermal generators
                min_output = self.candidate_units.loc[g, ('PARAMETERS', 'MIN_GEN_PERCENT')] / 100

            else:
                # Minimum power output = 0
                min_output = 0

            # Convert to float
            min_output = float(min_output)

            return min_output

        # Minimum power output (as a proportion of max capacity) for existing and candidate thermal generators
        m.P_MIN_PROP = Param(m.G, rule=min_power_output_proportion_rule)

        def marginal_cost_rule(m, g):
            """Marginal costs for existing and candidate generators

            Note: Marginal costs for existing and candidate thermal plant depend on fuel costs, which are time
            varying. Therefore marginal costs for thermal plant must be updated each time the model is run.
            These costs have been initialised to zero.
            """

            if (g in m.G_E_THERM) or (g in m.G_C_THERM):
                #  Initialise marginal cost for existing and candidate thermal plant = 0
                marginal_cost = float(0)

            elif (g in m.G_E_WIND) or (g in m.G_E_SOLAR) or (g in m.G_E_HYDRO):
                # Marginal cost = VOM cost for wind and solar generators
                marginal_cost = self.existing_units.loc[g, ('PARAMETERS', 'VOM')].astype(float)

            elif (g in m.G_C_WIND) or (g in m.G_C_SOLAR):
                # Marginal cost = VOM cost for wind and solar generators
                marginal_cost = self.candidate_units.loc[g, ('PARAMETERS', 'VOM')].astype(float)

            elif g in m.G_C_STORAGE:
                # Assume marginal cost = VOM cost of typically hydro generator (7 $/MWh)
                marginal_cost = float(7)

            else:
                raise Exception(f'Unexpected generator: {g}')

            assert marginal_cost >= 0, 'Cannot have negative marginal cost'

            return marginal_cost

        # Marginal costs for all generators (must be updated each time model is run)
        m.C_MC = Param(m.G, rule=marginal_cost_rule, mutable=True)

        def battery_efficiency_rule(m, g):
            """Battery efficiency"""

            return self.battery_properties.loc[g, 'CHARGE_EFFICIENCY'].astype(float)

        # Battery efficiency
        m.BATTERY_EFFICIENCY = Param(m.G_C_STORAGE, rule=battery_efficiency_rule)

        def network_incidence_matrix_rule(m, l, z):
            """Incidence matrix describing connections between adjacent NEM zones"""

            # Network incidence matrix
            df = self._get_network_incidence_matrix()

            return df.loc[l, z].astype(float)

        # Network incidence matrix
        m.INCIDENCE_MATRIX = Param(m.L, m.Z, rule=network_incidence_matrix_rule)

        def build_limits_rule(m, technology, z):
            """Build limits for each technology type by zone"""

            return self.candidate_unit_build_limits.loc[technology, z].astype(float)

        # Build limits for each technology and zone
        m.BUILD_LIMITS = Param(m.BUILD_LIMIT_TECHNOLOGIES, m.Z, rule=build_limits_rule)

        # Capacity factor wind (must be updated each time model is run)
        m.Q_WIND = Param(m.B, m.T, initialize=0, mutable=True)

        # Capacity factor solar (must be updated each time model is run)
        m.Q_SOLAR = Param(m.Z, m.T, initialize=0, mutable=True)

        # Zone demand (must be updated each time model is run)
        m.D = Param(m.Z, m.T, initialize=0, mutable=True)

        # Max MW out of storage device - discharging (must be updated each time model is run)
        m.P_STORAGE_MAX_OUT = Param(m.G_C_STORAGE, initialize=0)

        # Max MW into storage device - charging (must be updated each time model is run)
        m.P_STORAGE_MAX_IN = Param(m.G_C_STORAGE, initialize=0)

        # Duration of operating scenario (must be updated each time model is run)
        m.SCENARIO_DURATION = Param(initialize=0)

        # Value of lost-load [$/MWh]
        m.C_L = Param(initialize=float(1e4), mutable=True)

        # Variables
        # ---------
        # Emissions intensity baseline [tCO2/MWh] (must be fixed each time model is run)
        m.baseline = Var()

        # Permit price [$/tCO2] (must be fixed each time model is run)
        m.permit_price = Var()

        # Power output above minimum dispatchable level of output [MW]
        m.p = Var(m.G, m.T)

        # Variables to be determined in master program (will be fixed in sub-problems)
        # ----------------------------------------------------------------------------
        # Discrete investment decisions for candidate thermal generators
        m.d = Var(m.G_C_THERM, m.G_C_THERM_SIZE_OPTIONS, m.I, within=Binary)

        # Capacity of candidate units (defined for all years in model horizon)
        m.x_C = Var(m.G_C, m.I, within=NonNegativeReals)

        # Expressions
        # -----------
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

            return m.P_MIN[g] + m.p[g, t]

        # Total power output [MW]
        m.P_TOTAL = Expression(m.G, m.T, rule=total_power_output_rule)

        def generator_energy_output_rule(m, g, t):
            """
            Total generator energy output [MWh]

            No information regarding generator dispatch level in period prior to
            t=0. Assume that all generators are at their minimum dispatch level.
            """

            # TODO: Must take into account energy output for storage units
            if t != m.T.first():
                return (m.P_TOTAL[g, t-1] + m.P_TOTAL[g, t]) / 2
            else:
                return (m.P_MIN[g] + m.P_TOTAL[g, t]) / 2

        # Energy output for a given generator
        m.ENERGY_OUTPUT = Expression(m.G, m.T, rule=generator_energy_output_rule)

        # Constraints
        # -----------
        def candidate_thermal_capacity_rule(m, g, i):
            """Discrete candidate thermal unit investment decisions"""

            return m.x_C[g, i] == sum(m.d[g, n, i] * m.X_C_THERM[g, n] for n in m.G_C_THERM_SIZE_OPTIONS)

        # Constraining discrete investment sizes for thermal plant
        m.DISCRETE_THERMAL_OPTIONS = Constraint(m.G_C_THERM, m.I, rule=candidate_thermal_capacity_rule)

        def unique_discrete_choice_rule(m, g, i):
            """Can only choose one size per technology-year"""

            return sum(m.d[g, n, i] for n in m.G_C_THERM_SIZE_OPTIONS) == 1

        # Can only choose one size option in each year for each candidate size
        m.UNIQUE_SIZE_CHOICE = Constraint(m.G_C_THERM, m.I, rule=unique_discrete_choice_rule)







        # Update class attribute
        self.model = m

    def fix_rolling_window_variables(self):
        """
        Fix variables at the start of the rolling window (initial conditions)
        """
        pass

    def unfix_rolling_window_variables(self):
        """
        Free variables at the start of the rolling window
        """
        pass

    def fix_binary_variables(self):
        """
        Fix all binary variables. Allows the linear approximation of the UC
        program to be solved.
        """
        pass

    def unfix_binary_variables(self):
        """
        Free all binary variables.
        """
        pass

    def update_parameters(self, start, end, fix_intervals):
        """
        Update model parameters. Fix variables at the start of the rolling
        window to specified values.

        Parameters
        ----------
        start : datetime
            First interval within rolling window horizon

        end : datetime
            Last interval within rolling window horizon

        fix_intervals : datetime
            Number of intervals to fix at the start of the rolling window
        """
        pass

    def solve_model(self):
        """
        Solve model. First solve the MILP problem. Then fix binary variables
        to the solution obtained. Re-run the model, allowing dual information
        to be collected.
        """

        # Solve MILP problem

        # Fix binary variables

        # Solve LP problem (obtain dual variables)

        # Get summary of results that will be passed to master problem

        # Unfix binary variables

        pass

    def solve_rolling_window(self, start, end, overlap):
        """
        Solve model using rolling-window approach.

        Parameters
        ----------
        start : datetime
            Timestamp for start of rolling window

        end : datetime
            Timestamp for end of rolling window

        overlap : int
            Number of intervals successive windows must overlap with the
            previous window
        """

        # Compute parameters for each window (timestamp for start of window,
        # timestamp for end of window, number of intervals that must be
        # fixed at the beginning of each rolling window with values obtained
        # from the solution of the previous window).

        # Initialise container for model results

        # For window in windows

        # Solve the MILP problem

        # Fix binary variables

        # Solve the LP problem

        # Get summary of results for selected variables and expressions
        # (use this information when updating master problem)

        # Append results to container

        # Parse results so they are in a format which can be easily ingested
        # by the master problem

        pass

    def get_result_summary(self):
        """
        Summarise results for particular primal and dual variables, and
        evaluate selected expressions.

        Returns
        -------
        result_summary : dict
            Summary of model results
        """
        pass


if __name__ == '__main__':
    # Directory containing core data files
    data_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '1_collect_data', 'output')

    # Directory containing input traces
    input_traces_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '2_input_traces', 'output')

    # Instantiate UC model
    UC = UnitCommitmentModel(data_directory, input_traces_directory)

    # Construct model
    UC.construct_model()
