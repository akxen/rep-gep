import os
import pandas as pd
from math import ceil
from collections import OrderedDict

from pyomo.environ import *


class UnitCommitmentModel:
    def __init__(self, raw_data_dir, data_dir, input_traces_dir):
        # Initialise model object
        self.m = ConcreteModel()

        # Directory containing raw data files
        self.raw_data_dir = raw_data_dir

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
        self.battery_properties = self._load_battery_properties()

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

    def _load_battery_properties(self):
        """Load battery properties"""

        # Load battery properties from spreadsheet
        df = pd.read_csv(os.path.join(self.data_dir, 'battery_properties.csv'), header=0, index_col=0)

        # Add NEM zone as column
        df['NEM_ZONE'] = df.apply(lambda x: x.name.split('-')[0], axis=1)

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

    def _get_nem_region_zone_map(self):
        """
        Construct mapping between NEM regions and the zones belonging to
        those regions
        """

        # Map between NEM regions and zones
        region_zone_map = (self.existing_units[[('PARAMETERS', 'NEM_REGION'), ('PARAMETERS', 'NEM_ZONE')]]
                           .droplevel(0, axis=1).drop_duplicates(subset=['NEM_REGION', 'NEM_ZONE'])
                           .groupby('NEM_REGION')['NEM_ZONE'].apply(lambda x: list(x)))

        return region_zone_map

    def _get_wind_bubble_map(self):
        """Mapping between wind bubbles and NEM regions and zones"""

        # Load data
        df = pd.read_csv(os.path.join(self.raw_data_dir, 'maps', 'wind_bubbles.csv'), index_col=0)

        return df

    def _get_wind_bubbles(self):
        """
        Get unique wind bubbles

        Note: Not all wind bubbles in dataset are used. Some zones may have more
        than one wind bubble. Model arbitrarily selects bubbles so there is at
        most one bubble per zone.
        """

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

    def _get_generator_zone_map(self):
        """Get mapping between existing and candidate generators and NEM zones"""

        # Existing units
        existing = self.existing_units[('PARAMETERS', 'NEM_ZONE')]

        # Candidate units
        candidate = self.candidate_units[('PARAMETERS', 'ZONE')]

        # Candidate storage units
        storage = self.battery_properties['NEM_ZONE']

        # Concatenate series objects
        generator_zone_map = pd.concat([existing, candidate, storage])

        return generator_zone_map

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

    def define_sets(self, m):
        """Define sets to be used in model"""

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

        # All years in model horizon
        m.I = RangeSet(2016, 2049)

        # Build limit technology types
        m.BUILD_LIMIT_TECHNOLOGIES = Set(initialize=self.candidate_unit_build_limits.index)

        return m

    def define_parameters(self, m):
        """Define model parameters"""

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

        def wind_capacity_factor_rule(m, g):
            """Capacity factors for each existing and candidate wind generator"""

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

        # Initial state for thermal generators
        # TODO: If unit retires it should be set to 0. Could also consider baseload as being on.
        m.u0 = Param(m.G_E_THERM.union(m.G_C_THERM), initialize=1, mutable=True)

        return m

    @staticmethod
    def define_variables(m):
        """Define model variables"""

        # Emissions intensity baseline [tCO2/MWh] (must be fixed each time model is run)
        m.baseline = Var()

        # Permit price [$/tCO2] (must be fixed each time model is run)
        m.permit_price = Var()

        # Power output above minimum dispatchable level of output [MW]
        m.p = Var(m.G, m.T, within=NonNegativeReals)

        # Startup state variable
        m.v = Var(m.G_E_THERM.union(m.G_C_THERM), m.T, within=Binary)

        # Shutdown state variable
        m.w = Var(m.G_E_THERM.union(m.G_C_THERM), m.T, within=Binary)

        # On-state variable
        m.u = Var(m.G_E_THERM.union(m.G_C_THERM), m.T, within=Binary)

        # Upward reserve allocation [MW]
        m.r_up = Var(m.G_E_THERM.union(m.G_C_THERM).union(m.G_C_STORAGE), m.T, within=NonNegativeReals, initialize=0)

        # Variables to be determined in master program (will be fixed in sub-problems)
        # ----------------------------------------------------------------------------
        # Capacity of candidate units (defined for all years in model horizon)
        m.x_C = Var(m.G_C, m.I, within=NonNegativeReals)

        return m

    @staticmethod
    def define_expressions(m):
        """Define model expressions"""

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
                    return (m.P_MIN[g] * (m.u[g, t] + m.v[g, t + 1])) + m.p[g, t]
                else:
                    return (m.P_MIN[g] * m.u[g, t]) + m.p[g, t]

            # Define startup and shutdown trajectories for 'slow' generators
            elif g in m.G_THERM_SLOW:
                # Startup duration
                SU_D = ceil(m.P_MIN[g].expr() / m.SU_RAMP[g])

                # Startup power output trajectory increment
                ramp_up_increment = m.P_MIN[g].expr() / SU_D

                # Startup power output trajectory
                P_SU = OrderedDict({i + 1: ramp_up_increment * i for i in range(0, SU_D + 1)})

                # Shutdown duration
                SD_D = ceil(m.P_MIN[g].expr() / m.SD_RAMP[g])

                # Shutdown power output trajectory increment
                ramp_down_increment = m.P_MIN[g].expr() / SD_D

                # Shutdown power output trajectory
                P_SD = OrderedDict({i + 1: m.P_MIN[g].expr() - (ramp_down_increment * i) for i in range(0, SD_D + 1)})

                if t < m.T.last():
                    return ((m.P_MIN[g] * (m.u[g, t] + m.v[g, t + 1])) + m.p[g, t]
                            + sum(P_SU[i] * m.v[g, t - i + SU_D + 2] if t - i + SU_D + 2 in m.T else 0 for i in
                                  range(1, SU_D + 1))
                            + sum(P_SD[i] * m.w[g, t - i + 2] if t - i + 2 in m.T else 0 for i in range(2, SD_D + 2))
                            )
                else:
                    return ((m.P_MIN[g] * m.u[g, t]) + m.p[g, t]
                            + sum(P_SU[i] * m.v[g, t - i + SU_D + 2] if t - i + SU_D + 2 in m.T else 0 for i in
                                  range(1, SU_D + 1))
                            + sum(P_SD[i] * m.w[g, t - i + 2] if t - i + 2 in m.T else 0 for i in range(2, SD_D + 2))
                            )

            # Remaining generators with no startup / shutdown directory defined
            else:
                return m.P_MIN[g] + m.p[g, t]

        # Total power output [MW]
        m.P_TOTAL = Expression(m.G, m.T, rule=total_power_output_rule)

        def generator_energy_output_rule(m, g, t):
            """
            Total generator energy output [MWh]

            No information regarding generator dispatch level in period prior to
            t=0. Assume that all generators are at their minimum dispatch level.
            """

            # TODO: Must take into account power output for interval preceding t=0
            if t != m.T.first():
                return (m.P_TOTAL[g, t - 1] + m.P_TOTAL[g, t]) / 2
            else:
                return (m.P_MIN[g] + m.P_TOTAL[g, t]) / 2

        # Energy output for a given generator
        m.e = Expression(m.G, m.T, rule=generator_energy_output_rule)

        def thermal_operating_costs_rule(m):
            """Cost to operate existing and candidate thermal units"""

            return (m.SCENARIO_DURATION * sum((m.C_MC[g] + (m.E[g] - m.baseline) * m.permit_price) * m.e[g, t]
                                              + (m.C_SU[g] * m.v[g, t]) + (m.C_SD[g] * m.w[g, t])
                                              for g in m.G_E_THERM.union(m.G_C_THERM) for t in m.T))

        # Existing and candidate thermal unit operating costs
        m.C_OP_THERM = Expression(rule=thermal_operating_costs_rule)

        def hydro_operating_costs_rule(m):
            """Cost to operate existing hydro generators"""

            return m.SCENARIO_DURATION * sum(m.C_MC[g] * m.e[g, t] for g in m.G_E_HYDRO for t in m.T)

        # Existing hydro unit operating costs (no candidate hydro generators)
        m.C_OP_HYDRO = Expression(rule=hydro_operating_costs_rule)

        def solar_operating_costs_rule(m):
            """Cost to operate existing and candidate solar units"""

            return ((sum(m.C_MC[g] * m.e[g, t] for g in m.G_E_SOLAR for t in m.T)
                     + sum((m.C_MC[g] - m.baseline * m.permit_price) * m.e[g, t] for g in m.G_C_SOLAR for t in m.T)
                     ) * m.SCENARIO_DURATION)

        # Existing and candidate solar unit operating costs (only candidate solar eligible for credits)
        m.C_OP_SOLAR = Expression(rule=solar_operating_costs_rule)

        def wind_operating_costs_rule(m):
            """Cost to operate existing and candidate wind generators"""

            return ((sum(m.C_MC[g] * m.e[g, t] for g in m.G_E_WIND for t in m.T)
                     + sum((m.C_MC[g] - m.baseline * m.permit_price) * m.e[g, t] for g in m.G_C_WIND for t in m.T)
                     ) * m.SCENARIO_DURATION)

        # Existing and candidate solar unit operating costs (only candidate solar eligible for credits)
        m.C_OP_WIND = Expression(rule=wind_operating_costs_rule)

        def storage_operating_costs_rule(m):
            """Cost to operate candidate storage units"""

            return (sum((m.C_MC[g] - m.baseline * m.permit_price) * m.e[g, t] for g in m.G_C_STORAGE for t in m.T)
                    * m.SCENARIO_DURATION)

        # Candidate storage unit operating costs
        m.C_OP_STORAGE = Expression(rule=storage_operating_costs_rule)

        def total_operating_cost(m):
            """Total operating cost"""

            return m.C_OP_THERM + m.C_OP_HYDRO + m.C_OP_SOLAR + m.C_OP_WIND + m.C_OP_STORAGE

        # Total operating cost
        m.C_OP_TOTAL = Expression(rule=total_operating_cost)

        return m

    def define_constraints(self, m):
        """Define model constraints"""

        def upward_power_reserve_rule(m, r, t):
            """Upward reserve constraint"""

            # NEM region-zone map
            region_zone_map = self._get_nem_region_zone_map()

            # Mapping describing the zone to which each generator is assigned
            generator_zone_map = self._get_generator_zone_map()

            # Existing and candidate thermal gens + candidate storage units
            gens = m.G_E_THERM.union(m.G_C_THERM).union(m.G_C_STORAGE)

            #

            return sum(m.r_up[g, t] for g in gens if generator_zone_map.loc[g] in region_zone_map.loc[r]) >= m.D_UP[r]

        # Upward power reserve rule for each NEM zone
        m.UPWARD_POWER_RESERVE = Constraint(m.R, m.T, rule=upward_power_reserve_rule)

        def operating_state_logic_rule(m, g, t):
            """
            Determine the operating state of the generator (startup, shutdown
            running, off)
            """

            if t == m.T.first():
                # Must use u0 if first period (otherwise index out of range)
                return m.u[g, t] - m.u0[g] == m.v[g, t] - m.w[g, t]

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

            if t < m.T.last():
                return (m.p[g, t] + m.r_up[g, t]
                        <= ((m.P_MAX[g] - m.P_MIN[g]) * m.u[g, t])
                        - ((m.P_MAX[g] - m.SD_RAMP[g]) * m.w[g, t + 1])
                        + ((m.SU_RAMP[g] - m.P_MIN[g]) * m.v[g, t + 1]))
            else:
                return Constraint.Skip

        # Power production
        m.POWER_PRODUCTION = Constraint(m.G_E_THERM.union(m.G_C_THERM), m.T, rule=power_production_rule)

        def ramp_rate_up_rule(m, g, t):
            """Ramp-rate up constraint"""

            if t > m.T.first():
                return (m.p[g, t] + m.r_up[g, t]) - m.p[g, t - 1] <= m.RU[g]
            else:
                return Constraint.Skip

        # Ramp-rate up limit
        m.RAMP_RATE_UP = Constraint(m.G_E_THERM.union(m.G_C_THERM), m.T, rule=ramp_rate_up_rule)

        def ramp_rate_down_rule(m, g, t):
            """Ramp-rate down constraint"""

            if t > m.T.first():
                return - m.p[g, t] + m.p[g, t - 1] <= m.RD[g]
            else:
                return Constraint.Skip

        # Ramp-rate up limit
        m.RAMP_RATE_DOWN = Constraint(m.G_E_THERM.union(m.G_C_THERM), m.T, rule=ramp_rate_down_rule)

        def existing_wind_output_min_rule(m, g, t):
            """Constrain minimum output for existing wind generators"""

            return m.P_TOTAL[g, t] >= 0

        # Minimum wind output for existing generators
        m.EXISTING_WIND_MIN = Constraint(m.G_E_WIND, m.T, rule=existing_wind_output_min_rule)

        # TODO: Need to map existing and candidate units to wind bubbles
        # def existing_wind_output_max_rule(m, g, t):
        #     """Constrain maximum output for existing wind generators"""
        #
        #     return m.P_TOTAL[g, t] <= m.

        return m

    def _nem_zone_wind_bubble_map(self):
        """Assign wind bubble to existing wind unit"""

        wind_bubbles = self._get_wind_bubble_map()

        wind_bubbles_unique = wind_bubbles.loc[self._get_wind_bubbles()]

        bubbles = wind_bubbles_unique.reset_index().set_index('ZONE')['BUBBLE_ID']

        return bubbles

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

        # Construct constraints
        m = self.define_constraints(m)

        return m

    def update_parameters(self, m, year):
        """Update model parameters for a given year"""
        pass


if __name__ == '__main__':
    # Directory containing files from which dataset is derived
    raw_data_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'data')

    # Directory containing core data files
    data_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '1_collect_data', 'output')

    # Directory containing input traces
    input_traces_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '2_input_traces', 'output')

    # Instantiate UC model
    uc = UnitCommitmentModel(raw_data_directory, data_directory, input_traces_directory)

    # Construct model
    model = uc.construct_model()

    # Wind bubbles
    bubbles = uc._nem_zone_wind_bubble_map()
