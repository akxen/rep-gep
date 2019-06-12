import os

import pandas as pd
from pyomo.environ import *


class BaseComponents:
    def __init__(self, raw_data_dir, data_dir, input_traces_dir):

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

        # Map between DUIDs and wind bubble IDs for existing generators
        self.existing_wind_bubble_map = pd.read_csv(os.path.join(raw_data_dir, 'maps', 'existing_wind_bubble_map.csv'),
                                                    index_col='DUID')

        # Network incidence matrix
        self.incidence_matrix = self._get_network_incidence_matrix()

        # Flow over interconnectors
        self.powerflow_limits = self._get_link_powerflow_limits()

        # Solver details
        solver = 'gurobi'
        solver_io = 'lp'
        self.keepfiles = False
        self.solver_options = {}  # 'MIPGap': 0.0005
        self.opt = SolverFactory(solver, solver_io=solver_io)

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
    def _get_link_powerflow_limits():
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

        # Interconnectors for which flow limits are defined
        m.L_I = Set(initialize=list(self.powerflow_limits.keys()))

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

        # Available technologies
        m.G_C_SOLAR_TECHNOLOGIES = Set(initialize=list(set(i.split('-')[-1] for i in m.G_C_SOLAR)))

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

        # All candidate generators
        m.G_C = m.G_C_THERM.union(m.G_C_WIND).union(m.G_C_SOLAR)

        # All generators
        m.G = m.G_E.union(m.G_C)

        # Operating scenario hour
        m.T = RangeSet(0, 23, ordered=True)

        # All years in model horizon
        m.I = RangeSet(2016, 2050)

        # Build limit technology types
        m.BUILD_LIMIT_TECHNOLOGIES = Set(initialize=self.candidate_unit_build_limits.index)

        return m
