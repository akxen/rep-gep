"""Pre-process model data"""

import os
from dataclasses import dataclass

import pandas as pd


@dataclass
class ModelData:
    # def __init__(self, raw_data_dir, data_dir, input_traces_dir):
    # Directory containing raw data files
    raw_data_dir: str

    # Directory containing core data files
    data_dir: str

    # Directory containing input solar, wind, hydro, and demand traces
    input_traces_dir: str

    def __post_init__(self):
        # Input traces
        self.input_traces = self._load_input_traces()

        # Candidate units
        self.candidate_units = self._load_candidate_units()

        # Existing units
        self.existing_units = self._load_existing_units()

        # Battery build costs
        self.battery_build_costs = self._load_battery_build_costs()

        # Battery properties
        self.battery_properties = self._load_battery_properties()

        # Battery build costs
        self.candidate_unit_build_limits = self._load_candidate_unit_build_limits()

        # Region minimum reserve levels
        self.minimum_reserve_levels = self._load_minimum_reserve_levels()

        # Map between DUIDs and wind bubble IDs for existing generators
        self.existing_wind_bubble_map = self._load_existing_wind_bubble_map()

        # # Network incidence matrix
        # self.incidence_matrix = self._get_network_incidence_matrix()
        #
        # # Flow over interconnectors
        # self.powerflow_limits = self._get_link_powerflow_limits().to_dict()
        #
        # # Maps
        # self.nem_region_zone_map = self._get_nem_region_zone_map()
        # self.nem_region_zone_map_dict = self.nem_region_zone_map.to_dict().to_dict()
        #
        # self.generator_zone_map = self._get_generator_zone_map().to_dict()
        # self.generator_zone_map_dict = self.generator_zone_map.to_dict()

    @staticmethod
    def _col_mapper(text):
        """Try and convert text in columns to int if possible"""
        try:
            output = int(text)
        except ValueError:
            output = text
        return output

    def _load_input_traces(self) -> dict:
        """Load input traces"""

        return pd.read_pickle(os.path.join(self.input_traces_dir, 'centroids.pickle')).to_dict()

    def _load_existing_units(self) -> dict:
        """Load existing unit information"""

        # Load CSV
        df = pd.read_csv(os.path.join(self.data_dir, 'existing_units.csv'), header=[0, 1], index_col=0)

        # Rename columns
        df = df.rename(self._col_mapper, axis='columns', level=1)

        # Dictionary output
        output = df.to_dict()

        return output

    def _load_candidate_units(self) -> dict:
        """Load candidate unit information"""

        # Load CSV
        df = pd.read_csv(os.path.join(self.data_dir, 'candidate_units.csv'), header=[0, 1], index_col=0)

        # Rename columns
        df = df.rename(self._col_mapper, axis='columns', level=1)

        # Dictionary output
        output = df.to_dict()

        return output

    def _load_battery_build_costs(self) -> dict:
        """Load battery build costs"""

        # Load CSV
        df = pd.read_csv(os.path.join(self.data_dir, 'battery_build_costs.csv'), header=0, index_col=0)

        # Rename columns
        df = df.rename(self._col_mapper, axis='columns')

        # Dictionary output
        output = df.to_dict()

        return output

    def _load_battery_properties(self) -> dict:
        """Load battery properties"""

        # Load battery properties from spreadsheet
        df = pd.read_csv(os.path.join(self.data_dir, 'battery_properties.csv'), header=0, index_col=0)

        # Add NEM zone as column
        df['NEM_ZONE'] = df.apply(lambda x: x.name.split('-')[0], axis=1)

        # Dictionary output
        output = df.to_dict()

        return output

    def _load_candidate_unit_build_limits(self) -> dict:
        """Load candidate unit build limits"""

        # Candidate unit build limits
        df = pd.read_csv(os.path.join(self.data_dir, 'candidate_unit_build_limits.csv'), header=0, index_col=0)

        # Dictionary output
        output = df.to_dict()

        return output

    def _load_minimum_reserve_levels(self) -> dict:
        """Load minimum reserve levels for each NEM zone"""

        # Minimum reserve levels for each NEM zone
        df = pd.read_csv(os.path.join(self.data_dir, 'minimum_reserve_levels.csv'), header=0, index_col=0)

        # Dictionary output
        output = df.to_dict()

        return output

    def _load_existing_wind_bubble_map(self) -> dict:
        """Load mapping between existing wind units and NEM zones"""

        # Map between existing units and their corresponding NEM zones
        df = pd.read_csv(os.path.join(self.raw_data_dir, 'maps', 'existing_wind_bubble_map.csv'), index_col='DUID')

        # Dictionary output
        output = df.to_dict()

        return output

    def _get_nem_zones(self) -> tuple:
        """Get list of unique NEM zones"""

        # Extract nem zones from existing generators dataset
        zones = tuple(self.existing_units.loc[:, ('PARAMETERS', 'NEM_ZONE')].unique())

        # There should be 16 zones
        assert len(zones) == 16, 'Unexpected number of NEM zones'

        return zones

    def _get_nem_regions(self) -> tuple:
        """Get list of unique NEM regions"""

        # Extract nem regions from existing generators dataset
        regions = tuple((set(self.existing_units[('PARAMETERS', 'NEM_REGION')].values())))

        # There should be 5 NEM regions
        assert len(regions) == 5, 'Unexpected number of NEM regions'

        return regions

    def _get_nem_region_zone_map(self) -> tuple:
        """
        Construct mapping between NEM regions and the zones belonging to
        those regions
        """

        # Map between NEM regions and zones
        region_zone_map = (self.existing_units[[('PARAMETERS', 'NEM_REGION'), ('PARAMETERS', 'NEM_ZONE')]]
                           .droplevel(0, axis=1).drop_duplicates(subset=['NEM_REGION', 'NEM_ZONE'])
                           .groupby('NEM_REGION')['NEM_ZONE'].apply(lambda x: list(x)))

        return region_zone_map

    # def _get_wind_bubble_map(self):
    #     """Mapping between wind bubbles and NEM regions and zones"""
    #
    #     # Load data
    #     df = pd.read_csv(os.path.join(self.raw_data_dir, 'maps', 'wind_bubbles.csv'), index_col=0)
    #
    #     return df
    #
    #
    # def _get_wind_bubbles(self):
    #     """
    #     Get unique wind bubbles
    #
    #     Note: Not all wind bubbles in dataset are used. Some zones may have more
    #     than one wind bubble. Model arbitrarily selects bubbles so there is at
    #     most one bubble per zone.
    #     """
    #
    #     # Extract wind bubbles from input data traces file
    #     df = self.input_traces.copy()
    #
    #     # Unique wind bubbles
    #     bubbles = list(set([i[0] for i in df.loc[:, 'WIND'].columns.values]))
    #
    #     return bubbles
    #
    #
    # def _get_candidate_thermal_unit_ids(self):
    #     """Get all IDs for candidate thermal units"""
    #
    #     # Get candidate thermal units
    #     mask_candidate_thermal = self.candidate_units[('PARAMETERS', 'TECHNOLOGY_PRIMARY')].isin(['GAS', 'COAL'])
    #
    #     # IDs for candidate thermal units
    #     candidate_thermal_ids = self.candidate_units[mask_candidate_thermal].index
    #
    #     return candidate_thermal_ids
    #
    #
    # def _get_candidate_solar_unit_ids(self):
    #     """Get IDs for candidate solar units"""
    #
    #     # Filter candidate solar units
    #     mask_candidate_solar = self.candidate_units[('PARAMETERS', 'TECHNOLOGY_PRIMARY')].isin(['SOLAR'])
    #
    #     # IDs for existing thermal units
    #     candidate_solar_ids = self.candidate_units[mask_candidate_solar].index
    #
    #     return candidate_solar_ids
    #
    #
    # def _get_candidate_wind_unit_ids(self):
    #     """Get IDs for candidate wind units"""
    #
    #     # Filter candidate wind units
    #     mask_candidate_wind = self.candidate_units[('PARAMETERS', 'TECHNOLOGY_PRIMARY')].isin(['WIND'])
    #
    #     # IDs for existing thermal units
    #     candidate_wind_ids = self.candidate_units[mask_candidate_wind].index
    #
    #     return candidate_wind_ids
    #
    #
    # def _get_existing_thermal_unit_ids(self):
    #     """Get IDs for existing thermal units"""
    #
    #     # Filter existing thermal units
    #     mask_existing_thermal = self.existing_units[('PARAMETERS', 'TECHNOLOGY_CAT_PRIMARY')].isin(
    #         ['GAS', 'COAL', 'LIQUID'])
    #
    #     # IDs for existing thermal units
    #     existing_thermal_ids = self.existing_units[mask_existing_thermal].index
    #
    #     return existing_thermal_ids
    #
    #
    # def _get_existing_solar_unit_ids(self):
    #     """Get IDs for existing solar units"""
    #
    #     # Filter existing solar units
    #     mask_existing_solar = self.existing_units[('PARAMETERS', 'TECHNOLOGY_CAT_PRIMARY')].isin(['SOLAR'])
    #
    #     # IDs for existing solar units
    #     existing_solar_ids = self.existing_units[mask_existing_solar].index
    #
    #     return existing_solar_ids
    #
    #
    # def _get_existing_wind_unit_ids(self):
    #     """Get IDs for existing wind units"""
    #
    #     # Filter existing wind units
    #     mask_existing_wind = self.existing_units[('PARAMETERS', 'TECHNOLOGY_CAT_PRIMARY')].isin(['WIND'])
    #
    #     # IDs for existing wind units
    #     existing_wind_ids = self.existing_units[mask_existing_wind].index
    #
    #     return existing_wind_ids
    #
    #
    # def _get_existing_hydro_unit_ids(self):
    #     """Get IDs for existing hydro units"""
    #
    #     # Filter existing hydro units
    #     mask_existing_hydro = self.existing_units[('PARAMETERS', 'TECHNOLOGY_CAT_PRIMARY')].isin(['HYDRO'])
    #
    #     # IDs for existing hydro units
    #     existing_hydro_ids = self.existing_units[mask_existing_hydro].index
    #
    #     return existing_hydro_ids
    #
    #
    # def _get_candidate_storage_units(self):
    #     """Get IDs for candidate storage units"""
    #
    #     # IDs for candidate storage units
    #     candidate_storage_ids = self.battery_properties.index
    #
    #     return candidate_storage_ids
    #
    #
    # def _get_slow_start_thermal_generator_ids(self):
    #     """
    #     Get IDs for existing and candidate slow start unit
    #
    #     A generator is classified as 'slow' if it cannot reach its
    #     minimum dispatchable power output in one interval (e.g. 1 hour).
    #
    #     Note: A generator's classification of 'quick' or 'slow' depends on its
    #     minimum dispatchable output level and ramp-rate. For candidate units
    #     the minimum dispatchable output level is a function of the maximum
    #     output level, and so is variable. As this level is not known ex ante,
    #     all candidate thermal generators are assumed to operate the same way
    #     as quick start units (i.e. they can reach their minimum dispatchable
    #     output level in 1 trading interval (hour)).
    #     """
    #
    #     # True if number of hours to ramp to min generator output > 1
    #     mask_slow_start = (self.existing_units[('PARAMETERS', 'MIN_GEN')]
    #                        .div(self.existing_units[('PARAMETERS', 'RR_STARTUP')])
    #                        .gt(1))
    #
    #     # Only consider coal and gas units
    #     mask_technology = self.existing_units[('PARAMETERS', 'TECHNOLOGY_CAT_PRIMARY')].isin(['COAL', 'GAS', 'LIQUID'])
    #
    #     # Get IDs for slow start generators
    #     gen_ids = self.existing_units.loc[mask_slow_start & mask_technology, :].index
    #
    #     return gen_ids
    #
    #
    # def _get_quick_start_thermal_generator_ids(self):
    #     """
    #     Get IDs for existing and candidate slow start unit
    #
    #     Note: A generator is classified as 'quick' if it can reach its
    #     minimum dispatchable power output in one interval (e.g. 1 hour).
    #     """
    #
    #     # Slow start unit IDs - previously identified
    #     slow_gen_ids = self._get_slow_start_thermal_generator_ids()
    #
    #     # Filter for slow generator IDs (existing units)
    #     mask_slow_gen_ids = self.existing_units.index.isin(slow_gen_ids)
    #
    #     # Only consider coal and gas units
    #     mask_existing_technology = (self.existing_units[('PARAMETERS', 'TECHNOLOGY_CAT_PRIMARY')]
    #                                 .isin(['COAL', 'GAS', 'LIQUID']))
    #
    #     # Get IDs for quick start generators
    #     existing_quick_gen_ids = self.existing_units.loc[~mask_slow_gen_ids & mask_existing_technology, :].index
    #
    #     # Get IDs for quick start candidate generators (all candidate coal / gas generators)
    #     mask_candidate_technology = (self.candidate_units[('PARAMETERS', 'TECHNOLOGY_PRIMARY')].isin(['COAL', 'GAS']))
    #
    #     # Quick start candidate generator IDs
    #     candidate_quick_gen_ids = self.candidate_units.loc[mask_candidate_technology].index
    #
    #     # All quick start generator IDs (existing and candidate)
    #     gen_ids = existing_quick_gen_ids.union(candidate_quick_gen_ids)
    #
    #     return gen_ids
    #
    #
    # def _get_network_incidence_matrix(self):
    #     """Construct network incidence matrix"""
    #
    #     # All NEM zones:
    #     zones = self._get_nem_zones()
    #
    #     # Links connecting different zones. First zone is 'from' zone second is 'to' zone
    #     links = ['NQ-CQ', 'CQ-SEQ', 'CQ-SWQ', 'SWQ-SEQ', 'SEQ-NNS',
    #              'SWQ-NNS', 'NNS-NCEN', 'NCEN-CAN', 'CAN-SWNSW',
    #              'CAN-NVIC', 'SWNSW-NVIC', 'LV-MEL', 'NVIC-MEL',
    #              'TAS-LV', 'MEL-CVIC', 'SWNSW-CVIC', 'CVIC-NSA',
    #              'MEL-SESA', 'SESA-ADE', 'NSA-ADE']
    #
    #     # Initialise empty matrix with NEM zones as row and column labels
    #     incidence_matrix = pd.DataFrame(index=links, columns=zones, data=0)
    #
    #     # Assign values to 'from' and 'to' zones. +1 is a 'from' zone, -1 is a 'to' zone
    #     for link in links:
    #         # Get from and to zones
    #         from_zone, to_zone = link.split('-')
    #
    #         # Set from zone element to 1
    #         incidence_matrix.loc[link, from_zone] = 1
    #
    #         # Set to zone element to -1
    #         incidence_matrix.loc[link, to_zone] = -1
    #
    #     return incidence_matrix
    #
    #
    # def _get_network_links(self):
    #     """Links connecting adjacent NEM zones"""
    #
    #     return self._get_network_incidence_matrix().index
    #
    #
    # def _get_generator_zone_map(self):
    #     """Get mapping between existing and candidate generators and NEM zones"""
    #
    #     # Existing units
    #     existing = self.existing_units[('PARAMETERS', 'NEM_ZONE')]
    #
    #     # Candidate units
    #     candidate = self.candidate_units[('PARAMETERS', 'ZONE')]
    #
    #     # Candidate storage units
    #     storage = self.battery_properties['NEM_ZONE']
    #
    #     # Concatenate series objects
    #     generator_zone_map = pd.concat([existing, candidate, storage])
    #
    #     return generator_zone_map
    #
    #
    # @staticmethod
    # def _get_link_powerflow_limits():
    #     """Max forward and reverse power flow over links between zones"""
    #
    #     # Limits for interconnectors composed of single branches
    #     interconnector_limits = {'SEQ-NNS': {'forward': 210, 'reverse': 107},  # Terranora
    #                              'SWQ-NNS': {'forward': 1078, 'reverse': 600},  # QNI
    #                              'TAS-LV': {'forward': 594, 'reverse': 478},  # Basslink
    #                              'MEL-SESA': {'forward': 600, 'reverse': 500},  # Heywood
    #                              'CVIC-NSA': {'forward': 220, 'reverse': 200},  # Murraylink
    #                              }
    #
    #     return interconnector_limits


if __name__ == '__main__':
    # Directory containing files from which dataset is derived
    raw_data_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'data')

    # Directory containing core data files
    data_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '1_collect_data', 'output')

    # Directory containing input traces
    input_traces_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '2_input_traces', 'output')

    model = ModelData(raw_data_directory, data_directory, input_traces_directory)
