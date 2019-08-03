import os
import time
from math import ceil
from collections import OrderedDict

import pandas as pd
from pyomo.environ import *
from pyomo.util.infeasible import log_infeasible_constraints


class UnitCommitmentModel:
    def __init__(self, raw_data_dir, data_dir, input_traces_dir):

        # Directory containing raw data files
        self.raw_data_dir = raw_data_dir

        # Directory containing core data files
        self.data_dir = data_dir

        # Directory containing input solar, wind, hydro, and demand traces
        self.input_traces_dir = input_traces_dir

        # Input traces
        self.input_traces = pd.read_pickle(os.path.join(input_traces_dir, 'centroids.pickle'))
        self.input_traces_dict = self.input_traces.to_dict()

        # Candidate units
        self.candidate_units = self._load_candidate_units()
        self.candidate_units_dict = self.candidate_units.to_dict()

        # Existing units
        self.existing_units = self._load_existing_units()
        self.existing_units_dict = self.existing_units.to_dict()

        # Battery build costs
        self.battery_build_costs = self._load_battery_build_costs()

        # Battery properties
        self.battery_properties = self._load_battery_properties()
        self.battery_properties_dict = self.battery_properties.to_dict()

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
        self.existing_wind_bubble_map_dict = self.existing_wind_bubble_map.to_dict()

        # Network incidence matrix
        self.incidence_matrix = self._get_network_incidence_matrix()

        # Flow over interconnectors
        self.powerflow_limits = self._get_link_powerflow_limits()

        # Solver details
        solver = 'gurobi'
        solver_io = 'lp'
        self.keepfiles = False
        self.solver_options = {'Method': 1}  # 'MIPGap': 0.0005
        self.opt = SolverFactory(solver, solver_io=solver_io)

        # Maps
        self.nem_region_zone_map = self._get_nem_region_zone_map()
        self.nem_region_zone_map_dict = self.nem_region_zone_map.to_dict()

        self.generator_zone_map = self._get_generator_zone_map()
        self.generator_zone_map_dict = self.generator_zone_map.to_dict()

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

        # All years in model horizon
        m.I = RangeSet(2016, 2017)

        # Operating scenarios for each year
        m.O = RangeSet(0, 9)

        # Operating scenario hour
        m.T = RangeSet(0, 23, ordered=True)

        # Build limit technology types
        m.BUILD_LIMIT_TECHNOLOGIES = Set(initialize=self.candidate_unit_build_limits.index)

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

        def marginal_cost_rule(_s, g, i):
            """Marginal costs for existing and candidate generators

            Note: Marginal costs for existing and candidate thermal plant depend on fuel costs, which are time
            varying. Therefore marginal costs for thermal plant must be define for each year in model horizon.
            """

            if g in m.G_E_THERM:

                # Last year in the dataset for which fuel cost information exists
                max_year = self.existing_units.loc[g, 'FUEL_COST'].index.max()

                # If year in model horizon exceeds max year for which data are available use values for last
                # available year
                if i > max_year:
                    # Use final year in dataset to max year
                    i = max_year

                marginal_cost = float(self.existing_units.loc[g, ('FUEL_COST', i)]
                                      * self.existing_units.loc[g, ('PARAMETERS', 'HEAT_RATE')])

            elif g in m.G_C_THERM:
                # Last year in the dataset for which fuel cost information exists
                max_year = self.candidate_units.loc[g, 'FUEL_COST'].index.max()

                # If year in model horizon exceeds max year for which data are available use values for last
                # available year
                if i > max_year:
                    # Use final year in dataset to max year
                    i = max_year

                marginal_cost = float(self.candidate_units.loc[g, ('FUEL_COST', i)])

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

        # Marginal costs for all generators and time periods
        m.C_MC = Param(m.G.union(m.G_C_STORAGE), m.I, rule=marginal_cost_rule)

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
        m.emissions_target_exceeded = Var(within=NonNegativeReals)

        # Amount by which scheme revenue falls short of target
        m.revenue_shortfall = Var(within=NonNegativeReals)

        return m

    @staticmethod
    def define_expressions(m):
        """Define model expressions"""

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
        m.X_C = Expression(m.G_C, m.I, rule=capacity_sizing_rule)

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

        def min_power_output_rule(_m, g, i):
            """
            Minimum generator output in MW

            Note: Min power output can be a function of max capacity which is only known ex-post for candidate
            generators.
            """

            return m.P_MIN_PROP[g] * m.P_MAX[g, i]

        # Expression for minimum power output
        m.P_MIN = Expression(m.G, m.I, rule=min_power_output_rule)

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

        # Penalty imposed for violating emissions constraint
        m.C_EMISSIONS_VIOLATION = Expression(expr=m.emissions_target_exceeded * m.EMISSIONS_EXCEEDED_PENALTY)

        # Penalty imposed for violating revenue constraint
        m.C_REVENUE_VIOLATION = Expression(expr=m.revenue_shortfall * m.REVENUE_SHORTFALL_PENALTY)

        return m

    @staticmethod
    def define_constraints(m):
        """Define constraints - common for all blocks"""

        def fixed_baseline_rule(_m, i):
            """Fix emissions intensity baseline to given value in sub-problems"""

            return m.baseline[i] == m.FIXED_BASELINE[i]

        # Fix emissions intensity baseline for each year in model horizon
        m.FIXED_BASELINE_CONS = Constraint(m.I, rule=fixed_baseline_rule)

        def fixed_permit_price_rule(_m, i):
            """Fixed permit price rule"""

            return m.permit_price[i] == m.FIXED_PERMIT_PRICE[i]

        # Fix permit price to given value in sub-problems
        m.FIXED_PERMIT_PRICE_CONS = Constraint(m.I, rule=fixed_permit_price_rule)

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

        return m

    def define_blocks(self, m):
        """Define blocks for each operating scenario"""

        def operating_scenario_block_rule(s, i, o):
            """Operating scenario block"""

            def define_block_parameters(s):
                """Define all parameters within a given block"""

                # Fixed shutdown indicator binary variable value
                start = time.time()
                s.FIXED_W = Param(m.G_E_THERM.union(m.G_C_THERM), m.T, initialize=0, within=Binary, mutable=True)
                print(f'Constructed FIXED_W in: {time.time() - start}s')

                # Fixed startup indicator binary variable value
                start = time.time()
                s.FIXED_V = Param(m.G_E_THERM.union(m.G_C_THERM), m.T, initialize=0, within=Binary, mutable=True)
                print(f'Constructed FIXED_V in: {time.time() - start}s')

                # Fixed on-state binary variable value
                start = time.time()
                s.FIXED_U = Param(m.G_E_THERM.union(m.G_C_THERM), m.T, initialize=0, within=Binary, mutable=True)
                print(f'Constructed FIXED_U in: {time.time() - start}s')

                # Indicates if unit is available (assume available for all periods for now)
                start = time.time()
                s.AVAILABILITY_INDICATOR = Param(m.G, initialize=1, within=Binary, mutable=True)
                print(f'Constructed AVAILABILITY_INDICATOR in: {time.time() - start}s')

                # Power output in interval prior to model start (assume = 0 for now)
                start = time.time()
                s.P0 = Param(m.G, mutable=True, within=NonNegativeReals, initialize=0)
                print(f'Constructed P0 in: {time.time() - start}s')

                # Energy in battering in interval prior to model start (assume battery initially completely discharged)
                start = time.time()
                s.Y0 = Param(m.G_C_STORAGE, initialize=0)
                print(f'Constructed Y0 in: {time.time() - start}s')

                def wind_capacity_factor_rule(_s, b, t):
                    """Wind capacity factors for each operating scenario"""

                    # Capacity factor for given wind plant and operating scenario
                    # capacity_factor = float(self.input_traces.loc[(i, o), ('WIND', b, t)])
                    capacity_factor = float(self.input_traces_dict[('WIND', b, t)][(i, o)])

                    # Handle small values - prevent numerical instability when solving
                    if capacity_factor < 0.01:
                        return float(0)
                    else:
                        return capacity_factor

                # Wind capacity factors
                start = time.time()
                s.Q_WIND = Param(m.B, m.T, rule=wind_capacity_factor_rule)
                print(f'Constructed Q_WIND in: {time.time() - start}s')

                def solar_capacity_factor_rule(_s, z, g, t):
                    """Solar capacity factors in each NEM zone"""

                    # Replace FFP with FFP2 - so technology type names are consistent
                    if g == 'FFP':
                        g = 'FFP2'

                    # Column name is a composite of the zone and technology type (g)
                    col = f'{z}|{g}'

                    # capacity_factor = float(self.input_traces.loc[(i, o), ('SOLAR', col, t)])
                    capacity_factor = float(self.input_traces_dict[('SOLAR', col, t)][(i, o)])

                    if capacity_factor < 0.01:
                        return float(0)
                    else:
                        return capacity_factor

                # Solar capacity factors
                start = time.time()
                s.Q_SOLAR = Param(m.Z, m.G_C_SOLAR_TECHNOLOGIES, m.T, rule=solar_capacity_factor_rule)
                print(f'Constructed Q_SOLAR in: {time.time() - start}s')

                def demand_rule(_s, z, t):
                    """NEM demand in each zone"""

                    # Demand in given NEM zone
                    # demand = float(self.input_traces.loc[(i, o), ('DEMAND', z, t)])
                    demand = float(self.input_traces_dict[('DEMAND', z, t)][(i, o)])

                    return demand

                # Demand in each NEM zone
                start = time.time()
                s.DEMAND = Param(m.Z, m.T, rule=demand_rule)
                print(f'Constructed DEMAND in: {time.time() - start}s')

                def scenario_duration_rule(_s):
                    """Normalised duration for each operation scenario"""

                    return float(self.input_traces.loc[(i, o), ('K_MEANS', 'METRIC', 'NORMALISED_DURATION')])

                # Normalised duration for each operating scenario
                start = time.time()
                s.NORMALISED_DURATION = Param(rule=scenario_duration_rule)
                print(f'Constructed NORMALISED_DURATION in: {time.time() - start}s')

                def historic_hydro_output_rule(_s, g, t):
                    """Historic output for existing hydro generators"""
                    # output = float(self.input_traces.loc[(i, o), ('HYDRO', g, t)])
                    output = float(self.input_traces_dict[('HYDRO', g, t)][(i, o)])

                    # Remove small values - prevent numerical instability when solving
                    if output < 1:
                        return float(0)
                    else:
                        return output

                # Assumed output for hydro generators (based on historic values)
                start = time.time()
                s.P_HYDRO_HISTORIC = Param(m.G_E_HYDRO, m.T, rule=historic_hydro_output_rule)
                print(f'Constructed P_HYDRO_HISTORIC in: {time.time() - start}s')

                return s

            def define_block_variables(s):
                """Define variables for each block"""

                # Startup state variable
                s.v = Var(m.G_E_THERM.union(m.G_C_THERM), m.T, within=Binary, initialize=0)

                # Shutdown state variable
                s.w = Var(m.G_E_THERM.union(m.G_C_THERM), m.T, within=Binary, initialize=0)

                # On-state variable
                s.u = Var(m.G_E_THERM.union(m.G_C_THERM), m.T, within=Binary, initialize=1)

                # Power output above minimum dispatchable level of output [MW]
                s.p = Var(m.G, m.T, within=NonNegativeReals, initialize=0)

                # Upward reserve allocation [MW]
                s.r_up = Var(m.G_E_THERM.union(m.G_C_THERM).union(m.G_C_STORAGE), m.T, within=NonNegativeReals,
                             initialize=0)

                # Storage unit charging (power in) [MW]
                s.p_in = Var(m.G_C_STORAGE, m.T, within=NonNegativeReals, initialize=0)

                # Storage unit discharging (power out) [MW]
                s.p_out = Var(m.G_C_STORAGE, m.T, within=NonNegativeReals, initialize=0)

                # Energy in storage unit [MWh]
                s.y = Var(m.G_C_STORAGE, m.T, within=NonNegativeReals, initialize=0)

                # Powerflow between NEM zones [MW]
                s.p_flow = Var(m.L, m.T, initialize=0)

                # Lost load - up [MW]
                s.p_lost_up = Var(m.Z, m.T, within=NonNegativeReals, initialize=0)

                # Lost load - down [MW]
                s.p_lost_down = Var(m.Z, m.T, within=NonNegativeReals, initialize=0)

                return s

            def define_block_expressions(s):
                """Define expressions used when constructing each operating scenario"""

                def total_power_output_rule(_s, g, t):
                    """Total power output [MW]"""

                    if g in m.G_THERM_QUICK:
                        if t < m.T.last():
                            return ((m.P_MIN[g, i] * (s.u[g, t] + s.v[g, t + 1])) + s.p[g, t]) * \
                                   s.AVAILABILITY_INDICATOR[
                                       g]
                        else:
                            return ((m.P_MIN[g, i] * s.u[g, t]) + s.p[g, t]) * s.AVAILABILITY_INDICATOR[g]

                    # Define startup and shutdown trajectories for 'slow' generators
                    elif g in m.G_THERM_SLOW:
                        # Startup duration
                        SU_D = ceil(m.P_MIN[g, i].expr() / m.RR_SU[g])

                        # Startup power output trajectory increment
                        ramp_up_increment = m.P_MIN[g, i].expr() / SU_D

                        # Startup power output trajectory
                        P_SU = OrderedDict({k + 1: ramp_up_increment * k for k in range(0, SU_D + 1)})

                        # Shutdown duration
                        SD_D = ceil(m.P_MIN[g, i].expr() / m.RR_SD[g])

                        # Shutdown power output trajectory increment
                        ramp_down_increment = m.P_MIN[g, i].expr() / SD_D

                        # Shutdown power output trajectory
                        P_SD = OrderedDict(
                            {k + 1: m.P_MIN[g, i].expr() - (ramp_down_increment * k) for k in range(0, SD_D + 1)})

                        if t < m.T.last():
                            return (((m.P_MIN[g, i] * (s.u[g, t] + s.v[g, t + 1])) + s.p[g, t]
                                     + sum(P_SU[k] * s.v[g, t - k + SU_D + 2] if t - k + SU_D + 2 in m.T else 0 for k in
                                           range(1, SU_D + 1))
                                     + sum(P_SD[k] * s.w[g, t - k + 2] if t - k + 2 in m.T else 0 for k in
                                           range(2, SD_D + 2))
                                     ) * s.AVAILABILITY_INDICATOR[g])
                        else:
                            return (((m.P_MIN[g, i] * s.u[g, t]) + s.p[g, t]
                                     + sum(P_SU[k] * s.v[g, t - k + SU_D + 2] if t - k + SU_D + 2 in m.T else 0 for k in
                                           range(1, SU_D + 1))
                                     + sum(P_SD[k] * s.w[g, t - k + 2] if t - k + 2 in m.T else 0 for k in
                                           range(2, SD_D + 2))
                                     ) * s.AVAILABILITY_INDICATOR[g])

                    # Remaining generators with no startup / shutdown directory defined
                    else:
                        return (m.P_MIN[g, i] + s.p[g, t]) * s.AVAILABILITY_INDICATOR[g]

                # Total power output [MW]
                s.P_TOTAL = Expression(m.G, m.T, rule=total_power_output_rule)

                def generator_energy_output_rule(_s, g, t):
                    """
                    Total generator energy output [MWh]

                    No information regarding generator dispatch level in period prior to
                    t=0. Assume that all generators are at their minimum dispatch level.
                    """

                    if t != m.T.first():
                        return (s.P_TOTAL[g, t - 1] + s.P_TOTAL[g, t]) / 2

                    # Else, first interval (t-1 will be out of range)
                    else:
                        return (s.P0[g] + s.P_TOTAL[g, t]) / 2

                # Energy output for a given generator
                s.ENERGY = Expression(m.G, m.T, rule=generator_energy_output_rule)

                def storage_unit_energy_output_rule(_s, g, t):
                    """Energy output from storage units"""

                    if t != m.T.first():
                        return (s.p_out[g, t] + s.p_out[g, t - 1]) / 2

                    else:
                        return s.p_out[g, t] / 2

                # Energy output for a given storage unit
                s.ENERGY_OUT = Expression(m.G_C_STORAGE, m.T, rule=storage_unit_energy_output_rule)

                def storage_unit_energy_input_rule(_s, g, t):
                    """Energy input (charging) from storage units"""

                    if t != m.T.first():
                        return (s.p_in[g, t] + s.p_in[g, t - 1]) / 2

                    else:
                        return s.p_in[g, t] / 2

                # Energy input for a given storage unit
                s.ENERGY_IN = Expression(m.G_C_STORAGE, m.T, rule=storage_unit_energy_input_rule)

                def thermal_operating_costs_rule(_s):
                    """Cost to operate existing and candidate thermal units"""

                    return (
                        sum((m.C_MC[g, i] + (m.EMISSIONS_RATE[g] - m.baseline[i]) * m.permit_price[i]) * s.ENERGY[g, t]
                            + (m.C_SU[g, i] * s.v[g, t]) + (m.C_SD[g, i] * s.w[g, t])
                            for g in m.G_E_THERM.union(m.G_C_THERM) for t in m.T))

                # Existing and candidate thermal unit operating costs for given scenario
                s.C_OP_THERM = Expression(rule=thermal_operating_costs_rule)

                def hydro_operating_costs_rule(_s):
                    """Cost to operate existing hydro generators"""

                    return sum(m.C_MC[g, i] * s.ENERGY[g, t] for g in m.G_E_HYDRO for t in m.T)

                # Existing hydro unit operating costs (no candidate hydro generators)
                s.C_OP_HYDRO = Expression(rule=hydro_operating_costs_rule)

                def solar_operating_costs_rule(_s):
                    """Cost to operate existing and candidate solar units"""

                    return (sum(m.C_MC[g, i] * s.ENERGY[g, t] for g in m.G_E_SOLAR for t in m.T)
                            + sum(
                                (m.C_MC[g, i] - m.baseline[i] * m.permit_price[i]) * s.ENERGY[g, t] for g in m.G_C_SOLAR
                                for t in
                                m.T))

                # Existing and candidate solar unit operating costs (only candidate solar eligible for credits)
                s.C_OP_SOLAR = Expression(rule=solar_operating_costs_rule)

                def wind_operating_costs_rule(_s):
                    """Cost to operate existing and candidate wind generators"""

                    return (sum(m.C_MC[g, i] * s.ENERGY[g, t] for g in m.G_E_WIND for t in m.T)
                            + sum(
                                (m.C_MC[g, i] - m.baseline[i] * m.permit_price[i]) * s.ENERGY[g, t] for g in m.G_C_WIND
                                for t in
                                m.T))

                # Existing and candidate solar unit operating costs (only candidate solar eligible for credits)
                s.C_OP_WIND = Expression(rule=wind_operating_costs_rule)

                def storage_unit_charging_cost_rule(_s):
                    """Cost to charge storage unit"""

                    return sum(m.C_MC[g, i] * s.ENERGY_IN[g, t] for g in m.G_C_STORAGE for t in m.T)

                # Charging cost rule - no subsidy received when purchasing energy
                s.C_OP_STORAGE_CHARGING = Expression(rule=storage_unit_charging_cost_rule)

                def storage_unit_discharging_cost_rule(_s):
                    """
                    Cost to charge storage unit

                    Note: If storage units are included in the scheme this could create an undesirable outcome. Units
                    would be subsidised for each MWh they generate. Therefore they could be incentivised to continually charge
                    and then immediately discharge in order to receive the subsidy. For now assume the storage units are not
                    eligible to receive a subsidy for each MWh under the policy.
                    """
                    # - (m.baseline * m.permit_price))
                    return sum(m.C_MC[g, i] * s.ENERGY_OUT[g, t] for g in m.G_C_STORAGE for t in m.T)

                # Discharging cost rule - assumes storage units are eligible under REP scheme
                s.C_OP_STORAGE_DISCHARGING = Expression(rule=storage_unit_discharging_cost_rule)

                # Candidate storage unit operating costs
                s.C_OP_STORAGE = Expression(expr=s.C_OP_STORAGE_CHARGING + s.C_OP_STORAGE_DISCHARGING)

                def lost_load_cost_rule(_s):
                    """Value of lost-load"""

                    return sum((s.p_lost_up[z, t] + s.p_lost_down[z, t]) * m.C_LOST_LOAD for z in m.Z for t in m.T)

                # Total cost of lost-load
                s.C_OP_LOST_LOAD = Expression(rule=lost_load_cost_rule)

                def total_operating_cost_rule(_s):
                    """Total operating cost"""

                    return s.C_OP_THERM + s.C_OP_HYDRO + s.C_OP_SOLAR + s.C_OP_WIND + s.C_OP_STORAGE + s.C_OP_LOST_LOAD

                # Total operating cost
                s.C_OP_TOTAL = Expression(rule=total_operating_cost_rule)

                def storage_unit_energy_capacity_rule(_s, g):
                    """Energy capacity depends on installed capacity (variable in master problem)"""

                    return sum(m.x_c[g, y] for y in m.I if y <= i)

                # Capacity of storage unit [MWh]
                s.STORAGE_UNIT_ENERGY_CAPACITY = Expression(m.G_C_STORAGE, rule=storage_unit_energy_capacity_rule)

                def storage_unit_max_energy_interval_end_rule(_s, g):
                    """Maximum energy at end of storage interval. Assume equal to unit capacity."""

                    return s.STORAGE_UNIT_ENERGY_CAPACITY[g]

                # Max state of charge for storage unit at end of operating scenario (assume = unit capacity)
                s.STORAGE_INTERVAL_END_MAX_ENERGY = Expression(m.G_C_STORAGE,
                                                               rule=storage_unit_max_energy_interval_end_rule)

                def storage_unit_max_power_out_rule(_s, g):
                    """
                    Maximum discharging power of storage unit - set equal to energy capacity. Assumes
                    storage unit can completely discharge in 1 hour
                    """

                    return s.STORAGE_UNIT_ENERGY_CAPACITY[g]

                # Max MW out of storage device - discharging
                s.P_STORAGE_MAX_OUT = Expression(m.G_C_STORAGE, rule=storage_unit_max_power_out_rule)

                def storage_unit_max_power_in_rule(_s, g):
                    """
                    Maximum charging power of storage unit - set equal to energy capacity. Assumes
                    storage unit can completely charge in 1 hour
                    """

                    return s.STORAGE_UNIT_ENERGY_CAPACITY[g]

                # Max MW into storage device - charging
                s.P_STORAGE_MAX_IN = Expression(m.G_C_STORAGE, rule=storage_unit_max_power_in_rule)

                def scenario_emissions_rule(_s):
                    """
                    Total emissions for a given scenario

                    Note: Only thermal generators assumed to emit
                    """
                    return sum(s.ENERGY[g, t] * m.EMISSIONS_RATE[g] for g in m.G_E_THERM.union(m.G_C_THERM) for t in m.T)

                # Total emissions in a given scenario
                # TODO: Figure out how to handle scenario duration here
                s.TOTAL_EMISSIONS = Expression(rule=scenario_emissions_rule)

                def scenario_revenue_rule(_s):
                    """
                    Net policy revenue for a given scenario

                    Note: Only existing and candidate thermal, and candidate wind and solar eligible under scheme
                    (payments to existing renewables wouldn't lead to any emissions abatement)
                    """
                    gens = m.G_E_THERM.union(m.G_C_THERM).union(m.G_C_WIND).union(m.G_C_SOLAR)

                    return sum((m.EMISSIONS_RATE[g] - m.baseline[i]) * s.ENERGY[g, t] for g in gens for t in m.T)

                # Net scheme revenue for a given scenario
                # TODO: Figure out how to handle scenario duration here
                s.TOTAL_REVENUE = Expression(rule=scenario_revenue_rule)

                return s

            def define_block_constraints(s):
                """Define model constraints"""

                def upward_power_reserve_rule(_s, r, t):
                    """Upward reserve constraint"""

                    # NEM region-zone map
                    # region_zone_map = self._get_nem_region_zone_map()
                    region_zone_map = self.nem_region_zone_map_dict

                    # Mapping describing the zone to which each generator is assigned
                    # gen_zone_map = self._get_generator_zone_map()
                    gen_zone_map = self.generator_zone_map

                    # Existing and candidate thermal gens + candidate storage units
                    gens = m.G_E_THERM.union(m.G_C_THERM).union(m.G_C_STORAGE)

                    return (sum(s.r_up[g, t] for g in gens if gen_zone_map[g] in region_zone_map[r])
                            >= m.RESERVE_UP[r])

                # Upward power reserve rule for each NEM region
                s.UPWARD_POWER_RESERVE = Constraint(m.R, m.T, rule=upward_power_reserve_rule)

                def power_production_rule(_s, g, t):
                    """Power production rule"""

                    if t != m.T.last():
                        return (s.p[g, t] + s.r_up[g, t]
                                <= ((m.P_MAX[g, i] - m.P_MIN[g, i]) * s.u[g, t])
                                - ((m.P_MAX[g, i] - m.RR_SD[g]) * s.w[g, t + 1])
                                + ((m.RR_SU[g] - m.P_MIN[g, i]) * s.v[g, t + 1]))
                    else:
                        return s.p[g, t] + s.r_up[g, t] <= (m.P_MAX[g, i] - m.P_MIN[g, i]) * s.u[g, t]

                # Power production
                s.POWER_PRODUCTION = Constraint(m.G_E_THERM.union(m.G_C_THERM), m.T, rule=power_production_rule)

                def ramp_rate_up_rule(_s, g, t):
                    """Ramp-rate up constraint"""

                    if t > m.T.first():
                        return (s.p[g, t] + s.r_up[g, t]) - s.p[g, t - 1] <= m.RR_UP[g]

                    else:
                        # Ramp-rate for first interval
                        return s.p[g, t] - s.P0[g] <= m.RR_UP[g]

                # Ramp-rate up limit
                s.RAMP_RATE_UP = Constraint(m.G_E_THERM.union(m.G_C_THERM), m.T, rule=ramp_rate_up_rule)

                def ramp_rate_down_rule(_s, g, t):
                    """Ramp-rate down constraint"""

                    if t > m.T.first():
                        return - s.p[g, t] + s.p[g, t - 1] <= m.RR_DOWN[g]

                    else:
                        # Ramp-rate for first interval
                        return - s.p[g, t] + s.P0[g] <= m.RR_DOWN[g]

                # Ramp-rate up limit
                s.RAMP_RATE_DOWN = Constraint(m.G_E_THERM.union(m.G_C_THERM), m.T, rule=ramp_rate_down_rule)

                def existing_wind_output_min_rule(_s, g, t):
                    """Constrain minimum output for existing wind generators"""

                    return s.P_TOTAL[g, t] >= 0

                # Minimum wind output for existing generators
                s.EXISTING_WIND_MIN_OUTPUT = Constraint(m.G_E_WIND, m.T, rule=existing_wind_output_min_rule)

                def wind_output_max_rule(_s, g, t):
                    """
                    Constrain maximum output for wind generators

                    Note: Candidate unit output depends on investment decisions
                    """

                    # Get wind bubble to which generator belongs
                    if g in m.G_E_WIND:

                        # If an existing generator
                        # bubble = self.existing_wind_bubble_map.loc[g, 'BUBBLE']
                        bubble = self.existing_wind_bubble_map_dict['BUBBLE'][g]

                    elif g in m.G_C_WIND:

                        # If a candidate generator
                        # bubble = self.candidate_units.loc[g, ('PARAMETERS', 'WIND_BUBBLE')]
                        bubble = self.candidate_units_dict[('PARAMETERS', 'WIND_BUBBLE')][g]

                    else:
                        raise Exception(f'Unexpected generator: {g}')

                    return s.P_TOTAL[g, t] <= s.Q_WIND[bubble, t] * m.P_MAX[g, i]

                # Max output from existing wind generators
                s.EXISTING_WIND_MAX_OUTPUT = Constraint(m.G_E_WIND.union(m.G_C_WIND), m.T, rule=wind_output_max_rule)

                def solar_output_max_rule(_s, g, t):
                    """
                    Constrain maximum output for solar generators

                    Note: Candidate unit output depends on investment decisions
                    """

                    # Get NEM zone
                    if g in m.G_E_SOLAR:
                        # If an existing generator
                        # zone = self.existing_units.loc[g, ('PARAMETERS', 'NEM_ZONE')]
                        zone = self.existing_units_dict[('PARAMETERS', 'NEM_ZONE')][g]

                        # Assume existing arrays are single-axis tracking arrays
                        technology = 'SAT'

                    elif g in m.G_C_SOLAR:
                        # If a candidate generator
                        # zone = self.candidate_units.loc[g, ('PARAMETERS', 'ZONE')]
                        zone = self.candidate_units_dict[('PARAMETERS', 'ZONE')][g]

                        # Extract technology type from unit ID
                        technology = g.split('-')[-1]

                    else:
                        raise Exception(f'Unexpected generator: {g}')

                    return s.P_TOTAL[g, t] <= s.Q_SOLAR[zone, technology, t] * m.P_MAX[g, i]

                # Max output from existing wind generators
                s.EXISTING_SOLAR_MAX_OUTPUT = Constraint(m.G_E_SOLAR.union(m.G_C_SOLAR), m.T,
                                                         rule=solar_output_max_rule)

                def hydro_output_max_rule(_s, g, t):
                    """
                    Constrain hydro output to registered capacity of existing plant

                    Note: Assume no investment in hydro over model horizon (only consider existing units)
                    """

                    return s.P_TOTAL[g, t] <= s.P_HYDRO_HISTORIC[g, t]

                # Max output from existing hydro generators
                s.EXISTING_HYDRO_MAX_OUTPUT = Constraint(m.G_E_HYDRO, m.T, rule=hydro_output_max_rule)

                # TODO: May want to add energy constraint for hydro units

                def thermal_generator_max_output_rule(_s, g, t):
                    """Max MW output for thermal generators"""

                    return s.P_TOTAL[g, t] <= m.P_MAX[g, i]

                # Max output for existing and candidate thermal generators
                s.EXISTING_THERMAL_MAX_OUTPUT = Constraint(m.G_E_THERM.union(m.G_C_THERM), m.T,
                                                           rule=thermal_generator_max_output_rule)

                def storage_max_charge_rate_rule(_s, g, t):
                    """Maximum charging power for storage units [MW]"""

                    return s.p_in[g, t] <= s.P_STORAGE_MAX_IN[g]

                # Storage unit max charging power
                s.STORAGE_MAX_CHARGE_RATE = Constraint(m.G_C_STORAGE, m.T, rule=storage_max_charge_rate_rule)

                def storage_max_discharge_rate_rule(_s, g, t):
                    """Maximum discharging power for storage units [MW]"""

                    return s.p_out[g, t] + s.r_up[g, t] <= s.P_STORAGE_MAX_OUT[g]

                # Storage unit max charging power
                s.STORAGE_MAX_DISCHARGE_RATE = Constraint(m.G_C_STORAGE, m.T, rule=storage_max_discharge_rate_rule)

                def state_of_charge_rule(_s, g, t):
                    """Energy within a given storage unit [MWh]"""

                    return s.y[g, t] <= s.STORAGE_UNIT_ENERGY_CAPACITY[g]

                # Energy within storage unit [MWh]
                s.STATE_OF_CHARGE = Constraint(m.G_C_STORAGE, m.T, rule=state_of_charge_rule)

                def storage_energy_transition_rule(_s, g, t):
                    """Constraint that couples energy + power between periods for storage units"""

                    if t != m.T.first():
                        return s.y[g, t] == s.y[g, t - 1] + (m.BATTERY_EFFICIENCY[g] * s.p_in[g, t]) - (
                                s.p_out[g, t] / m.BATTERY_EFFICIENCY[g])
                    else:
                        # Assume battery completely discharged in first period
                        return s.y[g, t] == s.Y0[g] + (m.BATTERY_EFFICIENCY[g] * s.p_in[g, t]) - (
                                s.p_out[g, t] / m.BATTERY_EFFICIENCY[g])

                # Account for inter-temporal energy transition within storage units
                s.STORAGE_ENERGY_TRANSITION = Constraint(m.G_C_STORAGE, m.T, rule=storage_energy_transition_rule)

                def storage_interval_end_min_energy_rule(_s, g):
                    """Lower bound on permissible amount of energy in storage unit at end of operating scenario"""

                    return m.STORAGE_INTERVAL_END_MIN_ENERGY[g] <= s.y[g, m.T.last()]

                # Minimum amount of energy that must be in storage unit at end of operating scenario
                s.STORAGE_INTERVAL_END_MIN_ENERGY_CONS = Constraint(m.G_C_STORAGE,
                                                                    rule=storage_interval_end_min_energy_rule)

                def storage_interval_end_max_energy_rule(_s, g):
                    """Upper bound on permissible amount of energy in storage unit at end of operating scenario"""

                    return s.y[g, m.T.last()] <= s.STORAGE_INTERVAL_END_MAX_ENERGY[g]

                # Maximum amount of energy that must be in storage unit at end of operating scenario
                s.STORAGE_INTERVAL_END_MAX_ENERGY_CONS = Constraint(m.G_C_STORAGE,
                                                                    rule=storage_interval_end_max_energy_rule)

                def power_balance_rule(_s, z, t):
                    """Power balance for each NEM zone"""

                    # Existing units within zone
                    # existing_units = self.existing_units[
                    #     self.existing_units[('PARAMETERS', 'NEM_ZONE')] == z].index.tolist()
                    existing_units = [gen for gen, zone in self.existing_units_dict[('PARAMETERS', 'NEM_ZONE')].items()
                                      if zone == z]

                    # Candidate units within zone
                    # candidate_units = self.candidate_units[
                    #     self.candidate_units[('PARAMETERS', 'ZONE')] == z].index.tolist()
                    candidate_units = [gen for gen, zone in self.candidate_units_dict[('PARAMETERS', 'ZONE')].items()
                                       if zone == z]

                    # All generators within a given zone
                    generators = existing_units + candidate_units

                    # Storage units within a given zone
                    # storage_units = (self.battery_properties.loc[self.battery_properties['NEM_ZONE'] == z, 'NEM_ZONE']
                    #                  .index.tolist())
                    storage_units = [gen for gen, zone in self.battery_properties_dict['NEM_ZONE'].items() if zone == z]

                    return (sum(s.P_TOTAL[g, t] for g in generators) - s.DEMAND[z, t]
                            - sum(m.INCIDENCE_MATRIX[l, z] * s.p_flow[l, t] for l in m.L)
                            + sum(s.p_out[g, t] - s.p_in[g, t] for g in storage_units)
                            + s.p_lost_up[z, t] - s.p_lost_down[z, t] == 0)

                # Power balance constraint for each zone and time period
                s.POWER_BALANCE = Constraint(m.Z, m.T, rule=power_balance_rule)

                def powerflow_min_constraint_rule(_s, l, t):
                    """Minimum powerflow over a link connecting adjacent NEM zones"""

                    return s.p_flow[l, t] >= m.POWERFLOW_MIN[l]

                # Constrain max power flow over given network link
                s.POWERFLOW_MIN_CONS = Constraint(m.L_I, m.T, rule=powerflow_min_constraint_rule)

                def powerflow_max_constraint_rule(_s, l, t):
                    """Maximum powerflow over a link connecting adjacent NEM zones"""

                    return s.p_flow[l, t] <= m.POWERFLOW_MAX[l]

                # Constrain max power flow over given network link
                s.POWERFLOW_MAX_CONS = Constraint(m.L_I, m.T, rule=powerflow_max_constraint_rule)

            def construct_block(s):
                """Construct block for each operating scenario"""
                print(f'Constructing block: Year - {i}, Scenario - {o}')
                print('###############################################')

                # Define block parameters for given operating scenario
                start = time.time()
                s = define_block_parameters(s)
                print(f'Defined block parameters in: {time.time() - start}s')

                # Define block variables
                start = time.time()
                s = define_block_variables(s)
                print(f'Defined block variables in: {time.time() - start}s')

                # Define block expressions
                start = time.time()
                s = define_block_expressions(s)
                print(f'Defined block expressions in: {time.time() - start}s')

                # Define block constraints
                start = time.time()
                s = define_block_constraints(s)
                print(f'Defined block constraints in: {time.time() - start}s')

                return s

            # Construct block
            construct_block(s)

        # Construct block
        m.SCENARIO = Block(m.I, m.O, rule=operating_scenario_block_rule)

        return m

    @staticmethod
    def define_policy_constraints(m):
        """Define emissions and revenue constraints over model horizon"""

        # Revenue constraint - must break-even over model horizon
        m.REVENUE_CONSTRAINT = Constraint(
            expr=sum(m.SCENARIO[i, o].TOTAL_REVENUE for i in m.I for o in m.O) + m.revenue_shortfall
                 == m.REVENUE_TARGET)

        # Emissions constraint - must be less than some target, else penalty imposed for each unit above target
        m.EMISSIONS_CONSTRAINT = Constraint(expr=sum(m.SCENARIO[i, o].TOTAL_EMISSIONS for i in m.I for o in m.O)
                                                 == m.EMISSIONS_TARGET + m.emissions_target_exceeded)

        return m

    @staticmethod
    def define_objective(m):
        """Define objective function"""

        # Minimise operating cost
        m.OBJECTIVE = Objective(expr=sum(m.SCENARIO[i, o].C_OP_TOTAL for i in m.I for o in m.O)
                                     + m.C_EMISSIONS_VIOLATION + m.C_REVENUE_VIOLATION, sense=minimize)

        return m

    def solve_model(self, m):
        """Solve model for a given operating scenario"""

        # Solve model
        self.opt.solve(m, tee=True, options=self.solver_options, keepfiles=self.keepfiles)

        # Log infeasible constraints if they exist
        log_infeasible_constraints(m)

    def construct_model(self):
        """Construct components of unit commitment model"""

        # Base model object
        m = ConcreteModel()

        # Define sets
        start = time.time()
        m = self.define_sets(m)
        print(f'Defined sets in: {time.time() - start}s')

        # Define parameters - common to all blocks
        start = time.time()
        m = self.define_parameters(m)
        print(f'Defined parameters in: {time.time() - start}s')

        # Define variables - common to all blocks
        start = time.time()
        m = self.define_variables(m)
        print(f'Defined variables in: {time.time() - start}s')

        # Define expressions - common to all block
        start = time.time()
        m = self.define_expressions(m)
        print(f'Defined expressions in: {time.time() - start}s')

        # Define constraints - used to get sensitivities for master problem variables
        start = time.time()
        m = self.define_constraints(m)
        print(f'Defined constraints in: {time.time() - start}s')

        # Define blocks
        start = time.time()
        m = self.define_blocks(m)
        print(f'Defined blocks in: {time.time() - start}s')

        # Define objective
        start = time.time()
        m = self.define_objective(m)
        print(f'Defined objective in: {time.time() - start}s')

        return m

    @staticmethod
    def fix_baseline(m):
        """Fix emissions intensity baseline"""
        for i in m.I:
            m.baseline[i].fix(m.FIXED_BASELINE[i].value)

        return m

    @staticmethod
    def unfix_baseline(m):
        """Fix emissions intensity baseline"""
        for i in m.I:
            m.baseline[i].unfix()

        return m

    @staticmethod
    def fix_permit_price(m):
        """Fix permit price"""
        for i in m.I:
            m.permit_price[i].fix(m.FIXED_PERMIT_PRICE[i].value)

        return m

    @staticmethod
    def unfix_permit_price(m):
        """Unfix permit price"""
        for i in m.I:
            m.permit_price[i].unfix()

        return m

    @staticmethod
    def fix_discrete_capacity_options(m):
        """Fix binary variables used in discrete capacity sizing"""
        for g in m.G_C_THERM:
            for i in m.I:
                for n in m.G_C_THERM_SIZE_OPTIONS:
                    m.d[g, i, n].fix(m.FIXED_D[g, i, n].value)

        return m

    @staticmethod
    def fix_continuous_capacity_options(m):
        """Fix variables used in continuous capacity sizing options"""
        for g in m.G_C_WIND.union(m.G_C_SOLAR).union(m.G_C_STORAGE):
            for i in m.I:
                m.x_c[g, i].fix(m.FIXED_X_C[g, i].value)

        return m

    @staticmethod
    def fix_uc_integer_variables(m, last_value=True):
        """Fix unit commitment integer variables - allows duals to be obtained when re-solving"""

        for g in m.G_E_THERM.union(m.G_C_THERM):
            for i in m.I:
                for o in m.O:
                    for t in m.T:
                        if last_value:
                            m.SCENARIO[i, o].v[g, t].fix()
                            m.SCENARIO[i, o].w[g, t].fix()
                            m.SCENARIO[i, o].u[g, t].fix()
                        else:
                            m.SCENARIO[i, o].v[g, t].fix(m.SCENARIO[i, o].FIXED_V[g, t].value)
                            m.SCENARIO[i, o].w[g, t].fix(m.SCENARIO[i, o].FIXED_W[g, t].value)
                            m.SCENARIO[i, o].u[g, t].fix(m.SCENARIO[i, o].FIXED_U[g, t].value)
        return m

    @staticmethod
    def fix_uc_integer_variable_on_state(m):
        """Fix on-state integer variable"""

        for g in m.G_E_THERM.union(m.G_C_THERM):
            for i in m.I:
                for o in m.O:
                    for t in m.T:
                        m.SCENARIO[i, o].u[g, t].fix(m.SCENARIO[i, o].FIXED_U[g, t].value)

        return m

    @staticmethod
    def fix_uc_integer_variable_startup_state(m):
        """Fix shutdown state integer variable"""

        for g in m.G_E_THERM.union(m.G_C_THERM):
            for i in m.I:
                for o in m.O:
                    for t in m.T:
                        m.SCENARIO[i, o].w[g, t].fix(m.SCENARIO[i, o].FIXED_V[g, t].value)

        return m

    @staticmethod
    def fix_uc_integer_variable_shutdown_state(m):
        """Fix shutdown state integer variable"""

        for g in m.G_E_THERM.union(m.G_C_THERM):
            for i in m.I:
                for o in m.O:
                    for t in m.T:
                        m.SCENARIO[i, o].w[g, t].fix(m.SCENARIO[i, o].FIXED_W[g, t].value)

        return m

    @staticmethod
    def unfix_continuous_capacity_options(m):
        """Unfix variables used in continuous capacity sizing options"""
        for g in m.G_C_WIND.union(m.G_C_SOLAR).union(m.G_C_STORAGE):
            for i in m.I:
                # Unfix capacity size to zero for now
                m.x_c[g, i].unfix()

        return m

    @staticmethod
    def unfix_discrete_capacity_options(m):
        """Unfix binary variables used in discrete capacity sizing"""
        for g in m.G_C_THERM:
            for i in m.I:
                for n in m.G_C_THERM_SIZE_OPTIONS:
                    m.d[g, i, n].unfix()
        return m

    @staticmethod
    def fix_uc_continuous_variables(m):
        """Fix unit commitment model continuous  variables"""
        for i in m.I:
            for o in m.O:
                for t in m.T:
                    for g in m.G:

                        # Power output above minimum dispatchable level of output [MW]
                        m.SCENARIO[i, o].p[g, t].fix()

                        if g in m.G_E_THERM.union(m.G_C_THERM).union(m.G_C_STORAGE):
                            # Upward reserve allocation [MW]
                            m.SCENARIO[i, o].r_up[g, t].fix()

                        if g in m.G_C_STORAGE:
                            # Storage unit charging (power in) [MW]
                            m.SCENARIO[i, o].p_in[g, t].fix()

                            # Storage unit discharging (power out) [MW]
                            m.SCENARIO[i, o].p_out[g, t].fix()

                            # Energy in storage unit [MWh]
                            m.SCENARIO[i, o].y[g, t].fix()

                    for l in m.L:
                        # Powerflow between NEM zones [MW]
                        m.SCENARIO[i, o].p_flow[l, t].fix()

                    for z in m.Z:
                        # Lost load - up [MW]
                        m.SCENARIO[i, o].p_lost_up[z, t].fix()

                        # Lost load - down [MW]
                        m.SCENARIO[i, o].p_lost_down[z, t].fix()

        return m

    @staticmethod
    def update_fixed_baseline(m, fixed_baseline):
        """Update fixed baseline - obtained from master problem"""

        for i in m.I:
            # m.FIXED_BASELINE[i] = fixed_baseline[i]
            m.FIXED_BASELINE[i] = 0

        return m

    @staticmethod
    def update_fixed_permit_price(m, fixed_permit_price):
        """Update fixed permit price - obtained from master problem"""

        for i in m.I:
            # m.FIXED_PERMIT_PRICE[i] = fixed_permit_price[i]
            m.FIXED_PERMIT_PRICE[i] = 0
        return m

    @staticmethod
    def update_fixed_capacity_continuous(m, continuous_capacity):
        """Update installed capacity for units with continuous capacity sizing option"""

        for i in m.I:
            for g in m.G_C_WIND.union(m.G_C_SOLAR).union(m.G_C_STORAGE):
                # m.FIXED_X_C[g, i] = continuous_capacity[g][i]
                m.FIXED_X_C[g, i] = 0

        return m

    @staticmethod
    def update_fixed_capacity_discrete(m, discrete_capacity):
        """Update installed capacity for units with discrete capacity sizing option"""

        for i in m.I:
            for g in m.G_C_THERM:
                for n in m.G_C_THERM_SIZE_OPTIONS:
                    # m.FIXED_D[g, i, n] = discrete_capacity[g][i][n]
                    if n == 0:
                        m.FIXED_D[g, i, n] = 1
                    else:
                        m.FIXED_D[g, i, n] = 0

        return m

    @staticmethod
    def update_fixed_shutdown_state(m, shutdown_state):
        """Update fixed shutdown state variables"""
        for g in m.G_E_THERM.union(m.G_C_THERM):
            for i in m.I:
                for o in m.O:
                    for t in m.T:
                        # m.SCENARIO[i, o].FIXED_W[g, t] = shutdown_state[g][i][o][t]
                        m.SCENARIO[i, o].FIXED_W[g, t] = 0

        return m

    @staticmethod
    def update_fixed_startup_state(m, startup_state):
        """Update fixed startup state variables"""
        for g in m.G_E_THERM.union(m.G_C_THERM):
            for i in m.I:
                for o in m.O:
                    for t in m.T:
                        # m.SCENARIO[i, o].FIXED_V[g, t] = startup_state[g][i][o][t]
                        m.SCENARIO[i, o].FIXED_V[g, t] = 0

        return m

    @staticmethod
    def update_fixed_on_state(m, on_state):
        """Update fixed startup state variables"""
        for g in m.G_E_THERM.union(m.G_C_THERM):
            for i in m.I:
                for o in m.O:
                    for t in m.T:
                        # m.SCENARIO[i, o].FIXED_U[g, t] = on_state[g][i][o][t]
                        m.SCENARIO[i, o].FIXED_U[g, t] = 1

        return m


if __name__ == '__main__':
    start = time.time()
    # Directory containing files from which dataset is derived
    raw_data_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'data')

    # Directory containing core data files
    data_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '1_collect_data', 'output')

    # Directory containing input traces
    input_traces_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '2_input_traces', 'output')

    # Object used to construct UC model
    uc = UnitCommitmentModel(raw_data_directory, data_directory, input_traces_directory)

    # Model
    model = uc.construct_model()

    # Prepare to read suffix values (dual values)
    model.dual = Suffix(direction=Suffix.IMPORT)

    start_iteration = time.time()

    # Update fixed variables obtained from master program
    model = uc.update_fixed_baseline(model, 'test')
    model = uc.update_fixed_permit_price(model, 'test')
    model = uc.update_fixed_capacity_discrete(model, 'test')
    model = uc.update_fixed_capacity_continuous(model, 'test')

    model = uc.update_fixed_on_state(model, 'test')
    model = uc.update_fixed_startup_state(model, 'test')
    model = uc.update_fixed_shutdown_state(model, 'test')

    print(f'Constructed model in: {time.time() - start}s')

    # Fix master problem variables
    start = time.time()
    model = uc.fix_permit_price(model)
    print(f'Fixed permit_price in: {time.time() - start}s')

    start = time.time()
    model = uc.fix_baseline(model)
    print(f'Fixed baseline in: {time.time() - start}s')

    start = time.time()
    model = uc.fix_discrete_capacity_options(model)
    print(f'Fixed discrete_capacity_options in: {time.time() - start}s')

    start = time.time()
    model = uc.fix_continuous_capacity_options(model)
    print(f'Fixed continuous_capacity_options in: {time.time() - start}s')

    start = time.time()
    model = uc.fix_uc_integer_variables(model, last_value=False)
    print(f'Fixed uc_integer_variables in: {time.time() - start}s')

    # Solve model
    start = time.time()
    uc.solve_model(model)
    print(f'Solved model in: {time.time() - start}')

    # Fix integer variables
    # model = uc.fix_uc_integer_variables(model)

    # # Re-solve model (obtain dual variables0
    # uc.solve_model(model)
    #
    # # Get sensitivities for unit capacities and marginal prices
    # # ---------------------------------------------------------
    # # Fix integer variables for unit commitment problem
    # model = uc.fix_uc_integer_variables(model)
    #
    # # Unfix capacity variables
    # model = uc.unfix_continuous_capacity_options(model)
    # model = uc.unfix_discrete_capacity_options(model)
    #
    # # Solve model
    # uc.solve_model(model)
    #
    # # Get sensitivities for permit price and baseline
    # # -----------------------------------------------
    # # Fix capacity variables
    # model = uc.fix_continuous_capacity_options(model)
    # model = uc.fix_discrete_capacity_options(model)
    #
    # # Fix all unit commitment primal variables
    # model = uc.fix_uc_continuous_variables(model)
    #
    # # Unfix permit price and baseline
    # m_scenario = uc.unfix_permit_price(model)
    # m_scenario = uc.unfix_baseline(m_scenario)
    #
    # # Solve model
    # uc.solve_model(m_scenario)
    #
    # print(f'Completed iteration in: {time.time() - start_iteration}s')
