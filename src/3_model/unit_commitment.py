import os
import pandas as pd
from math import ceil
from collections import OrderedDict

from pyomo.environ import *
from pyomo.core.expr import current as EXPR
from pyomo.util.infeasible import log_infeasible_constraints


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
        self.solver_options = {}
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

        # All candidate generators + storage units
        m.G_C = m.G_C_THERM.union(m.G_C_WIND).union(m.G_C_SOLAR)

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

        # Model year - must update each time model is run
        m.YEAR = Param(initialize=2016, mutable=True)

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
        m.E = Param(m.G.union(m.G_C_STORAGE), rule=emissions_intensity_rule)

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

        def existing_generator_registered_capacities_rule(m, g):
            """Registered capacities of existing generators"""

            return float(self.existing_units.loc[g, ('PARAMETERS', 'REG_CAP')])

        # Registered capacities of existing generators
        m.EXISTING_GEN_REG_CAP = Param(m.G_E, rule=existing_generator_registered_capacities_rule)

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
                marginal_cost = float(1)

            elif (g in m.G_E_WIND) or (g in m.G_E_SOLAR) or (g in m.G_E_HYDRO):
                # Marginal cost = VOM cost for wind and solar generators
                marginal_cost = self.existing_units.loc[g, ('PARAMETERS', 'VOM')].astype(float)

            elif (g in m.G_C_WIND) or (g in m.G_C_SOLAR):
                # Marginal cost = VOM cost for wind and solar generators
                marginal_cost = self.candidate_units.loc[g, ('PARAMETERS', 'VOM')].astype(float)

            elif g in m.G_C_STORAGE:
                # Assume marginal cost = VOM cost of typical hydro generator (7 $/MWh)
                marginal_cost = float(7)

            else:
                raise Exception(f'Unexpected generator: {g}')

            assert marginal_cost >= 0, 'Cannot have negative marginal cost'

            return marginal_cost

        # Marginal costs for all generators (must be updated each time model is run)
        m.C_MC = Param(m.G.union(m.G_C_STORAGE), rule=marginal_cost_rule, mutable=True)

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

        # Capacity factor solar, for given technology j (must be updated each time model is run)
        m.Q_SOLAR = Param(m.Z, m.G_C_SOLAR_TECHNOLOGIES, m.T, initialize=0, mutable=True)

        # Zone demand (must be updated each time model is run)
        m.D = Param(m.Z, m.T, initialize=0, mutable=True)

        # Max MW out of storage device - discharging (must be updated each time model is run)
        m.P_STORAGE_MAX_OUT = Param(m.G_C_STORAGE, initialize=0)

        # Max MW into storage device - charging (must be updated each time model is run)
        m.P_STORAGE_MAX_IN = Param(m.G_C_STORAGE, initialize=0)

        # Duration of operating scenario (must be updated each time model is run)
        m.SCENARIO_DURATION = Param(initialize=0, mutable=True)

        # Value of lost-load [$/MWh]
        m.C_L = Param(initialize=float(1e4), mutable=True)

        # Initial state for thermal generators
        m.u0 = Param(m.G_E_THERM.union(m.G_C_THERM), within=Binary, mutable=True, initialize=1)

        # TODO: If unit retires it should be set to 0. Handle when updating parameters. Consider baseload state also.

        # TODO: how to handle storage unit power and energy?

        # Power output in interval prior to model start (handle when updating parameters)
        m.p0 = Param(m.G, mutable=True, within=NonNegativeReals, initialize=0)

        def battery_efficiency_rule(m, g):
            """Charging and discharging efficiency for a given storage unit"""

            return self.battery_properties.loc[g, 'CHARGE_EFFICIENCY']

        # Battery charging / discharging efficiency
        m.ETA = Param(m.G_C_STORAGE, rule=battery_efficiency_rule)

        # Min state of charge for storage unit at end of operating scenario (assume = 0)
        m.STORAGE_INTERVAL_END_MIN_ENERGY = Param(m.G_C_STORAGE, initialize=0)

        def powerflow_min_rule(m, l):
            """Minimum powerflow over network link"""

            return float(-self.powerflow_limits[l]['reverse'])

        # Lower bound for powerflow over link
        m.POWERFLOW_MIN = Param(m.L_I, rule=powerflow_min_rule)

        def powerflow_max_rule(m, l):
            """Maximum powerflow over network link"""

            return float(self.powerflow_limits[l]['forward'])

        # Lower bound for powerflow over link
        m.POWERFLOW_MAX = Param(m.L_I, rule=powerflow_max_rule)

        # Unit availability indicator (indicates if unit is available / not retired). Must be set each time model run.
        m.AVAILABILITY_INDICATOR = Param(m.G, initialize=1, within=Binary, mutable=True)

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

        # Power flow between NEM zones
        m.p_L = Var(m.L, m.T, initialize=0)

        # Lost load - up
        m.p_LL_up = Var(m.Z, m.T, within=NonNegativeReals, initialize=0)

        # Lost load - down
        m.p_LL_down = Var(m.Z, m.T, within=NonNegativeReals, initialize=0)

        # Variables to be determined in master program (will be fixed in sub-problems)
        # ----------------------------------------------------------------------------
        # Capacity of candidate units (defined for all years in model horizon)
        m.x_C = Var(m.G_C.union(m.G_C_STORAGE), m.I, within=NonNegativeReals, initialize=0)

        # Emissions intensity baseline [tCO2/MWh] (must be fixed each time model is run)
        m.baseline = Var(initialize=0)

        # Permit price [$/tCO2] (must be fixed each time model is run)
        m.permit_price = Var(initialize=0)

        return m

    @staticmethod
    def define_expressions(m):
        """Define model expressions"""

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
                return sum(m.x_C[g, i] for i in range(2016, m.YEAR.value))

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

                # If a storage unit
                if g in m.G_C_STORAGE:
                    return ((m.p_in[g, t] + m.p_out[g, t]) + (m.p_in[g, t - 1] + m.p_out[g, t - 1])) / 2

                else:
                    return (m.P_TOTAL[g, t - 1] + m.P_TOTAL[g, t]) / 2

            # Else, first interval (t-1 will be out of range)
            else:

                # If a storage unit assume no power output in preceding interval
                if g in m.G_C_STORAGE:
                    return (m.p_in[g, t] + m.p_out[g, t]) / 2

                else:
                    return (m.p0[g] + m.P_TOTAL[g, t]) / 2

        # Energy output for a given generator
        m.e = Expression(m.G.union(m.G_C_STORAGE), m.T, rule=generator_energy_output_rule)

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

        def lost_load_cost_rule(m):
            """Value of lost-load"""

            return sum((m.p_LL_up[z, t] + m.p_LL_down[z, t]) * m.C_L for z in m.Z for t in m.T)

        # Total cost of lost-load
        m.C_OP_LOST_LOAD = Expression(rule=lost_load_cost_rule)

        def total_operating_cost_rule(m):
            """Total operating cost"""

            return m.C_OP_THERM + m.C_OP_HYDRO + m.C_OP_SOLAR + m.C_OP_WIND + m.C_OP_STORAGE + m.C_OP_LOST_LOAD

        # Total operating cost
        m.C_OP_TOTAL = Expression(rule=total_operating_cost_rule)

        def storage_unit_energy_capacity_rule(m, g):
            """Energy capacity depends on installed capacity (variable in master problem)"""

            return sum(m.x_C[g, i] for i in range(2016, m.YEAR.value + 1))

        # Capacity of storage unit [MWh]
        m.STORAGE_UNIT_ENERGY_CAPACITY = Expression(m.G_C_STORAGE, rule=storage_unit_energy_capacity_rule)

        def storage_unit_max_energy_interval_end_rule(m, g):
            """Maximum energy at end of storage interval. Assume equal to unit capacity."""

            return m.STORAGE_UNIT_ENERGY_CAPACITY[g]

        # Max state of charge for storage unit at end of operating scenario (assume = unit capacity)
        m.STORAGE_INTERVAL_END_MAX_ENERGY = Expression(m.G_C_STORAGE, rule=storage_unit_max_energy_interval_end_rule)

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

            return sum(m.r_up[g, t] for g in gens if generator_zone_map.loc[g] in region_zone_map.loc[r]) >= m.D_UP[r]

        # Upward power reserve rule for each NEM zone
        # m.UPWARD_POWER_RESERVE = Constraint(m.R, m.T, rule=upward_power_reserve_rule)

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
        m.P_WIND_MAX = Constraint(m.G_E_WIND.union(m.G_C_WIND), m.T, rule=wind_output_max_rule)

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
        m.P_SOLAR_MAX = Constraint(m.G_E_SOLAR.union(m.G_C_SOLAR), m.T, rule=solar_output_max_rule)

        def hydro_output_max_rule(m, g, t):
            """
            Constrain hydro output to registered capacity of existing plant

            Note: Assume no investment in hydro over model horizon (only consider existing units)
            """

            return m.P_TOTAL[g, t] <= m.P_MAX[g]

        # Max output from existing hydro generators
        m.P_HYDRO_MAX = Constraint(m.G_E_HYDRO, m.T, rule=hydro_output_max_rule)

        # TODO: May want to add energy constraint for hydro units

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
                return m.y[g, t] == m.y[g, t - 1] + (m.ETA[g] * m.p_in[g, t]) - ((1 / m.ETA[g]) * m.p_out[g, t])
            else:
                return Constraint.Skip

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

            # return (sum(m.P_TOTAL[g, t] for g in m.G) - m.D[z, t]
            #         + m.p_LL_up[z, t] - m.p_LL_down[z, t] == 0)

            return (sum(m.P_TOTAL[g, t] for g in generators) - m.D[z, t]
                    - sum(m.INCIDENCE_MATRIX[l, z] * m.p_L[l, t] for l in m.L)
                    + sum(m.p_out[g, t] - m.p_in[g, t] for g in storage_units)
                    + m.p_LL_up[z, t] == 0)

        # Power balance constraint for each zone and time period
        m.POWER_BALANCE = Constraint(m.Z, m.T, rule=power_balance_rule)

        def powerflow_min_constraint_rule(m, l, t):
            """Minimum powerflow over a link connecting adjacent NEM zones"""

            return m.p_L[l, t] >= m.POWERFLOW_MIN[l]

        # Constrain max power flow over given network link
        m.POWERFLOW_MIN_CONS = Constraint(m.L_I, m.T, rule=powerflow_min_constraint_rule)

        def powerflow_max_constraint_rule(m, l, t):
            """Maximum powerflow over a link connecting adjacent NEM zones"""

            return m.p_L[l, t] <= m.POWERFLOW_MAX[l]

        # Constrain max power flow over given network link
        m.POWERFLOW_MAX_CONS = Constraint(m.L_I, m.T, rule=powerflow_max_constraint_rule)

        return m

    @staticmethod
    def define_objective(m):
        """Objective function - cost minimisation for each scenario"""

        m.OBJECTIVE = Objective(expr=m.C_OP_TOTAL, sense=minimize)

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

        # # Construct constraints
        m = self.define_constraints(m)

        # Construct objective function
        m = self.define_objective(m)

        return m

    def update_parameters(self, m, year, scenario, installed_storage_capacity):
        """Update model parameters for a given year"""

        def _update_model_year(m, year):
            """Update year for which model is to be run"""

            # Update model year
            m.YEAR = year

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

                m.C_MC[g] = float(self.existing_units.loc[g, ('FUEL_COST', year)])

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
                m.u0[g] = float(1)

        def _update_generator_availability(m, year):
            """Update whether unit is retired for a given year"""

            for g in m.G:
                m.AVAILABILITY_INDICATOR[g] = float(1)

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
                    m.D[z, t] = float(self.input_traces.loc[(year, scenario), ('DEMAND', z, t)])

        def _update_availability_indicator(m, year):
            """Update generator availability status (set = 0 if unit retires)"""

            for g in m.G:
                m.AVAILABILITY_INDICATOR[g] = float(1)

        def _update_storage_max_power_output(m, installed_storage_capacity):
            """Update max power output from storage units"""

            # For each storage unit
            for g in installed_storage_capacity.keys():
                # Update max output [MW]
                m.P_STORAGE_MAX_OUT[g] = installed_storage_capacity[g]

        def _update_storage_max_power_input(m, installed_storage_capacity):
            """Update max charging power for storage unit"""

            # For each storage unit
            for g in installed_storage_capacity.keys():
                # Update max output [MW]
                m.P_STORAGE_MAX_IN[g] = installed_storage_capacity[g]

        def _update_scenario_duration(m, year, scenario):
            """Update duration of operating scenario"""

            # TODO: Must figure out how scenario duration is handled in master and sub-problems
            m.SCENARIO_DURATION = float(1)

        def _update_all_parameters(m, year, scenario, installed_storage_capacity):
            """Run each function used to update model parameters"""

            _update_model_year(m, year)
            _update_short_run_marginal_costs(m, year)
            _update_u0(m, year)
            _update_generator_availability(m, year)
            _update_wind_capacity_factors(m, year, scenario)
            _update_solar_capacity_factors(m, year, scenario)
            _update_zone_demand(m, year, scenario)
            _update_availability_indicator(m, year)
            _update_storage_max_power_output(m, installed_storage_capacity)
            _update_storage_max_power_input(m, installed_storage_capacity)
            _update_scenario_duration(m, year, scenario)

        # Update model parameters
        _update_all_parameters(m, year, scenario, installed_storage_capacity)

        return m

    @staticmethod
    def fix_variables(m, candidate_unit_capacity, fixed_baseline, fixed_permit_price):
        """Fix variables in the sub-problem"""

        # For each candidate generator
        for g in m.G_C.union(m.G_C_STORAGE):

            # Each year in the model horizon
            for i in m.I:
                # Fix capacity of candidate generator
                # m.x_C[g, i].fix(candidate_unit_capacity[g][i])
                m.x_C[g, i].fix(0)

        # Fix emissions intensity baseline
        m.baseline.fix(fixed_baseline)

        # Fix permit price
        m.permit_price.fix(fixed_permit_price)

        return m

    def solve_model(self, m):
        """Solve model for a given operating scenario"""

        # Solve model
        self.opt.solve(m, tee=True, options=self.solver_options, keepfiles=self.keepfiles)

        # Log infeasible constraints if they exist
        log_infeasible_constraints(m)

        return m

    def run_operating_scenario(self, m, year, scenario, installed_storage_capacity, candidate_unit_capacity,
                               fixed_baseline, fixed_permit_price):
        """Run model for a given operating scenario"""

        # Update parameters
        m = self.update_parameters(m, year, scenario, installed_storage_capacity)

        # Fix variables
        m = self.fix_variables(m, candidate_unit_capacity, fixed_baseline, fixed_permit_price)

        # Solve model
        m = self.solve_model(m)

        return m


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

    m1 = uc.run_operating_scenario(model, 2018, 1, {}, {}, 0.8, 50)



