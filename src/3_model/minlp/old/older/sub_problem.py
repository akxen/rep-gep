import os

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

    def construct_model(self):
        """
        Initialise unit commitment model
        """
        # Create concrete model
        m = ConcreteModel()

        # Sets
        # ----
        # NEM zones
        m.Z = Set(initialize=self._get_nem_zones())

        # NEM regions
        m.R = Set(initialize=self._get_nem_regions())

        # NEM wind bubbles
        m.B = Set(initialize=self._get_wind_bubbles())

        # Existing thermal units
        m.G_E_THERM = Set(initialize=self._get_existing_thermal_unit_ids())

        # Candidate thermal units
        m.G_C_THERM = Set(initialize=self._get_candidate_thermal_unit_ids())

        # Candidate thermal unit size options
        m.G_C_THERM_SIZE_OPTIONS = RangeSet(0, 2, ordered=True)

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

        # All existing generators
        m.G_E = m.G_E_THERM.union(m.G_E_WIND).union(m.G_E_SOLAR).union(m.G_E_HYDRO)

        # All candidate generators + storage units
        m.G_C = m.G_C_THERM.union(m.G_C_WIND).union(m.G_C_SOLAR).union(m.G_C_STORAGE)

        # All generators + storage units
        m.G = m.G_E.union(m.G_C)

        # Investment periods
        m.I = RangeSet(2016, 2049, ordered=True)

        # Operating scenarios
        m.O = RangeSet(0, 9, ordered=True)

        # Hours within operating scenario o
        m.T = RangeSet(0, 23, ordered=True)

        # Parameters
        # ----------
        def candidate_build_costs_rule(m, g, i):
            """Candidate unit investment cost"""

            # Candidate thermal, wind, and solar generator build costs
            if g in m.G_C_THERM.union(m.G_C_WIND).union(m.G_C_SOLAR):
                return float(self.candidate_units.loc[g, ('BUILD_COST', i)])

            # Candidate storage unit build costs
            elif g in m.G_C_STORAGE:
                return float(self.battery_build_costs.loc[g, i])

        m.I_C = Param(m.G_C_THERM.union(m.G_C_WIND).union(m.G_C_SOLAR).union(m.G_C_STORAGE), m.I,
                      rule=candidate_build_costs_rule)

        def candidate_thermal_size_options_rule(m, g, i, n):
            """Candidate thermal unit discrete size options"""

            if self.candidate_units.loc[g, ('PARAMETERS', 'TECHNOLOGY_PRIMARY')] == 'GAS':

                if n == 0:
                    return float(100)
                elif n == 1:
                    return float(200)
                elif n == 2:
                    return float(300)
                else:
                    raise Exception('Unexpected size index')

            elif self.candidate_units.loc[g, ('PARAMETERS', 'TECHNOLOGY_PRIMARY')] == 'COAL':

                if n == 0:
                    return float(300)
                elif n == 1:
                    return float(500)
                elif n == 2:
                    return float(700)
                else:
                    raise Exception('Unexpected size index')

            else:
                raise Exception('Unexpected technology type')

        m.X_C_THERM = Param(m.G_C_THERM, m.I, m.G_C_THERM_SIZE_OPTIONS, rule=candidate_thermal_size_options_rule)

        def build_limits_solar_rule(m, z):
            """Build limits for each zone for solar generators"""
            return float(self.candidate_unit_build_limits.loc['SOLAR', z])

        m.X_C_SOLAR = Param(m.Z, rule=build_limits_solar_rule)

        def build_limits_wind_rule(m, z):
            """Build limits for wind generators"""
            return float(self.candidate_unit_build_limits.loc['WIND', z])

        m.X_C_WIND = Param(m.Z, rule=build_limits_wind_rule)

        def build_limits_storage_rule(m, z):
            """
            Build limits for wind generators

            Note: Value from ACIL Allen Fuel and Technology Costs Review spreadsheet
            """
            return float(100)

        m.X_C_STORAGE = Param(m.Z, rule=build_limits_storage_rule)

        # Budget constraint per year
        m.BUDGET = Param(initialize=500e6)

        def marginal_cost_rule(m, g, i):
            """Heat rates for existing and candidate generators"""

            # Marginal costs for existing thermal units
            if g in m.G_E_THERM:
                # Heat rate
                heat_rate = self.existing_units.loc[g, ('PARAMETERS', 'HEAT_RATE')].astype(float)

                # Fuel costs
                try:
                    fuel_cost = self.existing_units.loc[g, ('FUEL_COST', i)].astype(float)

                except KeyError:
                    # Last year for which fuel cost data is available
                    final_year = self.existing_units.loc[:, 'FUEL_COST'].columns[-1]

                    # Set fuel cost for final years to cost in last available year
                    fuel_cost = self.existing_units.loc[g, ('FUEL_COST', final_year)].astype(float)

                # Variable operating and maintenance cost (VOM)
                vom = self.existing_units.loc[g, ('PARAMETERS', 'VOM')].astype(float)

                # Marginal cost
                marginal_cost = vom + (heat_rate * fuel_cost)

            # Marginal costs for candidate thermal units
            elif g in m.G_C_THERM:
                # Heat rate
                heat_rate = self.candidate_units.loc[g, ('PARAMETERS', 'HEAT_RATE')].astype(float)

                # Fuel costs
                try:
                    fuel_cost = self.candidate_units.loc[g, ('FUEL_COST', i)].astype(float)

                except KeyError:
                    # Last year for which fuel cost data is available
                    final_year = self.candidate_units.loc[:, 'FUEL_COST'].columns[-1]

                    # Set fuel cost for final years to cost in last available year
                    fuel_cost = self.candidate_units.loc[g, ('FUEL_COST', final_year)].astype(float)

                # Variable operating and maintenance cost (VOM)
                vom = self.candidate_units.loc[g, ('PARAMETERS', 'VOM')].astype(float)

                # Marginal cost
                marginal_cost = vom + (heat_rate * fuel_cost)

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

        m.C_MC = Param(m.G, m.I, rule=marginal_cost_rule)

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

        m.C_SD = Param(m.G, rule=shutdown_cost_rule)

        def minimum_region_up_reserve_rule(m, r):
            """Minimum upward reserve rule"""

            # Minimum upward reserve for region
            up_reserve = self.minimum_reserve_levels.loc[r, 'MINIMUM_RESERVE_LEVEL'].astype(float)

            return up_reserve

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

        m.SD_RAMP = Param(m.G_E_THERM.union(m.G_C_THERM), rule=shutdown_ramp_rate_rule)

        def capacity_factors_wind_rule(m, b, i, o, t):
            """Wind capacity factors"""

            # Extract capacity factor for given year, hour and cluster
            capacity_factor = self.input_traces.loc[(i, o), ('WIND', b, t)].astype(float)

            # If capacity factor < 0, set = 0
            if capacity_factor < 0:
                capacity_factor = float(0)

            # Check that capacity factor is greater than 0
            assert capacity_factor >= 0, f'Wind capacity factor less than 0: {capacity_factor}'

            return capacity_factor

        m.Q_WIND = Param(m.B, m.I, m.O, m.T, rule=capacity_factors_wind_rule)

        def capacity_factors_solar_rule(m, z, i, o, t):
            """Solar capacity factors"""

            # Extract capacity factor for given year, hour and cluster
            capacity_factor = self.input_traces.loc[(i, o), ('SOLAR', z, t)].astype(float)

            # If capacity factor < 0, set = 0
            if capacity_factor < 0:
                capacity_factor = float(0)

            # Check that capacity factor is greater than 0
            assert capacity_factor >= 0, f'Solar capacity factor less than 0: {capacity_factor}'

            return capacity_factor

        m.Q_SOLAR = Param(m.Z, m.I, m.O, m.T, rule=capacity_factors_solar_rule)

        def zone_demand_rule(m, z, i, o, t):
            """Solar capacity factors"""

            # Extract capacity factor for given year, hour and cluster
            demand = self.input_traces.loc[(i, o), ('DEMAND', z, t)].astype(float)

            # If capacity factor < 0, set = 0
            if demand < 0:
                demand = float(0)

            # Check that capacity factor is greater than 0
            assert demand >= 0, f'Zone demand less than 0: {demand}'

            return demand

        m.D = Param(m.Z, m.I, m.O, m.T, rule=zone_demand_rule)

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
