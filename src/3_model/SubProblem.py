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
        self.candidate_units = pd.read_csv(os.path.join(data_dir, 'candidate_units.csv'), header=[0, 1], index_col=0)

        # Existing units
        self.existing_units = pd.read_csv(os.path.join(data_dir, 'existing_units.csv'), header=[0, 1], index_col=0)

        # Battery build costs
        self.battery_build_costs = pd.read_csv(os.path.join(data_dir, 'battery_build_costs.csv'), header=0, index_col=0)

        # Battery properties
        self.battery_properties = pd.read_csv(os.path.join(data_dir, 'battery_properties.csv'), header=0, index_col=0)

        # Battery build costs
        self.candidate_unit_build_limits = pd.read_csv(os.path.join(data_dir, 'candidate_unit_build_limits.csv'),
                                                       header=0, index_col=0)

        # Initialise unit commitment model
        self.model = None

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
        # Existing thermal units
        m.G_E_THERM = Set(initialize=self._get_existing_thermal_unit_ids())

        # Candidate thermal units
        m.G_C_THERM = Set(initialize=self._get_candidate_thermal_unit_ids())

        # Existing wind units
        m.G_E_WIND = Set(initialize=self._get_existing_wind_unit_ids())

        # Candidate wind units
        m.G_C_WIND = Set(initialize=self._get_candidate_wind_unit_ids())

        # Existing storage units
        # m.S_E = Set()

        # Candidate storage units
        m.S_C = Set(initialize=self._get_candidate_storage_units())

        # Investment periods
        m.I = RangeSet(2016, 2049, ordered=True)

        # Operating scenarios
        m.O = RangeSet(0, 9, ordered=True)

        # Hours within operating scenario o
        m.T = RangeSet(0, 23, ordered=True)

        # Parameters
        # ----------

        # Update model attribute
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





