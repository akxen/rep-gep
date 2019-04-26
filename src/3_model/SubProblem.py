# Import required packages
from pyomo.environ import *

# Class to instantiate UC model, with methods to solve the model, update
# parameters, and fix variables.

# Class to run the UC model using a rolling window approach. Accepts as
# arguments data dictionaries describing demand traces, solar traces, wind
# traces, and hydro output.


class UnitCommitment:
    def __init__(self):
        # Initialise unit commitment model
        self.model = self._initialise_model()

        # Specify solver options

    def _construct_model(self):
        """
        Initialise unit commitment model
        """
        # Create concrete model
        m = ConcreteModel()

        # Sets
        # ----
        # Existing thermal units
        m.G_E_THERM = Set()

        # Candidate thermal units
        m.G_C_THERM = Set()

        # Existing wind units
        m.G_E_WIND = Set()

        # Candidate wind units
        m.G_C_WIND = Set()

        # Existing

        # Existing storage units
        m.S_E = Set()

        # Candidate storage units
        m.S_C = Set()

        # Investment periods
        m.I = Set()

        # Operating scenarios
        m.O = Set()

        # Hours within operating scenario o
        m.T = Set()

        #






        #

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
