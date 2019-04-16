# Import required packages

# Class used to define master problem. Note: Benders cuts must be iteratively
# added, so problem should be defined in such a way that allows this to occur.

# Class used to run the master problem.


class Master:
    def __init__(self):
        # Initialise master problem parameters
        self.model = self._construct_model()

        # Initialise solver parameters

    def _construct_model(self):
        """
        Initialise master problem
        """
        pass

    def add_cuts(self, subproblem_results):
        """
        Add Benders cuts using information obtained from each sub-problem
        solution
        """
        pass

    def solve_model(self):
        """
        Solve the master problem
        """
        pass

    def get_result_summary(self):
        """
        Extract results for selected variables and expressions.

        Returns
        -------
        result_summary : dict
            Summary of results for selected variables and expressions
        """
        pass
