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

# Discrete investment decisions for candidate thermal generators
m.d = Var(m.G_C_THERM, m.G_C_THERM_SIZE_OPTIONS, m.I, within=Binary)


def build_limits_rule(m, technology, z):
    """Build limits for each technology type by zone"""

    return self.candidate_unit_build_limits.loc[technology, z].astype(float)


# Build limits for each technology and zone
m.BUILD_LIMITS = Param(m.BUILD_LIMIT_TECHNOLOGIES, m.Z, rule=build_limits_rule)


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