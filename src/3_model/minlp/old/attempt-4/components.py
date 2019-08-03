"""Model components common to both subproblem and master problem"""

from collections import OrderedDict

from pyomo.environ import *

from data import ModelData


class CommonComponents:
    # Model data
    data = ModelData()

    def __init__(self):
        pass

    def define_sets(self, m):
        """Define sets to be used in model"""

        # NEM regions
        m.R = Set(initialize=self.data.nem_regions)

        # NEM zones
        m.Z = Set(initialize=self.data.nem_zones)

        # Links between NEM zones
        m.L = Set(initialize=self.data.network_links)

        # Interconnectors for which flow limits are defined
        m.L_I = Set(initialize=list(self.data.powerflow_limits.keys()))

        # NEM wind bubbles
        m.B = Set(initialize=self.data.wind_bubbles)

        # Existing thermal units
        m.G_E_THERM = Set(initialize=self.data.existing_thermal_unit_ids)

        # Candidate thermal units
        m.G_C_THERM = Set(initialize=self.data.candidate_thermal_unit_ids)

        # All existing and candidate thermal generators
        m.G_THERM = Set(initialize=m.G_E_THERM.union(m.G_C_THERM))

        # Index for candidate thermal unit size options
        m.G_C_THERM_SIZE_OPTIONS = RangeSet(0, 3, ordered=True)

        # Existing wind units
        m.G_E_WIND = Set(initialize=self.data.existing_wind_unit_ids)

        # Candidate wind units
        m.G_C_WIND = Set(initialize=self.data.candidate_wind_unit_ids)

        # Existing solar units
        m.G_E_SOLAR = Set(initialize=self.data.existing_solar_unit_ids)

        # Candidate solar units
        m.G_C_SOLAR = Set(initialize=self.data.candidate_solar_unit_ids)

        # Available technologies
        m.G_C_SOLAR_TECHNOLOGIES = Set(initialize=list(set(y.split('-')[-1] for y in m.G_C_SOLAR)))

        # Existing hydro units
        m.G_E_HYDRO = Set(initialize=self.data.existing_hydro_unit_ids)

        # Candidate storage units
        m.G_C_STORAGE = Set(initialize=self.data.candidate_storage_units)

        # Slow start thermal generators (existing and candidate)
        m.G_THERM_SLOW = Set(initialize=self.data.slow_start_thermal_generator_ids)

        # Quick start thermal generators (existing and candidate)
        m.G_THERM_QUICK = Set(initialize=self.data.quick_start_thermal_generator_ids)

        # All existing generators
        m.G_E = m.G_E_THERM.union(m.G_E_WIND).union(m.G_E_SOLAR).union(m.G_E_HYDRO)

        # All candidate generators
        m.G_C = m.G_C_THERM.union(m.G_C_WIND).union(m.G_C_SOLAR)

        # All generators
        m.G = m.G_E.union(m.G_C)

        # All years in model horizon
        m.Y = RangeSet(2016, 2017)

        # Operating scenarios for each year
        m.O = RangeSet(0, 9)

        # Operating scenario hour
        m.T = RangeSet(0, 23, ordered=True)

        # Build limit technology types
        m.BUILD_LIMIT_TECHNOLOGIES = Set(initialize=self.data.candidate_unit_build_limits.index)

        return m

    @staticmethod
    def define_parameters(m):
        """Define model parameters - these are common to all blocks"""

        def thermal_unit_discrete_size_rule(m, g, n):
            """Possible discrete sizes for candidate thermal units"""

            # Discrete sizes available for candidate thermal unit investment
            options = {0: 0, 1: 100, 2: 200, 3: 400}

            return float(options[n])

        # Candidate thermal unit size options
        m.X_CT = Param(m.G_C_THERM, m.G_C_THERM_SIZE_OPTIONS, rule=thermal_unit_discrete_size_rule)

        return m

    @staticmethod
    def define_variables(m):
        """Define model variables common to all sub-problems"""

        # Capacity of candidate units (defined for all years in model horizon)
        m.x_c = Var(m.G_C.union(m.G_C_STORAGE), m.Y, within=NonNegativeReals, initialize=0)

        # Binary variable used to determine size of candidate thermal units
        m.d = Var(m.G_C_THERM, m.Y, m.G_C_THERM_SIZE_OPTIONS, within=NonNegativeReals, initialize=0)

        # Startup indicator
        m.v = Var(m.G_THERM, m.T, within=Binary)

        # Shutdown indicator
        m.w = Var(m.G_THERM, m.T, within=Binary)

        return m

    @staticmethod
    def define_expressions(m):
        """Define expressions common to all sub-problems"""

        return m
