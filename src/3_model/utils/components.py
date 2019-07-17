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

        # Existing storage units TODO: Not considering incumbent units for now. Could perhaps include later.
        m.G_E_STORAGE = Set(initialize=[])

        # Candidate storage units
        m.G_C_STORAGE = Set(initialize=self.data.candidate_storage_units)

        # Existing + candidate storage units
        m.G_STORAGE = m.G_E_STORAGE.union(m.G_C_STORAGE)

        # Slow start thermal generators (existing and candidate)
        m.G_THERM_SLOW = Set(initialize=self.data.slow_start_thermal_generator_ids)

        # Quick start thermal generators (existing and candidate)
        m.G_THERM_QUICK = Set(initialize=self.data.quick_start_thermal_generator_ids)

        # All existing generators
        m.G_E = m.G_E_THERM.union(m.G_E_WIND).union(m.G_E_SOLAR).union(m.G_E_HYDRO)

        # All candidate generators
        m.G_C = m.G_C_THERM.union(m.G_C_WIND).union(m.G_C_SOLAR).union(m.G_C_STORAGE)

        # All generators
        m.G = m.G_E.union(m.G_C)

        # All years in model horizon
        # m.Y = RangeSet(2016, 2050)
        m.Y = RangeSet(2016, 2026)

        # Operating scenarios for each year
        # m.S = RangeSet(1, 10)
        m.S = RangeSet(1, 10)

        # Operating scenario hour
        m.T = RangeSet(1, 24, ordered=True)

        # Build limit technology types
        m.BUILD_LIMIT_TECHNOLOGIES = Set(initialize=self.data.candidate_unit_build_limits.index)

        return m

    def define_parameters(self, m):
        """Define common parameters"""

        def existing_units_max_output_rule(_m, g):
            """Max power output for existing units"""

            return float(self.data.existing_units_dict[('PARAMETERS', 'REG_CAP')][g])

        # Max output for existing units
        m.P_MAX = Param(m.G_E, rule=existing_units_max_output_rule)

        def retirement_indicator_rule(_m, g, y):
            """Indicates if unit is retired in a given year. 1 - retired, 0 - still in service"""
            # TODO: Use announced retirement data from NTNDP. Assume no closures for now.
            return float(0)

        # Retirement indicator: 1 - unit is retired, 0 - unit is still in service
        m.F = Param(m.G_E, m.Y, rule=retirement_indicator_rule)

        return m
