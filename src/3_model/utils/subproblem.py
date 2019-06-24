"""Classes used to construct investment planning subproblem"""

import time
from collections import OrderedDict

from pyomo.environ import *
from pyomo.util.infeasible import log_infeasible_constraints

from data import ModelData
from components import CommonComponents


class Subproblem:
    # Pre-processed model data
    data = ModelData()

    def __init__(self):
        # Solver options
        self.keepfiles = False
        self.solver_options = {'Method': 1}  # 'MIPGap': 0.0005
        self.opt = SolverFactory('gurobi', solver_io='lp')

    def construct_model(self):
        """Construct subproblem model components"""

        # Used to define sets and parameters common to both master and subproblem
        common_components = CommonComponents()

        # Initialise base model object
        m = ConcreteModel()

        # Define sets - common to both master and subproblem
        m = common_components.define_sets(m)

        # Define parameters - common to both master and subproblem
        m = common_components.define_parameters(m)

        # Define variables - common to both master and subproblem
        m = common_components.define_variables(m)

        # Define expressions
        m = self.define_expressions(m)

        # Define blocks
        m = self.define_blocks(m)

        # Define constraints
        m = self.define_constraints(m)

        # Define objective
        m = self.define_objective(m)

        return m

    @staticmethod
    def define_expressions(m):
        """Define subproblem expressions"""

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
        m.X_C = Expression(m.G_C.union(m.G_C_STORAGE), m.I, rule=capacity_sizing_rule)

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

        def fom_cost_rule(_m):
            """Fixed operating and maintenance cost for candidate generators over model horizon"""

            return sum(m.C_FOM[g] * m.X_C[g, i] for g in m.G_C.union(m.G_C_SOLAR) for i in m.I)

        # Fixed operation and maintenance cost - absolute cost [$]
        m.C_FOM_TOTAL = Expression(rule=fom_cost_rule)

        def investment_cost_rule(_m):
            """Cost to invest in candidate technologies"""

            return sum(m.C_INV[g, i] * m.X_C[g, i] for g in m.G_C.union(m.G_C_STORAGE) for i in m.I)

        # Investment cost (fixed in subproblem) - absolute cost [$]
        m.C_INV_TOTAL = Expression(rule=investment_cost_rule)

        # Penalty imposed for violating emissions constraint
        m.C_EMISSIONS_VIOLATION = Expression(expr=m.emissions_target_exceeded * m.EMISSIONS_EXCEEDED_PENALTY)

        # Penalty imposed for violating revenue constraint
        m.C_REVENUE_VIOLATION = Expression(expr=m.revenue_shortfall * m.REVENUE_SHORTFALL_PENALTY)

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
                    # capacity_factor = float(self.data.input_traces.loc[(i, o), ('WIND', b, t)])
                    capacity_factor = float(self.data.input_traces_dict[('WIND', b, t)][(i, o)])

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

                    # capacity_factor = float(self.data.input_traces.loc[(i, o), ('SOLAR', col, t)])
                    capacity_factor = float(self.data.input_traces_dict[('SOLAR', col, t)][(i, o)])

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
                    # demand = float(self.data.input_traces.loc[(i, o), ('DEMAND', z, t)])
                    demand = float(self.data.input_traces_dict[('DEMAND', z, t)][(i, o)])

                    return demand

                # Demand in each NEM zone
                start = time.time()
                s.DEMAND = Param(m.Z, m.T, rule=demand_rule)
                print(f'Constructed DEMAND in: {time.time() - start}s')

                def scenario_duration_rule(_s):
                    """Normalised duration for each operation scenario"""

                    return float(self.data.input_traces.loc[(i, o), ('K_MEANS', 'METRIC', 'NORMALISED_DURATION')])

                # Normalised duration for each operating scenario
                start = time.time()
                s.NORMALISED_DURATION = Param(rule=scenario_duration_rule)
                print(f'Constructed NORMALISED_DURATION in: {time.time() - start}s')

                def historic_hydro_output_rule(_s, g, t):
                    """Historic output for existing hydro generators"""
                    # output = float(self.data.input_traces.loc[(i, o), ('HYDRO', g, t)])
                    output = float(self.data.input_traces_dict[('HYDRO', g, t)][(i, o)])

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
                    return sum(
                        s.ENERGY[g, t] * m.EMISSIONS_RATE[g] for g in m.G_E_THERM.union(m.G_C_THERM) for t in m.T)

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
                    region_zone_map = self.data.nem_region_zone_map_dict

                    # Mapping describing the zone to which each generator is assigned
                    # gen_zone_map = self._get_generator_zone_map()
                    gen_zone_map = self.data.generator_zone_map

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
                        bubble = self.data.existing_wind_bubble_map_dict['BUBBLE'][g]

                    elif g in m.G_C_WIND:

                        # If a candidate generator
                        # bubble = self.candidate_units.loc[g, ('PARAMETERS', 'WIND_BUBBLE')]
                        bubble = self.data.candidate_units_dict[('PARAMETERS', 'WIND_BUBBLE')][g]

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
                        zone = self.data.existing_units_dict[('PARAMETERS', 'NEM_ZONE')][g]

                        # Assume existing arrays are single-axis tracking arrays
                        technology = 'SAT'

                    elif g in m.G_C_SOLAR:
                        # If a candidate generator
                        # zone = self.candidate_units.loc[g, ('PARAMETERS', 'ZONE')]
                        zone = self.data.candidate_units_dict[('PARAMETERS', 'ZONE')][g]

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
                    existing_units = [gen for gen, zone in
                                      self.data.existing_units_dict[('PARAMETERS', 'NEM_ZONE')].items()
                                      if zone == z]

                    # Candidate units within zone
                    # candidate_units = self.candidate_units[
                    #     self.candidate_units[('PARAMETERS', 'ZONE')] == z].index.tolist()
                    candidate_units = [gen for gen, zone in
                                       self.data.candidate_units_dict[('PARAMETERS', 'ZONE')].items()
                                       if zone == z]

                    # All generators within a given zone
                    generators = existing_units + candidate_units

                    # Storage units within a given zone
                    # storage_units = (self.battery_properties.loc[self.battery_properties['NEM_ZONE'] == z, 'NEM_ZONE']
                    #                  .index.tolist())
                    storage_units = [gen for gen, zone in self.data.battery_properties_dict['NEM_ZONE'].items() if
                                     zone == z]

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
    def define_constraints(m):
        """Define subproblem constraints"""

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

        def scheme_revenue_constraint_rule(m):
            """Scheme must be revenue neutral over model horizon"""

            return (sum(m.SCENARIO[i, o].TOTAL_REVENUE for i in m.I for o in m.O) + m.revenue_shortfall
                    >= m.REVENUE_TARGET)

        # Revenue constraint - must break-even over model horizon
        m.REVENUE_CONSTRAINT = Constraint(rule=scheme_revenue_constraint_rule)

        def scheme_emissions_constraint_rule(m):
            """Emissions limit over model horizon"""

            return (sum(m.SCENARIO[i, o].TOTAL_EMISSIONS for i in m.I for o in m.O)
                    <= m.EMISSIONS_TARGET + m.emissions_target_exceeded)

        # Emissions constraint - must be less than some target, else penalty imposed for each unit above target
        m.EMISSIONS_CONSTRAINT = Constraint(rule=scheme_emissions_constraint_rule)

        def fixed_on_state_rule(m, g, i, t):
            """Fixed on state for a given generator"""

        return m

    @staticmethod
    def define_objective(m):
        """Define subproblem objective"""

        # Minimise total operating cost - include penalty for revenue / emissions constraint violations
        m.OBJECTIVE = Objective(expr=sum(m.SCENARIO[i, o].C_OP_TOTAL for i in m.I for o in m.O)
                                     + m.C_FOM_TOTAL
                                     + m.C_INV_TOTAL
                                     + m.C_EMISSIONS_VIOLATION + m.C_REVENUE_VIOLATION, sense=minimize)

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
    def unfix_discrete_capacity_options(m):
        """Unfix binary variables used in discrete capacity sizing"""
        for g in m.G_C_THERM:
            for i in m.I:
                for n in m.G_C_THERM_SIZE_OPTIONS:
                    m.d[g, i, n].unfix()
        return m

    @staticmethod
    def fix_continuous_capacity_options(m):
        """Fix variables used in continuous capacity sizing options"""
        for g in m.G_C_WIND.union(m.G_C_SOLAR).union(m.G_C_STORAGE):
            for i in m.I:
                m.x_c[g, i].fix(m.FIXED_X_C[g, i].value)

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
    def fix_uc_integer_variables(m, last_value=True):
        """
        Fix unit commitment integer variables - allows duals to be obtained when re-solving

        Parameters
        ----------
        m : pyomo model object
            Object defining optimisation model

        last_value : bool (default=True)
            Fix variable to value obtained in last solution (e.g. from previous model solution). Else fix to value
            to be same as corresponding parameter.
        """

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
    def unfix_uc_integer_variables(m):
        """Unfix unit commitment integer variables"""

        for g in m.G_E_THERM.union(m.G_C_THERM):
            for i in m.I:
                for o in m.O:
                    for t in m.T:
                        m.SCENARIO[i, o].v[g, t].unfix()
                        m.SCENARIO[i, o].w[g, t].unfix()
                        m.SCENARIO[i, o].u[g, t].unfix()
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

    def _solve_for_output(self, m):
        """Solve for output"""

        # Solve model
        self.opt.solve(m, tee=True, options=self.solver_options, keepfiles=self.keepfiles)
        result = None

        # Log infeasible constraints if they exist
        log_infeasible_constraints(m)

        return m, result

    @staticmethod
    def _solve_for_prices(m):
        """Solve for marginal prices"""
        results = None
        return m, results

    @staticmethod
    def _unfix_variables(m):
        """Unfix subproblem variables - return model variable fixed state to what it was prior to first solve"""
        return m

    @staticmethod
    def _save_results(results):
        """
        Save (potentially intermediary) results

        Parameters
        ----------
        results : dict
            Sub-problem results stored in a dictionary
        """

        # Save results

        pass

    def solve_model(self, m):
        """Solve model"""

        # Container for model results
        results = {}

        # Solve for output
        m, output_results = self._solve_for_output(m)

        # Solve for prices
        m, output_prices = self._solve_for_output(m)

        # Store results in dictionary
        results['output'] = output_results
        results['prices'] = output_prices

        # Unfix variables - returning model to state prior to first solve
        m = self._unfix_variables(m)

        return m, results

    def _solve_model(self, m):
        """Solve model"""

        # Solve model
        self.opt.solve(m, tee=True, options=self.solver_options, keepfiles=self.keepfiles)

        # Log infeasible constraints if they exist
        log_infeasible_constraints(m)

        return m


if __name__ == '__main__':
    # Start timer
    start = time.time()

    # Define object used to construct subproblem model
    subproblem = Subproblem()

    # Construct subproblem model
    subproblem_model = subproblem.construct_model()

    # Prepare to read suffix values (dual values)
    subproblem_model.dual = Suffix(direction=Suffix.IMPORT)

    start_iteration = time.time()

    # Update fixed variables obtained from master program
    subproblem_model = subproblem.update_fixed_baseline(subproblem_model, 'test')
    subproblem_model = subproblem.update_fixed_permit_price(subproblem_model, 'test')
    subproblem_model = subproblem.update_fixed_capacity_discrete(subproblem_model, 'test')
    subproblem_model = subproblem.update_fixed_capacity_continuous(subproblem_model, 'test')

    subproblem_model = subproblem.update_fixed_on_state(subproblem_model, 'test')
    subproblem_model = subproblem.update_fixed_startup_state(subproblem_model, 'test')
    subproblem_model = subproblem.update_fixed_shutdown_state(subproblem_model, 'test')

    print(f'Constructed model in: {time.time() - start}s')

    # Fix master problem variables
    start = time.time()
    subproblem_model = subproblem.fix_permit_price(subproblem_model)
    print(f'Fixed permit_price in: {time.time() - start}s')

    start = time.time()
    subproblem_model = subproblem.fix_baseline(subproblem_model)
    print(f'Fixed baseline in: {time.time() - start}s')

    start = time.time()
    subproblem_model = subproblem.fix_discrete_capacity_options(subproblem_model)
    print(f'Fixed discrete_capacity_options in: {time.time() - start}s')

    start = time.time()
    subproblem_model = subproblem.fix_continuous_capacity_options(subproblem_model)
    print(f'Fixed continuous_capacity_options in: {time.time() - start}s')

    start = time.time()
    subproblem_model = subproblem.fix_uc_integer_variables(subproblem_model, last_value=False)
    print(f'Fixed uc_integer_variables in: {time.time() - start}s')

    # Solve model
    start = time.time()
    subproblem_model = subproblem._solve_model(subproblem_model)
    print(f'Solved model in: {time.time() - start}')

    # Get sensitivities for fixed capacity sizing options
    start = time.time()
    subproblem.fix_uc_integer_variables(subproblem_model, last_value=True)
    print(f'Fixed UC integer variables in: {time.time() - start}')

    # Unfix capacity variables
    start = time.time()
    subproblem_model = subproblem.unfix_continuous_capacity_options(subproblem_model)
    print(f'Unfixed continuous capacity option variables in: {time.time() - start}')

    start = time.time()
    subproblem_model = subproblem.unfix_discrete_capacity_options(subproblem_model)
    print(f'Unfixed discrete capacity option variables in: {time.time() - start}')

    start = time.time()
    subproblem_model = subproblem._solve_model(subproblem_model)
    print(f'Solved model (to obtain sensitivities) in: {time.time() - start}')

    # # Fix integer variables
    # subproblem_model = subproblem.fix_uc_integer_variables(subproblem_model)
    #
    # # Re-solve model (obtain dual variables0
    # subproblem.solve_model(subproblem_model)

    # # Get sensitivities for unit capacities and marginal prices
    # # ---------------------------------------------------------
    # # Fix integer variables for unit commitment problem
    # subproblem_model = subproblem.fix_uc_integer_variables(subproblem_model)
    #
    # # Unfix capacity variables
    # subproblem_model = subproblem.unfix_continuous_capacity_options(subproblem_model)
    # subproblem_model = subproblem.unfix_discrete_capacity_options(subproblem_model)
    #
    # # Solve model
    # subproblem.solve_model(subproblem_model)
    #
    # # Get sensitivities for permit price and baseline
    # # -----------------------------------------------
    # # Fix capacity variables
    # subproblem_model = subproblem.fix_continuous_capacity_options(subproblem_model)
    # subproblem_model = subproblem.fix_discrete_capacity_options(subproblem_model)
    #
    # # Fix all unit commitment primal variables
    # subproblem_model = subproblem.fix_uc_continuous_variables(subproblem_model)
    #
    # # Unfix permit price and baseline
    # m_scenario = subproblem.unfix_permit_price(subproblem_model)
    # m_scenario = subproblem.unfix_baseline(subproblem_model)
    #
    # # Solve model
    # subproblem.solve_model(subproblem_model)
    #
    # print(f'Completed iteration in: {time.time() - start_iteration}s')
