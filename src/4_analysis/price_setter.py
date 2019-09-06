"""Identify price setting generators in each dispatch interval"""

import os

import pandas as pd

from analysis import AnalyseResults


def get_existing_unit_marginal_costs(baselines, permit_prices):
    """Existing unit marginal costs"""

    # Container for generator marginal costs
    generator_costs = {}

    for g in analysis.data.existing_units.index:
        for y in baselines.keys():

            # If years in model exceed last year of fuel cost horizon, use fuel cost in last available year
            if y > analysis.data.existing_units.loc[g, 'FUEL_COST'].index[-1]:
                y_fc = analysis.data.existing_units.loc[g, 'FUEL_COST'].index[-1]
            else:
                y_fc = y

            fuel_cost = analysis.data.existing_units.loc[g, ('FUEL_COST', y_fc)]

            heat_rate = analysis.data.existing_units.loc[g, ('PARAMETERS', 'HEAT_RATE')]

            vom = analysis.data.existing_units.loc[g, ('PARAMETERS', 'VOM')]

            emissions_rate = analysis.data.existing_units.loc[g, ('PARAMETERS', 'EMISSIONS')]

            permit_price = permit_prices[y]

            baseline = baselines[y]

            if analysis.data.existing_units.loc[g, ('PARAMETERS', 'SCHEDULE_TYPE')] == 'SCHEDULED':
                emissions_cost = (emissions_rate - baseline) * permit_price
            else:
                emissions_cost = 0

            marginal_cost = (fuel_cost * heat_rate) + vom + emissions_cost

            generator_costs[(g, y)] = marginal_cost

    return generator_costs


def get_candidate_unit_marginal_costs(baselines, permit_prices):
    """Candidate unit marginal costs"""

    # Container for generator marginal costs
    generator_costs = {}

    for g in analysis.data.candidate_units.index:
        for y in baselines.keys():

            # If years in model exceed last year of fuel cost horizon, use fuel cost in last available year
            if y > analysis.data.candidate_units.loc[g, 'FUEL_COST'].index[-1]:
                y_fc = analysis.data.candidate_units.loc[g, 'FUEL_COST'].index[-1]
            else:
                y_fc = y

            fuel_cost = analysis.data.candidate_units.loc[g, ('FUEL_COST', y_fc)]

            heat_rate = analysis.data.candidate_units.loc[g, ('PARAMETERS', 'HEAT_RATE')]

            vom = analysis.data.candidate_units.loc[g, ('PARAMETERS', 'VOM')]

            emissions_rate = analysis.data.candidate_units.loc[g, ('PARAMETERS', 'EMISSIONS')]

            permit_price = permit_prices[y]

            baseline = baselines[y]

            emissions_cost = (emissions_rate - baseline) * permit_price

            marginal_cost = (fuel_cost * heat_rate) + vom + emissions_cost

            generator_costs[(g, y)] = marginal_cost

    return generator_costs


# def get_marginal_costs(filename):
#     """Get all marginal costs"""
#
#     # Load results
#     results = analysis.load_results(filename)
#
#     # Baselines applying for a given year
#     baselines, permit_prices = results['baseline'], results['permit_price']
#
#     # Existing generators
#     existing_cost = get_existing_unit_marginal_costs(baselines, permit_prices)
#
#     # Candidate generators
#     candidate_cost = get_candidate_unit_marginal_costs(baselines, permit_prices)
#
#     # Combine into single dictionary
#     marginal_costs = {**existing_cost, **candidate_cost}
#
#     return marginal_costs


def get_operating_cost(filename):
    """Get fuel + VOM cost for each unit and year over model horizon"""

    # Load results
    results = analysis.load_results(filename)

    return results['C_MC']


def get_scheme_cost(filename):
    """Get penalty / credit received per MWh by each generator"""

    # Load results
    results = analysis.load_results(filename)

    # Initialise container for emissions costs
    emissions_cost = {}

    for g, y in results['C_MC'].keys():

        if g in analysis.data.existing_units.index:
            emissions_rate = analysis.data.existing_units.loc[g, ('PARAMETERS', 'EMISSIONS')]

        elif g in analysis.data.candidate_units.index:
            emissions_rate = analysis.data.candidate_units.loc[g, ('PARAMETERS', 'EMISSIONS')]

        elif g in analysis.data.candidate_storage_units:
            emissions_rate = float(0)

        else:
            raise Exception(f'Unexpected generator: {g}')

        # Compute net cost faced by generators under the scheme
        emissions_cost[(g, y)] = (emissions_rate - results['baseline'][y]) * results['permit_price'][y]

    return emissions_cost


def get_net_marginal_costs(filename):
    """Compute short-run marginal costs for each generator"""

    # Operating cost per MWh
    operating_costs = get_operating_cost(filename)

    # Net scheme penalty / credit per MWh
    scheme_cost = get_scheme_cost(filename)

    # Check keys are the same
    assert set(operating_costs.keys()) == set(scheme_cost.keys()), 'Keys are not the same'

    # Net marginal cost
    net_marginal_cost = {k: operating_costs[k] + scheme_cost[k] for k in operating_costs.keys()}

    # Convert to pandas series
    df_net_cost = pd.Series(net_marginal_cost)

    return df_net_cost


if __name__ == '__main__':
    # Path where results can be found
    results_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, '3_model', 'linear', 'output', 'local')

    # Object used to analyse results
    analysis = AnalyseResults(results_directory)

    # Results file
    results_filename = 'primal_bau_case.pickle'

    # Average price
    df_p = analysis.parse_prices(results_filename)

    # Real price adjusted for discount factor
    df_p['real_price'] = df_p['average_price'] * df_p['discount_factor']

    # Net short-run marginal cost
    df_c = get_net_marginal_costs(results_filename)

    def get_price_setter(row):
        """Get price setting generator, and price difference for each interval and zone"""

        year = row.name[0]

        duid = df_c.loc[(slice(None), year)].subtract(row['real_price']).abs().idxmin()

        difference = df_c.loc[(slice(None), year)].subtract(row['real_price']).abs().min()

        if difference > 9000:
            duid = 'load shedding'

        return pd.Series(data={'duid': duid, 'difference': difference})

    df_d = df_p.apply(get_price_setter, axis=1)