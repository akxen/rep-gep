"""Generate emissions intensity target based on on BAU results"""

import pandas as pd
import matplotlib.pyplot as plt


from analysis import AnalyseResults


def get_year_emission_intensity_target(initial_emissions_intensity, half_life, year, start_year):
    """Get half-life emissions intensity target for each year in model horizon"""

    # Re-index such that first year in model horizon is t = 0
    t = year - start_year

    exponent = (-t / half_life)

    return initial_emissions_intensity * (2 ** exponent)


def get_emissions_intensity_target(half_life):
    """Get sequence of yearly emissions intensity targets"""

    # Object used to analyse results
    analysis = AnalyseResults()

    # Get emissions intensities for each year of model horizon - BAU case
    df_bau = analysis.get_year_system_emissions_intensities('primal_bau_results.pickle')
    df_bau = df_bau.rename(columns={'emissions_intensity': 'bau_emissions_intensity'})

    # First and last years of model horizon
    start, end = df_bau.index[[0, -1]]

    # Initial emissions intensity
    E_0 = df_bau.loc[start, 'bau_emissions_intensity']

    # Emissions intensity target sequence
    target_sequence = {y: get_year_emission_intensity_target(E_0, half_life, y, start) for y in range(start, end + 1)}

    # Convert to DataFrame
    df_sequence = pd.Series(target_sequence).rename_axis('year').to_frame('emissions_intensity_target')

    # Combine with bau emissions intensities
    df_c = pd.concat([df_bau, df_sequence], axis=1)

    return df_c


def get_first_year_average_real_bau_price():
    """Get average price in first year of model horizon"""

    # Object used to analyse results
    analysis = AnalyseResults()

    # Get average price in first year of model horizon (real price)
    prices = analysis.get_year_average_price('primal_bau_results.pickle')

    return prices.iloc[0]['average_price_real']


if __name__ == '__main__':
    # Emissions intensity target - assumes system emissions intensity will halve every 25 years
    df_emissions_target = get_emissions_intensity_target(half_life=25)

    # Average real BAU price in first year of model horizon
    first_year_average_real_bau_price = get_first_year_average_real_bau_price()
