"""Plot input traces to visualise data"""

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.data import ModelData


def plot_coal_fuel_costs():
    """Coal fuel costs"""

    # Class containing model data
    data = ModelData()

    # Get coal DUIDs
    mask = data.existing_units['PARAMETERS']['FUEL_TYPE_PRIMARY'].isin(['COAL'])

    # Initialise figures
    fig, ax = plt.subplots()

    # Generators in order of increasing fuel cost (based on cost in last year)
    generator_order = data.existing_units.loc[mask, 'FUEL_COST'].T.iloc[-1].sort_values().index

    # Plot fuel costs for existing units
    data.existing_units.loc[mask, 'FUEL_COST'].T.loc[:, generator_order].plot(ax=ax, cmap='tab20c', alpha=0.9)

    # Add legend
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=7, mode="expand", borderaxespad=0., prop={'size': 6})

    # Add axes labels
    ax.set_ylabel('Fuel cost (\$/GJ)')
    ax.set_xlabel('Year')

    # Format axes ticks
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    # Adjust figure placement and size
    fig.subplots_adjust(top=0.8, left=0.1, right=0.95)
    fig.set_size_inches(6.69291, 6.69291 * 0.8)

    # Save figure
    fig.savefig(os.path.join(os.path.dirname(__file__), 'output', 'figures', 'fuel_cost_coal.pdf'))

    plt.show()


def plot_gas_fuel_costs():
    """Plot fuel costs for candidate gas generators"""

    # Class containing model data
    data = ModelData()

    # Get coal DUIDs
    mask = data.existing_units['PARAMETERS']['FUEL_TYPE_PRIMARY'].isin(['GAS'])

    # Initialise figures
    fig, ax = plt.subplots()

    # Generators in order of increasing fuel cost (based on cost in last year)
    generator_order = data.existing_units.loc[mask, 'FUEL_COST'].T.iloc[-1].sort_values().index

    # Plot fuel costs for existing units
    data.existing_units.loc[mask, 'FUEL_COST'].T.loc[:, generator_order].plot(ax=ax, cmap='tab20c', alpha=0.9)

    # Add legend
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=7, mode="expand", borderaxespad=0., prop={'size': 6})

    # Add axes labels
    ax.set_ylabel('Fuel cost (\$/GJ)')
    ax.set_xlabel('Year')

    # Format axes ticks
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    # Adjust figure placement and size
    fig.subplots_adjust(top=0.7, left=0.1, right=0.95)
    fig.set_size_inches(6.69291, 6.69291 * 0.8)

    # Save figure
    fig.savefig(os.path.join(os.path.dirname(__file__), 'output', 'figures', 'fuel_cost_gas.pdf'))

    plt.show()


def plot_coal_build_costs():
    """Plot build costs for candidate coal generators"""

    # Class containing model data
    data = ModelData()

    # Get coal DUIDs
    mask = data.candidate_units['PARAMETERS']['FUEL_TYPE_PRIMARY'].isin(['COAL'])

    # Initialise figures
    fig, ax = plt.subplots()

    # Plot build costs for candidate units
    data.candidate_units.loc[mask, 'BUILD_COST'].T.plot(ax=ax, cmap='tab20c', alpha=0.9)

    # Add legend
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode="expand", borderaxespad=0., prop={'size': 6})

    # Add axes labels
    ax.set_ylabel('Build cost (\$/kW)')
    ax.set_xlabel('Year')

    # Format axes ticks
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_minor_locator(MultipleLocator(200))

    # Adjust figure placement and size
    fig.subplots_adjust(top=0.85, left=0.1, right=0.95)
    fig.set_size_inches(6.69291, 6.69291 * 0.6)

    # Save figure
    fig.savefig(os.path.join(os.path.dirname(__file__), 'output', 'figures', f'build_costs_coal.pdf'))

    plt.show()


def plot_gas_build_costs():
    """Plotting candidate gas generator build costs"""

    # Class containing model data
    data = ModelData()

    # Get coal DUIDs
    mask = data.candidate_units['PARAMETERS']['FUEL_TYPE_PRIMARY'].isin(['GAS'])

    # Initialise figures
    fig, ax = plt.subplots()

    # Plot build costs for candidate units
    data.candidate_units.loc[mask, 'BUILD_COST'].T.plot(ax=ax, cmap='tab20c', alpha=0.9)

    # Add legend
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode="expand", borderaxespad=0., prop={'size': 6})

    # Add axes labels
    ax.set_ylabel('Build cost (\$/kW)')
    ax.set_xlabel('Year')

    # Format axes ticks
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_minor_locator(MultipleLocator(200))

    # Adjust figure placement and size
    fig.subplots_adjust(top=0.73, left=0.1, right=0.95)
    fig.set_size_inches(6.69291, 6.69291 * 0.6)

    # Save figure
    fig.savefig(os.path.join(os.path.dirname(__file__), 'output', 'figures', f'build_costs_gas.pdf'))

    plt.show()


def plot_solar_build_costs():
    """Plot solar build costs"""

    # Class containing model data
    data = ModelData()

    # Get coal DUIDs
    mask = data.candidate_units['PARAMETERS']['FUEL_TYPE_PRIMARY'].isin(['SOLAR'])

    # Initialise figures
    fig, ax = plt.subplots()

    # Plot build costs for candidate units
    data.candidate_units.loc[mask, 'BUILD_COST'].T.plot(ax=ax, cmap='tab20c', alpha=0.9)

    # Add legend
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode="expand", borderaxespad=0., prop={'size': 6})

    # Add axes labels
    ax.set_ylabel('Build cost (\$/kW)')
    ax.set_xlabel('Year')

    # Format axes ticks
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_minor_locator(MultipleLocator(200))

    # Adjust figure placement and size
    fig.subplots_adjust(top=0.75, left=0.1, right=0.95)
    fig.set_size_inches(6.69291, 6.69291 * 0.6)

    # Save figure
    fig.savefig(os.path.join(os.path.dirname(__file__), 'output', 'figures', f'build_costs_solar.pdf'))

    plt.show()


def plot_wind_build_cost():
    """Plot build costs for candidate wind generators"""

    # Class containing model data
    data = ModelData()

    # Get coal DUIDs
    mask = data.candidate_units['PARAMETERS']['FUEL_TYPE_PRIMARY'].isin(['WIND'])

    # Initialise figures
    fig, ax = plt.subplots()

    # Plot build costs for candidate units
    data.candidate_units.loc[mask, 'BUILD_COST'].T.plot(ax=ax, cmap='tab20c', alpha=0.9)

    # Add legend
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode="expand", borderaxespad=0., prop={'size': 6})

    # Add axes labels
    ax.set_ylabel('Build cost (\$/kW)')
    ax.set_xlabel('Year')

    # Format axes ticks
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(50))

    # Adjust figure placement and size
    fig.subplots_adjust(top=0.85, left=0.1, right=0.95)
    fig.set_size_inches(6.69291, 6.69291 * 0.6)

    # Save figure
    fig.savefig(os.path.join(os.path.dirname(__file__), 'output', 'figures', f'build_costs_wind.pdf'))

    plt.show()


def plot_demand_profiles(nem_zone='ADE'):
    """Plot demand profiles"""

    # Class containing model data
    data = ModelData()

    df = pd.read_hdf(os.path.join(os.path.dirname(__file__), os.path.pardir, '2_input_traces', 'output', 'dataset.h5'))

    df_d = df.loc[:, [('DEMAND', 'ADE')]]

    # Data for 20202
    df_d = df_d.sort_index()
    df_d = df_d.loc[df_d.index.year == 2020, :]

    # Day of year
    df_d['day_of_year'] = df_d.index.dayofyear.values

    # Hour in a given day
    df_d['hour'] = df_d.index.hour.values

    # Adjust last hour in each day - set as 24th hour
    df_d['hour'] = df_d['hour'].replace(0, 24)

    plt.clf()
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    df_d.pivot(index='day_of_year', columns='hour', values=('DEMAND', nem_zone)).T.plot(ax=ax1, legend=False, alpha=0.4,
                                                                                        cmap='viridis')

    df_d.pivot(index='day_of_year', columns='hour', values=('DEMAND', nem_zone)).T.plot(ax=ax2, legend=False, alpha=0.1,
                                                                                        cmap='viridis')

    # Plot traces obtained from k-means clustering
    data.input_traces.loc[2020, ('DEMAND', nem_zone)].T.plot(ax=ax2, color='r', legend=False, alpha=0.8)

    ax1.set_ylabel('ADE Demand (MW)')
    ax2.set_ylabel('ADE Demand (MW)')
    ax2.set_xlabel('Hour')

    # Format axes ticks
    ax2.xaxis.set_major_locator(MultipleLocator(5))
    ax2.xaxis.set_minor_locator(MultipleLocator(1))

    ax1.yaxis.set_major_locator(MultipleLocator(500))
    ax1.yaxis.set_minor_locator(MultipleLocator(100))
    ax2.yaxis.set_major_locator(MultipleLocator(500))
    ax2.yaxis.set_minor_locator(MultipleLocator(100))

    # Adjust figure placement and size
    fig.subplots_adjust(top=0.95, bottom=0.08, left=0.1, right=0.95)
    fig.set_size_inches(6.69291, 6.69291)

    # Save figure
    fig.savefig(os.path.join(os.path.dirname(__file__), 'output', 'figures', f'demand_profiles_{nem_zone}.pdf'))

    plt.show()


def plot_wind_capacity_factors(wind_bubble='YPS'):
    """Plotting wind capacity factors for a given wind bubble"""

    # Class containing model data
    data = ModelData()

    df = pd.read_hdf(os.path.join(os.path.dirname(__file__), os.path.pardir, '2_input_traces', 'output', 'dataset.h5'))

    df_d = df.loc[:, [('WIND', wind_bubble)]]

    # Data for 20202
    df_d = df_d.sort_index()
    df_d = df_d.loc[df_d.index.year == 2020, :]

    # Day of year
    df_d['day_of_year'] = df_d.index.dayofyear.values

    # Hour in a given day
    df_d['hour'] = df_d.index.hour.values

    # Adjust last hour in each day - set as 24th hour
    df_d['hour'] = df_d['hour'].replace(0, 24)

    plt.clf()
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    df_d.pivot(index='day_of_year', columns='hour', values=('WIND', wind_bubble)).T.plot(ax=ax1, legend=False,
                                                                                         alpha=0.4, cmap='viridis')

    df_d.pivot(index='day_of_year', columns='hour', values=('WIND', wind_bubble)).T.plot(ax=ax2, legend=False,
                                                                                         alpha=0.1, cmap='viridis')

    # Plot traces obtained from k-means clustering
    data.input_traces.loc[2020, ('WIND', wind_bubble)].T.plot(ax=ax2, color='r', legend=False, alpha=0.8)

    ax1.set_ylabel(f'{wind_bubble} capacity factor [-]')
    ax2.set_ylabel(f'{wind_bubble}  capacity factor [-]')
    ax2.set_xlabel('Hour')

    # Format axes ticks
    ax2.xaxis.set_major_locator(MultipleLocator(5))
    ax2.xaxis.set_minor_locator(MultipleLocator(1))

    ax1.yaxis.set_major_locator(MultipleLocator(0.2))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax2.yaxis.set_major_locator(MultipleLocator(0.2))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.05))

    # Adjust figure placement and size
    fig.subplots_adjust(top=0.95, bottom=0.08, left=0.1, right=0.95)
    fig.set_size_inches(6.69291, 6.69291)

    # Save figure
    fig.savefig(os.path.join(os.path.dirname(__file__), 'output', 'figures', f'wind_profiles_{wind_bubble}.pdf'))

    plt.show()


def plot_solar_capacity_factors(nem_zone='ADE', technology='DAT'):
    """Plot solar capacity factors"""

    # Construct solar technology ID based on NEM zone and technology type
    tech_id = f'{nem_zone}|{technology}'

    # Class containing model data
    data = ModelData()

    df = pd.read_hdf(os.path.join(os.path.dirname(__file__), os.path.pardir, '2_input_traces', 'output', 'dataset.h5'))

    df_d = df.loc[:, [('SOLAR', tech_id)]]

    # Data for 2020
    df_d = df_d.sort_index()
    df_d = df_d.loc[df_d.index.year == 2020, :]

    # Day of year
    df_d['day_of_year'] = df_d.index.dayofyear.values

    # Hour in a given day
    df_d['hour'] = df_d.index.hour.values

    # Adjust last hour in each day - set as 24th hour
    df_d['hour'] = df_d['hour'].replace(0, 24)

    plt.clf()
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    df_d.pivot(index='day_of_year', columns='hour', values=('SOLAR', tech_id)).T.plot(ax=ax1, legend=False,
                                                                                      alpha=0.4, cmap='viridis')

    df_d.pivot(index='day_of_year', columns='hour', values=('SOLAR', tech_id)).T.plot(ax=ax2, legend=False,
                                                                                      alpha=0.1, cmap='viridis')

    # Plot traces obtained from k-means clustering
    data.input_traces.loc[2020, ('SOLAR', tech_id)].T.plot(ax=ax2, color='r', legend=False, alpha=0.8)

    ax1.set_ylabel(f'{tech_id} capacity factor [-]')
    ax2.set_ylabel(f'{tech_id}  capacity factor [-]')
    ax2.set_xlabel('Hour')

    # Format axes ticks
    ax2.xaxis.set_major_locator(MultipleLocator(5))
    ax2.xaxis.set_minor_locator(MultipleLocator(1))

    ax1.yaxis.set_major_locator(MultipleLocator(0.2))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax2.yaxis.set_major_locator(MultipleLocator(0.2))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.05))

    # Adjust figure placement and size
    fig.subplots_adjust(top=0.95, bottom=0.08, left=0.1, right=0.95)
    fig.set_size_inches(6.69291, 6.69291)

    # Save figure
    fig.savefig(os.path.join(os.path.dirname(__file__), 'output', 'figures',
                             f"solar_profiles_{tech_id.replace('|', '-')}.pdf"))

    plt.show()


if __name__ == '__main__':
    # Create plots
    plot_coal_build_costs()
    plot_gas_build_costs()
    plot_solar_build_costs()
    plot_wind_build_cost()
    plot_coal_fuel_costs()
    plot_gas_fuel_costs()
    plot_demand_profiles(nem_zone='ADE')
    plot_wind_capacity_factors(wind_bubble='YPS')
    plot_solar_capacity_factors(nem_zone='ADE', technology='DAT')
