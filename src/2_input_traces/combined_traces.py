"""Combine demand, hydro, wind, and solar traces into a single DataFrame"""

import os

import numpy as np
import pandas as pd


def format_wind_traces(data_dir):
    """Format wind traces"""

    # Load wind traces
    df = pd.read_pickle(os.path.join(data_dir, 'wind_traces.pickle'))

    # Reset index and pivot
    df = df.reset_index().pivot(index='timestamp', columns='bubble', values='capacity_factor')

    # Add level to column index
    df.columns = pd.MultiIndex.from_product([['WIND'], df.columns])

    return df


def format_demand_traces(data_dir, network_dir):
    """Format demand traces"""

    # Load demand traces
    df_region_demand = pd.read_pickle(os.path.join(data_dir, 'demand_traces.pickle'))

    # Only consider neutral demand scenario
    df_region_demand = df_region_demand.loc[df_region_demand['scenario'] == 'Neutral', :]

    # Reindex and pivot
    df_region_demand = df_region_demand.reset_index().pivot(index='timestamp', columns='region', values='demand')

    # Add suffix so region IDs are consistent with other MMSDM datasets
    df_region_demand = df_region_demand.add_suffix('1')

    # Network nodes
    df_nodes = pd.read_csv(os.path.join(network_dir, 'network_nodes.csv'), index_col='NODE_ID')

    # Proportion of region demand consumed in each zone
    df_allocation = df_nodes.groupby(['NEM_REGION', 'NEM_ZONE'])['PROP_REG_D'].sum()

    df_zone_demand = pd.DataFrame(index=df_region_demand.index, columns=df_allocation.index)

    # Demand for each zone
    def _get_zone_demand(row):
        """Disaggregate region demand into zonal demand"""

        # Demand in each zone
        zone_demand = df_allocation.loc[(row.name[0], row.name[1])] * df_region_demand[row.name[0]]

        return zone_demand

    # Demand in each zone
    df_zone_demand = df_zone_demand.apply(_get_zone_demand)

    # Remove region ID from column index
    df_zone_demand = df_zone_demand.droplevel(0, axis=1)

    # Add label to columns
    df_zone_demand.columns = pd.MultiIndex.from_product([['DEMAND'], df_zone_demand.columns])

    return df_zone_demand


# def format_hydro_traces(data_dir):
#     """Format hydro data"""
#
#     # Load hydro traces
#     df = pd.read_pickle(os.path.join(data_dir, 'hydro_traces.pickle'))
#
#
#
#     # Rename index
#     df.index.name = 'timestamp'
#
#     return df


def format_solar_traces(data_dir):
    """Format solar data"""

    # Solar traces
    df = pd.read_pickle(os.path.join(data_dir, 'solar_traces.pickle'))

    # Pivot so different solar technologies and their respective zones constitute the columns
    # and timestamps the index
    df = df.reset_index().pivot_table(index='timestamp', columns=['zone', 'technology'], values='capacity_factor')

    # Merge column index levels so NEM zone and technology represented in single label
    df.columns = df.columns.map('|'.join).str.strip('|')

    # Add column index denoting solar data
    df.columns = pd.MultiIndex.from_product([['SOLAR'], df.columns])

    return df


def format_hydro_traces(data_dir):
    """
    Repeat hydro traces for each year in model horizon

    Note: Assuming that hydro traces are mainly influenced by seasonal
    weather events, and similar cycles are observed year to year. Signals
    in 2016 are repeated for corresponding hour-day-months in the following
    years. E.g. hydro output on 2016-01-06 05:00:00 is the same on
    2030-01-06 05:00:00.
    """

    # Solar traces
    df = pd.read_pickle(os.path.join(data_dir, 'solar_traces.pickle'))

    # Add hour, day, month to DataFrame
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month

    # Construct new DataFrame with index from 2016 - 2050
    model_horizon_index = pd.date_range(start='2016-01-01 01:00:00', end='2050-01-01 00:00:00', freq='1H')

    # Initialize DataFrame for hydro traces over the entire model horizon
    df_o = pd.DataFrame(index=model_horizon_index, columns=['hour', 'day', 'month'])

    # Add hour, day, month to new DataFrame
    df_o['hour'] = df_o.index.hour
    df_o['day'] = df_o.index.day
    df_o['month'] = df_o.index.month

    # Reset index
    df_o = df_o.reset_index()

    # Merge hydro traces for days in 2016 to horizon DataFrame
    df_o = pd.merge(df_o, df, how='left', left_on=['hour', 'day', 'month'],
                          right_on=['hour', 'day', 'month'])

    # Set index and drop redundant columns
    df_o = df_o.set_index('index').drop(['hour', 'day', 'month'], axis=1)

    # Check there are no missing values
    assert not df_o.isna().any().any(), 'NaNs in hydro traces DataFrame'

    # Add label to column index
    df_o.columns = pd.MultiIndex.from_product([['HYDRO'], df_o.columns])

    return df_o


if __name__ == '__main__':
    # Paths
    # -----
    # Directory containing output files (contains inputs from previous steps)
    output_directory = os.path.join(os.path.curdir, 'output')

    # Directory containing network node information
    network_data_directory = os.path.join(os.path.curdir, os.path.pardir, os.path.pardir, 'data', 'files',
                                          'egrimod-nem-dataset-v1.3', 'akxen-egrimod-nem-dataset-4806603', 'network')

    # Formatted traces
    # ----------------
    # Wind traces
    df_wind = format_wind_traces(output_directory)

    # Demand traces
    df_demand = format_demand_traces(output_directory, network_data_directory)

    # Hydro traces
    df_hydro = format_hydro_traces(output_directory)

    # Solar traces
    df_solar = format_solar_traces(output_directory)

    # Merge into single DataFrame
    # ---------------------------
    df_dataset = pd.concat([df_wind, df_demand, df_hydro, df_solar], axis=1)
