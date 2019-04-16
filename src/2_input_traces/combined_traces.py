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


def format_hydro_traces(data_dir):
    """Format hydro data"""

    # Load hydro traces
    df = pd.read_pickle(os.path.join(data_dir, 'hydro_traces.pickle'))

    # Add label to column index
    df.columns = pd.MultiIndex.from_product([['HYDRO'], df.columns])

    # Rename index
    df.index.name = 'timestamp'

    return df


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
    # TODO: Fix hydro traces - must tile / repeat them for each year in model horizon
    df_dataset = pd.concat([df_wind, df_demand, df_hydro, df_solar], axis=1)
