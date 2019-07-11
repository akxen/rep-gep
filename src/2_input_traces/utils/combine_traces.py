"""Combine demand, hydro, wind, and solar traces into a single DataFrame"""

import os
import time

import pandas as pd
import matplotlib.pyplot as plt


def _pad_column(col, direction):
    """Pad values forwards or backwards to a specified date"""
    # Drop missing values
    df = col.dropna()

    # Convert to DataFrame
    df = df.to_frame()

    # Options that must change depending on direction in which to pad
    if direction == 'forward':
        keep = 'last'
        new_index = pd.date_range(start=df.index[0], end='2051-01-01 00:00:00', freq='1H')

    elif direction == 'backward':
        keep = 'first'
        new_index = pd.date_range(start='2016-01-01 01:00:00', end=df.index[-1], freq='1H')

    else:
        raise Exception(f'Unexpected direction: {direction}')

    # Update index
    df = df.reindex(new_index)

    def _get_hour_of_year(row):
        """Get hour of year"""

        # Get day of year - adjust by 1 minute so last timestamp (2051-01-01 00:00:00)
        # is assigned to 2050. Note this timestamp actually corresponds to the interval
        # 2050-12-31 23:00:00 to 2051-01-01 00:00:00
        day_timestamp = row.name - pd.Timedelta(minutes=1)

        # Day of year
        day = day_timestamp.dayofyear

        # Hour of year
        hour = ((day - 1) * 24) + day_timestamp.hour + 1

        return hour

    # Hour of year
    df['hour_of_year'] = df.apply(_get_hour_of_year, axis=1).to_frame('hour_of_year')

    # Last year with complete data
    fill_year = df.dropna(subset=[col.name]).drop_duplicates(subset=['hour_of_year'], keep=keep)

    # DataFrame that will have values padded forward
    padded = df.reset_index().set_index('hour_of_year')

    # Pad using values from last year with complete data
    padded.update(fill_year.set_index('hour_of_year'), overwrite=False)

    # Set timestamp as index
    padded = padded.set_index('index')

    # Return series
    padded = padded[col.name]

    return padded


def pad_dataframe(col):
    """Apply padding - forwards and backwards for each column in DataFrame"""

    # Pad values forwards
    padded = _pad_column(col, direction='forward')

    # Pad values backwards
    padded = _pad_column(padded, direction='backward')

    return padded


def format_wind_traces(data_dir):
    """Format wind traces"""

    # Load wind traces
    df = pd.read_hdf(os.path.join(data_dir, 'wind_traces.h5'))

    # Reset index and pivot
    df = df.reset_index().pivot(index='timestamp', columns='bubble', values='capacity_factor')

    # Pad data forward
    df = df.apply(pad_dataframe)

    # Add level to column index
    df.columns = pd.MultiIndex.from_product([['WIND'], df.columns])

    return df


def format_demand_traces(data_dir, root_data_dir):
    """
    Format demand traces

    Note: Only considering the 'neutral' demand scenario
    """

    # Construct directory containing network data
    network_dir = os.path.join(root_data_dir, 'files', 'egrimod-nem-dataset-v1.3', 'akxen-egrimod-nem-dataset-4806603',
                               'network')

    # Load demand traces
    df_region_demand = pd.read_hdf(os.path.join(data_dir, 'demand_traces.h5'))

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

    # Pad corresponding day-hour values to extend the series to a specified date
    df_zone_demand = df_zone_demand.apply(pad_dataframe)

    # Add label to columns
    df_zone_demand.columns = pd.MultiIndex.from_product([['DEMAND'], df_zone_demand.columns])

    return df_zone_demand


def format_solar_traces(data_dir):
    """Format solar data"""

    # Solar traces
    df = pd.read_hdf(os.path.join(data_dir, 'solar_traces.h5'))

    # Pivot so different solar technologies and their respective zones constitute the columns
    # and timestamps the index
    df = df.reset_index().pivot_table(index='timestamp', columns=['zone', 'technology'], values='capacity_factor')

    # Merge column index levels so NEM zone and technology represented in single label
    df.columns = df.columns.map('|'.join).str.strip('|')

    # Pad dates (ensure data starts from 2016 and ends at end of 2050)
    df = df.apply(pad_dataframe)

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

    # Hydro traces
    df = pd.read_hdf(os.path.join(data_dir, 'hydro_traces.h5'))

    # Add hour, day, month to DataFrame
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month

    # Construct new DataFrame with index from 2016 - 2050
    model_horizon_index = pd.date_range(start='2016-01-01 01:00:00', end='2051-01-01 00:00:00', freq='1H')

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


def main(root_data_dir, output_dir):
    """
    Combine all input traces

    Parameters
    ----------
    root_data_dir : str
        Directory containing core data files on which scenarios are based

    output_dir : str
        Directory for output files

    Returns
    -------
    df_dataset : pandas DataFrame
        Dataset containing all input traces
    """

    print('Combining all input traces')
    print('--------------------------')

    # Formatted traces
    # ----------------
    # Wind traces
    print('Processing wind data')
    start = time.time()
    df_wind = format_wind_traces(output_dir)
    print(f'Processed wind data in: {time.time() - start}s')

    # Demand traces
    start = time.time()
    df_demand = format_demand_traces(output_dir, root_data_dir)
    print(f'Processed demand data in: {time.time() - start}s')

    # Hydro traces
    start = time.time()
    df_hydro = format_hydro_traces(output_dir)
    print(f'Processed hydro data in: {time.time() - start}s')

    # Solar traces
    start = time.time()
    df_solar = format_solar_traces(output_dir)
    print(f'Processed solar data in: {time.time() - start}s')

    # Merge into single DataFrame
    # ---------------------------
    # Join datasets
    df_dataset = df_hydro.join(df_wind, how='left').join(df_demand, how='left').join(df_solar, how='left')

    # Check for missing values
    assert not df_dataset.isna().any().any(), 'Missing values in dataset'

    # Check for duplicated indices
    assert not df_dataset.index.duplicated().any(), 'Dataset has duplicated indices'

    # Check for duplicated columns
    assert not df_dataset.columns.duplicated().any(), 'Dataset has duplicated columns'

    # Plots of selected series to check data
    # --------------------------------------
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4)

    # Demand
    df_dataset[('DEMAND', 'CAN')].plot(title='Demand - CAN', ax=ax1)

    # Hydro
    df_dataset[('HYDRO', 'MEADOWBK')].plot(title='Hydro - MEADOWBK', ax=ax2)

    # Wind
    df_dataset[('WIND', 'FNQ')].plot(title='Wind - FNQ', ax=ax3)

    # Solar
    df_dataset[('SOLAR', 'ADE|FFP2')].plot(title='Solar - ADE|FFP2', ax=ax4)

    # Save to file
    # ------------
    df_dataset.to_hdf(os.path.join(output_dir, 'dataset.h5'), key='dataset')

    return df_dataset


if __name__ == '__main__':
    # Root data directory
    root_data_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir,
                                       'data')

    # Directory containing output files (contains inputs from previous steps)
    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, 'output')

    # Combine input traces
    df_dataset = main(root_data_directory, output_directory)
