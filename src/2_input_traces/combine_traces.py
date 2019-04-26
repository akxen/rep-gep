"""Combine demand, hydro, wind, and solar traces into a single DataFrame"""

import os

import pandas as pd
import matplotlib.pyplot as plt


def pad_dates_backward(df):
    """
    Pad beginning of DataFrame with values corresponding to the closest day-hour

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame with timestamp index

    Returns
    -------
    df_o : pandas DataFrame
        DataFrame with padded values at beginning of time index
    """

    # Check for missing values
    assert not df.isna().any().any(), 'Missing demand values'

    # Backward fill using the first available observation for corresponding day-hour
    # ------------------------------------------------------------------------------
    df_tmp = df.copy()
    df_tmp[('INDEX', 'MINUS_1_YEAR')] = df_tmp.apply(lambda x: x.name - pd.Timedelta(days=365), axis=1)
    df_tmp = df_tmp.set_index(('INDEX', 'MINUS_1_YEAR'))

    # New index for beginning of demand data
    new_index_start = pd.date_range(start='2016-01-01 01:00:00', end=df.index[-1], freq='1H')

    # Reindex DataFrame
    df_o = df.reindex(new_index_start).copy()

    # Update missing value at start of index with corresponding day-hours from following year
    df_o.update(df_tmp, overwrite=False)

    return df_o


def pad_dates_forward(df):
    """
    Pad end of DataFrame with values corresponding to the closest day-hour

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame with timestamp index

    Returns
    -------
    df_o : pandas DataFrame
        DataFrame with padded values at end of time index
    """

    # Data for last year in dataset (include 366 days in case of leap year)
    df_last_year = df[-8784:].copy()

    # Add index for day and hour of year
    df_last_year[('INDEX', 'DAY_OF_YEAR')] = df_last_year.index.dayofyear
    df_last_year[('INDEX', 'HOUR')] = df_last_year.index.hour
    df_last_year = df_last_year.set_index([('INDEX', 'DAY_OF_YEAR'), ('INDEX', 'HOUR')])

    # Drop duplicates (if no leap year some rows will be removed)
    df_last_year = df_last_year[~df_last_year.index.duplicated(keep='last')]

    # In the event of a leap year, assume that the 366th day of the year is similar to the 365th
    if 366 not in df_last_year.index.levels[0]:
        leap_day = df_last_year.loc[365].copy()
        leap_day.loc[:, ('INDEX', 'DAY')] = 366
        leap_day = leap_day.set_index(('INDEX', 'DAY'), append=True).swaplevel(1, 0, 0)

        # Update last year data with leap day
        df_last_year = pd.concat([df_last_year, leap_day])

    # Assert no duplicates in index
    assert not df_last_year.index.duplicated().any(), 'Duplicated day-hour index'

    # Check no missing values before updating (don't want to mistakenly update other years)
    assert not df.isna().any().any(), 'Missing values'

    # New index for end of DataFrame
    new_index_end = pd.date_range(start=df.index[0], end='2050-01-01 00:00:00', freq='1H')

    # Reindex DataFrame
    df_o = df.reindex(new_index_end).copy()

    # Add timestamp, day of year, and hour to DataFrame
    df_o[('INDEX', 'TIMESTAMP')] = df_o.index
    df_o[('INDEX', 'DAY_OF_YEAR')] = df_o.index.dayofyear
    df_o[('INDEX', 'HOUR')] = df_o.index.hour

    # Set index as day of year and hour
    df_o = df_o.set_index([('INDEX', 'DAY_OF_YEAR'), ('INDEX', 'HOUR')])

    # Update data for final years in dataset
    df_o.update(df_last_year, overwrite=False)

    # Use timestamp as index
    df_o = df_o.set_index(('INDEX', 'TIMESTAMP'))

    # Check no missing values
    # assert not df_o.isna().any().any(), 'Missing values'

    return df_o


def format_wind_traces(data_dir):
    """Format wind traces"""

    # Load wind traces
    df = pd.read_hdf(os.path.join(data_dir, 'wind_traces.h5'))

    # Reset index and pivot
    df = df.reset_index().pivot(index='timestamp', columns='bubble', values='capacity_factor')

    # Add level to column index
    df.columns = pd.MultiIndex.from_product([['WIND'], df.columns])

    # Pad data forward
    df = pad_dates_forward(df)

    return df


def format_demand_traces(data_dir, network_dir):
    """Format demand traces"""

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

    # Add label to columns
    df_zone_demand.columns = pd.MultiIndex.from_product([['DEMAND'], df_zone_demand.columns])

    # Backfill corresponding day-hour values to extend the series
    df_zone_demand = pad_dates_backward(df_zone_demand)

    # Forwardfill corresponding day-hour values to extend the series
    df_zone_demand = pad_dates_forward(df_zone_demand)

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

    # Add column index denoting solar data
    df.columns = pd.MultiIndex.from_product([['SOLAR'], df.columns])

    # Pad dates backward (ensure data starts from 2016)
    df = pad_dates_backward(df)

    # Pad dates forward (ensure data ends at beginning of 2050
    df = pad_dates_forward(df)

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
    # Join datasets
    df_dataset = df_hydro.join(df_wind, how='left').join(df_demand, how='left').join(df_solar, how='left')

    # Check for missing values
    assert not df_dataset.isna().any().any(), 'Missing values in dataset'

    # Check for duplicated indices
    assert not df_dataset.index.duplicated().any(), 'Dataset has duplicated indices'

    # Check for duplicated columns
    assert not df_dataset.columns.duplicated().any(), 'Dataset has duplicated columns'

    # Plots of selected series to double check data to double check
    # -------------------------------------------------------------
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
    df_dataset.to_hdf(os.path.join(output_directory, 'dataset.h5'), key='dataset')
