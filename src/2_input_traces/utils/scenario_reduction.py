"""Apply K-means scenario reduction algorithm to identify representative set of operating conditions"""

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def get_all_samples(df):
    """Construct samples to be used in K-mean algorithm"""

    # Add timestamp as column
    df[('INDEX', 'TIMESTAMP')] = df.index

    # Add hour to index
    df[('INDEX', 'HOUR')] = df[('INDEX', 'TIMESTAMP')].subtract(pd.Timedelta(minutes=10)).dt.hour

    # Add year to index
    df[('INDEX', 'YEAR')] = df[('INDEX', 'TIMESTAMP')].subtract(pd.Timedelta(minutes=10)).dt.year

    # Add day of year as column (will be used as chunk ID)
    df[('INDEX', 'DAY_OF_YEAR')] = df[('INDEX', 'TIMESTAMP')].subtract(pd.Timedelta(minutes=10)).dt.dayofyear

    # Set index to hour ID
    df = df.set_index(df[('INDEX', 'HOUR')])

    # Unstack DataFrame - each row corresponds to a sample in the K-mean algorithm
    df_s = df.groupby([('INDEX', 'YEAR'), ('INDEX', 'DAY_OF_YEAR')]).apply(lambda x: x.unstack())

    # Sort columns and drop columns with index values
    df_s = df_s.sort_index(axis=1).drop('INDEX', axis=1)

    return df_s


def get_samples_max_demand_days(df):
    """Identify days with greatest demand"""

    # Filter days with max demand in each year
    max_demand = df['DEMAND'].sum(level=[1], axis=1).max(axis=1).groupby(('INDEX', 'YEAR')).idxmax()
    max_demand = pd.MultiIndex.from_tuples(max_demand)

    # Samples for days on which max system demand occurs each year
    df_o = df.loc[max_demand, :].copy()

    return df_o


def standardise_samples(df):
    """Standardise samples"""

    # Standardise samples
    mean = df.mean()
    std = df.std()

    # If no variation in column, set std = 1 (avoid divide 0 errors). Same
    # as what sklearn does when scaling.
    std = std.replace(0, 1)

    # Standardised samples
    df_o = (df - mean) / std

    # Check that NaNs are not in standardised DataFrame
    assert not df_o.replace(-np.inf, np.nan).replace(np.inf, np.nan).isna().any().any()

    return df_o, mean, std


def get_centroids(samples, samples_max_demand, year, scenarios_per_year=10):
    """Construct centroids for a given year"""

    # Extract samples for a given year
    df = samples.loc[year, :].copy()

    # Standardise samples for a given year
    df_in, mean, std = standardise_samples(df)

    # Samples to be used in K-mean classifier
    X = df_in.values

    # Construct and fit K-means classifier
    kmeans = KMeans(n_clusters=scenarios_per_year-1, random_state=0).fit(X)

    # Assign each sample to a cluster
    X_prediction = kmeans.predict(X)

    # Assign cluster ID to each sample
    df_in[('K_MEANS', 'METRIC', 'CLUSTER')] = X_prediction
    df_in[('K_MEANS', 'METRIC', 'CLUSTER')] = df_in[('K_MEANS', 'METRIC', 'CLUSTER')].astype(int)

    def _compute_cluster_distance(row):
        """Compute distance between sample and it's assigned centroid"""

        # Get centroid
        centroid = kmeans.cluster_centers_[int(row[('K_MEANS', 'METRIC', 'CLUSTER')])]

        # Sample (removing cluster number information)
        sample = row.drop(('K_MEANS', 'METRIC', 'CLUSTER'))

        # Euclidean norm between sample and centroid
        distance = np.linalg.norm(sample - centroid)

        return distance

    # Compute Euclidean norm between each sample and its respective centroid
    df_in[('K_MEANS', 'METRIC', 'CENTROID_DISTANCE')] = df_in.apply(_compute_cluster_distance, axis=1)

    # Find samples that minimise distance to each respective centroid
    centroid_samples = (df_in.groupby(('K_MEANS', 'METRIC', 'CLUSTER'))[[('K_MEANS', 'METRIC', 'CENTROID_DISTANCE')]]
                        .apply(lambda x: pd.Series({'DURATION': x.size, 'HOUR_ID': x.idxmin()[0]})))

    # Extract clusters that minimise the distance to their respective centroid
    b = df_in.loc[centroid_samples['HOUR_ID']]

    # Re-scale values
    df_rescaled = (b.drop('K_MEANS', axis=1) * std) + mean

    # Add duration of each cluster to DataFrame
    df_rescaled[('K_MEANS', 'METRIC', 'DURATION')] = (df_rescaled.apply(lambda x: centroid_samples.set_index('HOUR_ID')
                                                                        .loc[x.name, 'DURATION'], axis=1))

    # Add series with max demand
    df_o = pd.concat([df_rescaled, samples_max_demand.loc[year]])

    # Add duration for day with max demand (1 day)
    df_o.loc[samples_max_demand.loc[year].index[0], ('K_MEANS', 'METRIC', 'DURATION')] = 1

    # Compute normalised duration (number of days per centroid / total days in year)
    df_o[('K_MEANS', 'METRIC', 'NORMALISED_DURATION')] = (df_o[('K_MEANS', 'METRIC', 'DURATION')]
                                                          .div(df_o[('K_MEANS', 'METRIC', 'DURATION')].sum()))

    # Retain day ID in column
    df_o[('K_MEANS', 'METRIC', 'DAY_ID')] = df_o.index

    # Reset index
    df_o = df_o.reset_index(drop=True)

    # Add year to index
    df_o['YEAR'] = year

    # Set index
    df_o = df_o.set_index('YEAR', append=True).swaplevel(1, 0, 0)

    return df_o


def check_plots(centroids, samples):
    """
    Plot selected demand, wind, and solar data to visually inspect
    algorithm output
    """

    # Plot figures
    fig, ax = plt.subplots()

    # All demand curves
    samples[('DEMAND', 'ADE')].T.plot(ax=ax, color='r', alpha=0.2)

    # Clustered demand curves
    centroids[('DEMAND', 'ADE')].T.plot(ax=ax, color='b', alpha=0.8, legend=False)

    # Plot figures
    fig, ax = plt.subplots()

    # All demand curves
    samples[('WIND', 'WEN')].T.plot(ax=ax, color='r', alpha=0.2)

    # Clustered demand curves
    centroids[('WIND', 'WEN')].T.plot(ax=ax, color='b', alpha=0.8, legend=False)

    # Plot figures
    fig, ax = plt.subplots()

    # All solar curves
    samples[('SOLAR', 'ADE|SAT')].T.plot(ax=ax, color='r', alpha=0.2)

    # Clustered solar curves
    centroids[('SOLAR', 'ADE|SAT')].T.plot(ax=ax, color='b', alpha=0.8, legend=False)


def main(output_dir, scenarios_per_year):
    """
    Process scenario data

    Parameters
    ----------
    output_dir : str
        Directory where output files are to be stored (also contains outputs from previous steps)

    scenarios_per_year : int
        Number of operating scenarios per year

    Returns
    -------
    df_all_centroids : pandas DataFrame
        Pandas DataFrame containing centroids for all scenarios generated
    """

    # Load dataset of pre-processed input traces
    dataset = pd.read_hdf(os.path.join(os.path.dirname(__file__), os.path.pardir, 'output', f'dataset.h5'))

    # All samples
    all_samples = get_all_samples(dataset)

    # Samples for days with max demand
    max_demand_days = get_samples_max_demand_days(all_samples)

    # Remaining samples
    remaining_samples = all_samples.loc[all_samples.index.difference(max_demand_days.index), :].copy()

    # Container for all centroids
    all_centroids = []

    for y in range(2016, 2051):
        print(f'Processing year {y}')

        # Centroids computed for a given year
        df_c = get_centroids(remaining_samples, max_demand_days, y, scenarios_per_year)

        # Append to main container
        all_centroids.append(df_c)

    # Compute all centroids
    df_all_centroids = pd.concat(all_centroids)

    # Save to file
    df_all_centroids.to_pickle(os.path.join(output_dir, f'centroids_{scenarios_per_year}.pickle'))

    return df_all_centroids


if __name__ == '__main__':
    # Output data directory
    output_directory = os.path.join(os.path.dirname(__file__), os.path.pardir, 'output')

    # Construct scenarios
    df_centroids = main(output_directory, scenarios_per_year=10)
