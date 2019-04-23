"""Apply K-mean scenario algorithm to identify representative set of scenarios"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt


def partition_timestamps(year):
    """Partition timestamps into 24 hour segments for a given year"""

    # All hourly timestamps within one year
    timestamps = pd.date_range(start=f'{year}-01-01 01:00:00', end=f'{year+1}-01-01 00:00:00', freq='1H')

    # Construct chunks
    # ----------------
    # Number of timestamps per chunk
    n = 24

    # Timestamps within each chunk
    timestamp_chunks = [timestamps[i:i + n] for i in range(0, len(timestamps), n)]

    return timestamp_chunks


# Construct vector describing parameters for each segment
def compile_parameters(timestamps):
    """Construct sample given a list of timestamps"""

    # Unstack DataFrame
    series = df.loc[timestamps, :].unstack().sort_index()

    pass


# Apply K-means algorithm to construct centroids (compute 7 centroids for each year)
def construct_centroids(samples):
    """Apply K-mean algorithm to construct centroids"""
    pass

# Compare load curve of resulting centroids to load curve for whole year

# (check if day with highest demand should be included)

# Compute the

# Output
# index: year, scenario_id (1-7), hour_id (1-48)
# columns: cat 1 (data type e.g. wind, hydro), cat 2 (region / sub-id e.g. zone, DUID)


if __name__ == '__main__':
    # Load dataset
    df = pd.read_hdf('output/dataset.h5')

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

    # Sample
    X = df_s.loc[2020].fillna(0).values

    # Construct and fit K-means classifier
    kmeans = KMeans(n_clusters=7, random_state=0).fit(X)

    # K-means assignment for each profile
    assignment = kmeans.predict(X)
    print(np.unique(assignment, return_counts=True))

    # Convert centroid array into DataFrame
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=df_s.columns)













