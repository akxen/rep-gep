"""Apply K-mean scenario algorithm to identify representative set of scenarios"""

from math import sqrt

import numpy as np
import pandas as pd

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
    max_demand_days = df['DEMAND'].sum(level=[1], axis=1).max(axis=1).groupby(('INDEX', 'YEAR')).idxmax()
    max_demand_days = pd.MultiIndex.from_tuples(max_demand_days)

    # Samples for days on which max system demand occurs each year
    df_o = df.loc[max_demand_days, :].copy()

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

    return df_o


def get_centroids(df, year, n_clusters=7):
    """Apply K-means algorithm to compute centroids for a given year"""

    # Sample
    x = df.loc[year].values

    # Construct and fit K-means classifier
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(x)

    # K-means assignment for each sample
    assignment = kmeans.predict(x)

    # Number of profiles assigned to each centroid
    assignment_count = np.unique(assignment, return_counts=True)

    # Duration assigned to each profile
    duration = {i: j for i, j in zip(assignment_count[0], assignment_count[1])}

    # Convert centroid array into DataFrame
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=df.columns)

    # Add duration information to DataFrame
    centroids['DURATION'] = centroids.apply(lambda x: duration[x.name], axis=1)

    return centroids


if __name__ == '__main__':
    # Load dataset
    dataset = pd.read_hdf('output/dataset.h5')

    # All samples
    samples = get_all_samples(dataset)

    # Samples for days with max demand
    df_max_demand = get_samples_max_demand_days(samples)

    # Remaining samples
    df_s = samples.loc[samples.index.difference(df_max_demand.index), :].copy()

    # Scale sample for one year
    df = df_s.loc[2016, :].copy()

    # Standardise samples for a given year
    df_in = standardise_samples(df)

    # Samples to be used in K-mean classifier
    X = df_in.values

    # Construct and fit K-means classifier
    n_clusters = 9
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    # Assign each sample to a cluster
    X_pred = kmeans.predict(X)

    # Assign cluster ID to each sample
    df_in[('K_MEANS', 'METRIC', 'CLUSTER')] = X_pred
    df_in[('K_MEANS', 'METRIC', 'CLUSTER')] = df_in[('K_MEANS', 'METRIC', 'CLUSTER')].astype(int)

    # df_in.apply(lambda x: kmeans.cluster_centers_[int(x[('K_MEANS', 'METRIC', 'CLUSTER')])], axis=1)

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
    centroid_samples = df_in.groupby(('K_MEANS', 'METRIC', 'CLUSTER'))[[('K_MEANS', 'METRIC', 'CENTROID_DISTANCE')]].idxmin()

    # Reset index
    centroid_samples = centroid_samples.reset_index()

    a = centroid_samples.reset_index().droplevel(0, axis=1).droplevel(0, axis=1)

    b = pd.merge(a, df_in, how='left', left_on=['CENTROID_DISTANCE'], right_index=True)





    # # Container for all centroids
    # all_centroids = []
    #
    # for year in range(2016, 2051):
    #
    #     try:
    #         # Compute centroids and associated duration for each sample
    #         centroids = get_centroids(samples, year)
    #
    #         # Add year to index
    #         centroids['year'] = year
    #         centroids = centroids.set_index('year', append=True).swaplevel(1, 0, 0)
    #
    #         # Append to main container
    #         all_centroids.append(centroids)
    #
    #         print(f'Finished processing centroids for {year}')
    #
    #     except:
    #         print(f'Failed to process year {year}')
    #
    # # Combine all centroids into single DataFrame
    # df_c = pd.concat(all_centroids)
    #
    # # Save output
    # df_c.to_pickle('output/centroids.pickle')
