"""Apply K-mean scenario algorithm to identify representative set of scenarios"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# import matplotlib.pyplot as plt


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


def get_centroids(df, year, n_clusters=7):
    """Apply K-means algorithm to compute centroids for a given year"""

    # Sample
    x = df.loc[year].values
    # .fillna(0)

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

    # Container for all centroids
    all_centroids = []

    for year in range(2016, 2051):

        try:
            # Compute centroids and associated duration for each sample
            centroids = get_centroids(samples, year)

            # Add year to index
            centroids['year'] = year
            centroids = centroids.set_index('year', append=True).swaplevel(1, 0, 0)

            # Append to main container
            all_centroids.append(centroids)

            print(f'Finished processing centroids for {year}')

        except:
            print(f'Failed to process year {year}')

    # Combine all centroids into single DataFrame
    df_c = pd.concat(all_centroids)

    # Save output
    df_c.to_pickle('output/centroids.pickle')
