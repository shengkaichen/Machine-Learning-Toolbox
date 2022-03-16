import Data_Transformation
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def k_mean(data):
    """
    :param data: data frame
    :return: data frame with the cluster column
    """
    # standardizing the data
    data_scaled = Data_Transformation.zScore(data)

    # statistics of scaled data
    pd.DataFrame(data_scaled).describe()

    # defining the kmeans function with initialization as k-means++
    kmeans = KMeans(n_clusters=2, init='k-means++')

    # fitting the k means algorithm on scaled data
    kmeans.fit(data_scaled)

    # inertia on the fitted data - it is sum of squared distances of samples to their closest cluster center
    kmeans.inertia_

    # fitting multiple k-means algorithms and storing the values in an empty list
    SSE = []
    for cluster in range(1, 20):
        kmeans = KMeans(n_clusters=cluster, init='k-means++')
        kmeans.fit(data_scaled)
        SSE.append(kmeans.inertia_)

    # converting the results into a dataframe and plotting them
    frame = pd.DataFrame({'Cluster': range(1, 20), 'SSE': SSE})
    plt.figure(figsize=(12, 6))
    plt.plot(frame['Cluster'], frame['SSE'], marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

    # k means using 5 clusters and k-means++ initialization
    kmeans = KMeans(n_clusters=20, init='k-means++')
    kmeans.fit(data_scaled)
    res = kmeans.predict(data_scaled)

    frame = pd.DataFrame(data)
    frame['cluster'] = res
    frame['cluster'].value_counts()

    return frame
