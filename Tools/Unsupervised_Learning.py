from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd


def k_mean_find_k(data_scaled):
    """
    :param data_scaled: data frame with standardizing
    :return: plot
    """
    # statistics of scaled data
    pd.DataFrame(data_scaled).describe()

    # defining the kmeans function with initialization as k-means++
    kmeans = KMeans(n_clusters=2, init='k-means++')

    # fitting the k means algorithm on scaled data
    kmeans.fit(data_scaled)

    # inertia on the fitted data - it is sum of squared distances of samples to their closest cluster center
    print('kmeans = 2, inertia = ', kmeans.inertia_)

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


def k_mean(data_scaled, k):
    """
    :param data_scaled: data frame with standardizing
    :param k: number of clusters
    :return: data frame with the cluster column
    """
    # k means using input clusters and k-means++ initialization
    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(data_scaled)
    res = kmeans.predict(data_scaled)

    # Inertia measures how well a dataset was clustered by K-Means
    print('inertia: ', kmeans.inertia_)

    # show the cluster
    plt.scatter(data_scaled.iloc[:, 0], data_scaled.iloc[:, 1], c=res, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()

    # return the cluster value in same order of input
    frame = pd.DataFrame({'cluster': res})
    frame['cluster'].value_counts()

    return frame




