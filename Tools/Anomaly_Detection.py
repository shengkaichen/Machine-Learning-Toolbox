from sklearn.neighbors import NearestNeighbors
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def nn(df, k):
    """
    Nearest Neighbors model
    :param df: data frame
    :param k: top k-nearest-neighbors have to be found
    :return: new data frame contains outlier
    """
    # transform input to array
    data = []
    if not isinstance(df, np.ndarray):
        data = df.values

    # implement Nearest Neighbors model
    model = NearestNeighbors(n_neighbors=k)
    model.fit(data)

    # array contains the lengths to points & index of the nearest points in the population matrix
    distances, neighbors = model.kneighbors(data)

    # find the cutoff values from the mean of k-distances in each observation
    plt.plot(distances.mean(axis=1))
    plt.show()
    plt.close()

    # set the cutoff value to filter outlier
    outlier = float(input("Enter value:"))
    index = np.where(distances.mean(axis=1) > outlier)
    values = df.iloc[index]

    # plot data
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], color="b", s=65)
    # plot outlier values
    plt.scatter(values.iloc[:, 0], values.iloc[:, 1], color="r")
    plt.show()

    return values
