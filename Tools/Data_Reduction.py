from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def pca(data):
    """
    Dimensionality Reduction Method
    :param data: multidimensional data frame
    :return: 2D data frame
    """
    # Visualize the raw data
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], 'gray')

    # implement PCA model
    model = PCA(n_components=2)
    model.fit(data)

    # Lets visualize the transformed data
    data_pca = model.transform(data)
    print("original shape:   ", data.shape)
    print("transformed shape:", data_pca.shape)

    data_new = model.inverse_transform(data_pca)
    plt.scatter(data_new[:, 0], data_new[:, 1], alpha=0.8)
    plt.axis('equal')
    plt.show()

    return pd.DataFrame(data_pca)
