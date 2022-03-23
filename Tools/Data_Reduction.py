from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt


def pca(data):
    """
    Dimensionality Reduction Method
    :param data: multidimensional data frame
    :return: 2D data frame
    """
    # visualize the raw data in 3D
    if data.shape[1] == 3:
        ax = plt.axes(projection='3d')
        ax.scatter3D(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], 'gray')
        plt.show()
        plt.close()

    # implement PCA model
    model = PCA(n_components=2)
    model.fit(data)

    # visualize the transformed data
    data_pca = model.transform(data)
    print("original shape:   ", data.shape)
    print("transformed shape:", data_pca.shape)

    data_new = model.inverse_transform(data_pca)
    plt.scatter(data_new[:, 0], data_new[:, 1], alpha=0.8)
    plt.axis('equal')
    plt.show()

    return pd.DataFrame(data_pca)
