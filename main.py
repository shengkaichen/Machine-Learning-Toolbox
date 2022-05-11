from AnomalyDetection import NearestNeighbors
from DataIntegration import Correlation, Similarity
from DataReduction import PrincipalComponentAnalysis
from DataTransformation import Normalization
from SupervisedLearning import Classification
from UnsupervisedLearning import Clustering
import pandas as pd
import numpy as np
import random
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


# Anomaly Detection examples
def ad():
    # random dataset with three 2D clusters totalling 1000 points
    r_pts = []
    r_seed = 47
    random.seed(r_seed)
    X, y = make_blobs(n_samples=5000, centers=7, n_features=2, random_state=r_seed)
    for i in range(200):  # random noise points
        r_pts.append([random.randint(-10, 10), random.randint(-10, 10)])
    test_arr = np.append(X, r_pts, axis=0)
    test_df = pd.DataFrame(dict(x=test_arr[:, 0], y=test_arr[:, 1]))
    test_df.plot(kind='scatter', x='x', y='y')
    plt.show()

    NearestNeighbors.nn(test_df, 10)


# Data Integration examples
def di():
    x = [1, 1, 1, 2, 2, 1, 0, 0, 0, 0]
    y = [0, 1, 0, 2, 2, 0, 1, 0, 0, 0]

    print('------------Similarity------------')
    print('Euclidean Distance between x and y is: ', Similarity.euclidean_distance(x, y))
    print('Manhattan Distance between x and y is: ', Similarity.manhattan_distance(x, y))
    print('Minkowski Distance between x and y is: ', Similarity.minkowski_distance(x, y))
    print('Cosine Similarity between x and y is: ', Similarity.cosine_similarity(x, y))
    print('Cosine Distance between x and y is: ', Similarity.cosine_distance(x, y))

    print('------------Correlation------------')
    print('Pearson Correlation Coefficient between x and y is: ', Correlation.pcc(x, y)[0][0])


# Data Reduction examples
def dr():
    # random dataset with three columns and name them V1, V2, and V3
    rng = np.random.RandomState(1)
    test_arr = np.dot(rng.rand(3, 3), rng.randn(3, 200)).T
    test_df = pd.DataFrame(test_arr, columns=["V1", "V2", "V3"])

    PrincipalComponentAnalysis.pca(test_df)


# Data Transformation examples
def dt(mode):
    test_df = pd.DataFrame(np.random.randint(0, 40, size=(1, 200)).T)
    test_df[1] = np.random.randint(0, 10, size=(1, 200)).T
    test_df.loc[len(test_df.index)] = [100, 5]

    if mode is 'z-score':
        test_data = Normalization.z_score(test_df)
    else:
        test_data = Normalization.min_max(test_df)

    plt.scatter(test_df.loc[:, 0], test_df.loc[:, 1], alpha=0.8)
    plt.axis('equal')
    plt.title('before')
    plt.show()

    plt.scatter(test_data.loc[:, 0], test_data.loc[:, 1], alpha=0.8)
    plt.axis('equal')
    plt.title('after')
    plt.show()


# di()
# ad()
# dr()
# dt('z-score')
dt('min-max')

