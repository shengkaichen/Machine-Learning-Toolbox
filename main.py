"""
How to see the examples?
    -> remove the # on the example that you want to test
How to use the algorithms in your project?
    -> you can use the following command (from "folder" import "python file")  to use any method
"""

from Tools import Anomaly_Detection
from Tools import Data_Integration
from Tools import Data_Reduction
from Tools import Data_Transformation
from Tools import Supervised_Learning
from Tools import Unsupervised_Learning
import pandas as pd
import numpy as np
import random
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


# Example of data integration that finds the distance and similarity of x and y
def example_data_integration():
    x = [1, 1, 1, 2, 2, 1, 0, 0, 0, 0]
    y = [0, 1, 0, 2, 2, 0, 1, 0, 0, 0]
    print('------------Similarity------------')
    similarity = Data_Integration.Similarity(x, y)
    similarity.euclidean_distance()
    similarity.manhattan_distance()
    similarity.minkowski_distance()
    similarity.cosine_similarity()
    similarity.cosine_distance()
    print('------------Correlation------------')
    correlation = Data_Integration.Correlation(x, y)
    correlation.pearson_correlation_coefficient()


# Example of data reduction that reduces the 3D table into 2D table
def example_data_reduction():
    rng = np.random.RandomState(1)
    r = np.dot(rng.rand(3, 3), rng.randn(3, 200)).T
    test = pd.DataFrame(r, columns=["a1", "a2", "a3"])

    print('------------PCA------------')
    Data_Reduction.pca(test)


# Example of data transformation that normalizes the data
def example_data_transformation():
    r = np.random.randint(200, size=(100, 5))
    test = pd.DataFrame(r)

    print('------------Z-score normalization------------')
    print(Data_Transformation.z_score(test))
    print('------------Min-max normalization------------')
    print(Data_Transformation.min_max(test))


# Example of anomaly detection that find the outlier in the data
def example_anomaly_detection():
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

    # this detection requires user to manually set the cutoff value to filter outlier
    Anomaly_Detection.nn(test_df, 10)


# Example of anomaly detection that find the outlier in the data
def example_supervised_learning():
    return 0


# Example of anomaly detection that find the outlier in the data
def example_unsupervised_learning():
    return 0


# example_data_integration()
# example_data_reduction()
# example_data_transformation()
# example_anomaly_detection()
# example_supervised_learning()
# example_unsupervised_learning()
