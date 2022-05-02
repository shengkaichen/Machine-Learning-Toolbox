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

test_df = pd.DataFrame
test_arr = []

# random dataset with three columns and name them V1, V2, and V3
rng = np.random.RandomState(1)
test_arr = np.dot(rng.rand(3, 3), rng.randn(3, 200)).T
test_df = pd.DataFrame(test_arr, columns=["V1", "V2", "V3"])
print(test_df)

Data_Reduction.pca(test_df)

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

Anomaly_Detection.nn(test_df, 10)


def integration(x, y):
    euclidean_distance = Data_Integration.Similarity(x, y).Euclidean_distance()
    manhattan_distance = Data_Integration.Similarity(x, y).Manhattan_distance()
    minkowski_distance = Data_Integration.Similarity(x, y).Minkowski_distance()
    cosine_similarity = Data_Integration.Similarity(x, y).Cosine_similarity()
    cosine_distance = Data_Integration.Similarity(x, y).Cosine_distance()
    pearson_correlation_coefficient = Data_Integration.Correlation(x, y).Pearson_correlation_coefficient()

    print('------------Similarity------------')
    print('Euclidean Distance between a and b is: ', euclidean_distance[0])
    print('Manhattan Distance between a and b is: ', manhattan_distance[0])
    print('Minkowski Distance between a and b is: ', minkowski_distance[0])
    print('Cosine Similarity between a and b is: ', cosine_similarity[0])
    print('Cosine Distance between a and b is: ', cosine_distance[0])
    print('------------Correlation------------')
    print('Pearson Correlation Coefficient between a and b is: ', pearson_correlation_coefficient[0][0])


# Enter the value
t1 = [1, 1, 1, 2, 2, 1, 0, 0, 0, 0]
t2 = [0, 1, 0, 2, 2, 0, 1, 0, 0, 0]
integration(t1, t2)
