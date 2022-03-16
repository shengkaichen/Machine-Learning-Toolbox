from Tools import Data_Transformation
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


# project
raw = pd.read_csv(
    '.../H_M_transaction_2020.csv')
data = Data_Transformation.min_max(raw.iloc[:, 1:5])
newData = Data_Reduction.pca(data)
Anomaly_Detection.nn(newData)

