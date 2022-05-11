"""
Z-score normalization: Features will be rescaled so that they’ll have the properties of a standard normal distribution
Min-max normalization: The traditional method of rescaling, it transforms a feature such that all of its values fall
in a range between 0 and 1:
"""

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def z_score(data):
    return pd.DataFrame(StandardScaler().fit_transform(data))


def min_max(data):
    return pd.DataFrame(MinMaxScaler().fit_transform(data))

