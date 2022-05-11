from scipy.spatial import distance
import math


def euclidean_distance(x, y):
    """
    :return: Euclidean Distance and it represents the shortest distance between two points
    """
    ed = distance.euclidean(x, y)

    formula = math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))
    return ed


def manhattan_distance(x, y):
    """
    :return: Manhattan Distance and it represents the sum of absolute differences between points
    across all the dimensions
    """
    md = distance.cityblock(x, y)

    formula = sum(abs(a - b) for a, b in zip(x, y))
    return md


def minkowski_distance(x, y):
    """
    :return: Minkowski Distance and that is generalized form of Euclidean and Manhattan distance
    p represents the order of the norm
    q = 1 is Manhattan distance and q = 2 is Euclidean distance
    """
    p = 3
    md = distance.minkowski(x, y, p)

    formula = pow(sum(abs(a - b) ** p for a, b in zip(x, y)), (1 / p))
    return md


def cosine_similarity(x, y):
    """
    The difference between Cosine Similarity and Cosine Distance is if 2 vectors are
    perfectly the same, the similarity will be 1 (angle=0) and the distance will be 0 (1-1=0)
    """
    cs = 1 - distance.cosine(x, y)

    formula = (sum(a * b for a, b in zip(x, y))
               / ((math.sqrt(sum(a * a for a in x))) *
                  (math.sqrt(sum(b * b for b in y)))))
    return cs


def cosine_distance(x, y):
    cd = distance.cosine(x, y)

    formula = 1 - cosine_similarity(x, y)
    return cd

