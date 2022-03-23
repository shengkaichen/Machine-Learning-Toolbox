"""

Cosine Similarity VS Pearson Correlation Coefficient
The two quantities represent two different physical entities.
The cosine similarity computes the similarity between two samples, whereas the Pearson correlation
coefficient computes the correlation between two jointly distributed random variables.
"""

from scipy.spatial import distance
from scipy.stats import linregress
from scipy.stats.stats import pearsonr
import math


class Similarity:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Euclidean Distance is the shortest distance between two points
    def Euclidean_distance(self):
        ed = distance.euclidean(self.x, self.y)

        formula = math.sqrt(sum((a - b) ** 2 for a, b in zip(self.x, self.y)))

        return ed, formula

    # Manhattan Distance is the sum of absolute differences between points across all the dimensions
    def Manhattan_distance(self):
        md = distance.cityblock(self.x, self.y)

        formula = sum(abs(a - b) for a, b in zip(self.x, self.y))

        return md, formula

    # Minkowski Distance generalized form of Euclidean and Manhattan distance
    # p represents the order of the norm
    # Manhattan distance when q = 1, and Euclidean distance when q = 2
    def Minkowski_distance(self):
        p = 3
        md = distance.minkowski(self.x, self.y, p)

        formula = pow(sum(abs(a - b) ** p for a, b in zip(self.x, self.y)), (1 / p))

        return md, formula

    # The different between Cosine Similarity and Cosine Distance is the intuition behind this is
    # that if 2 vectors are perfectly the same then similarity is 1 (angle=0) and
    # thus, distance is 0 (1-1=0)
    def Cosine_similarity(self):
        cs = 1 - distance.cosine(self.x, self.y)

        formula = (sum(a * b for a, b in zip(self.x, self.y))
                   / ((math.sqrt(sum(a * a for a in self.x))) *
                      (math.sqrt(sum(b * b for b in self.y)))))

        return cs, formula

    def Cosine_distance(self):
        cd = distance.cosine(self.x, self.y)

        formula = 1 - self.Cosine_similarity()[1]

        return cd, formula


class Correlation:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def Pearson_correlation_coefficient(self):
        """
        pcc_1 Return:
             Pearson's correlation coefficient, 2-tailed p-value

        pcc_2 Return:
            slope : slope of the regression line
            intercept : intercept of the regression line
            r-value : correlation coefficient
            p-value : two-sided p-value for a hypothesis test whose null hypothesis is that
                      the slope is zero
            stderr : Standard error of the estimate
        """

        pcc1 = pearsonr(self.x, self.y)
        pcc2 = linregress(self.x, self.y)

        return pcc1, pcc2

