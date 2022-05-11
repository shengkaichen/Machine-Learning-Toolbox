"""
Notes:
 Cosine Similarity vs Pearson Correlation Coefficient
  The two quantities represent two different physical entities.
  The cosine similarity computes the similarity between two samples, whereas the Pearson correlation
  coefficient computes the correlation between two jointly distributed random variables.
"""

from scipy.stats import linregress
from scipy.stats.stats import pearsonr


def pcc(x, y):  # pearson correlation coefficient
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

    pcc1 = pearsonr(x, y)
    pcc2 = linregress(x, y)
    return pcc1, pcc2
