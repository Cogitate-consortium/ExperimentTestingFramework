import numpy as np
from scipy import stats


def mean_confidence_interval(data, confidence=0.95, axis=0):
    """
    This function computes the mean and the 95% confidence interval from the data, according to the axis.
    :param data: (numpy array) data on which to compute the confidence interval and mean
    :param confidence: (float) interval of the confidence interval: 0.95 means 95% confidence interval
    :param axis: (int) axis from the data along which to compute the mean and confidence interval
    :return:
    mean (numpy array) mean of the data along specified dimensions
    mean - ci (numpy array) mean of the data minus the confidence interval
    mean + ci (numpy array) mean of the data plus the confidence interval
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, axis=axis), stats.sem(a, axis=axis)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h
