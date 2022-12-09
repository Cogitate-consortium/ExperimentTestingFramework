import numpy as np
from scipy import stats
import statistics


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


def pooled_stdev(sample_1, sample_2):
    """
    This function computes the pooled standard deviation of two samples according to:
    https://en.wikipedia.org/wiki/Pooled_variance
    :param sample_1: (np array or list) first sample for which to compute the pooled variance
    :param sample_2: (np array or list) first sample for which to compute the pooled variance
    :return:
    """
    sd1 = statistics.stdev(sample_1)
    sd2 = statistics.stdev(sample_2)
    n1 = sample_1.shape[0]
    n2 = sample_2.shape[0]
    return np.sqrt(((n1 - 1) * sd1 * sd1 + (n2 - 1) * sd2 * sd2) / (n1 + n2 - 2))
