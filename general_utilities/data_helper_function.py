import numpy as np
from scipy.stats import ttest_ind, sem, t
import statistics
import mne
import pandas as pd


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
    m, se = np.mean(a, axis=axis), sem(a, axis=axis)
    h = se * t.ppf((1 + confidence) / 2., n - 1)
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


def find_nearest_ind(array, value):
    return (np.abs(array - value)).argmin()


def create_epochs(data, ch_names, ch_types, sfreq, conditions, times, n_trials_per_cond):
    print("A")
    # Convert to an mne epochs (as this is what the different jittering functions require:
    ch_names = ch_names
    ch_types = [ch_types] * len(ch_names)
    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sfreq)
    # Generate the events:
    trial_conds = [[cond] * n_trials_per_cond for cond in conditions]
    evts_ids = [[ind] * n_trials_per_cond for ind, cond in enumerate(trial_conds)]
    onsets = (np.linspace(0,
                          (times[-1] - times[0]) * n_trials_per_cond * len(trial_conds),
                          n_trials_per_cond * len(conditions), endpoint=False)
              + np.abs(times[0]))
    # Add all of that into the metadata:
    metadata = pd.DataFrame({
        "condition": np.array([item for sublist in trial_conds for item in sublist]),
        "event_id": np.array([item for sublist in evts_ids for item in sublist]),
        "onset": onsets,
    })
    events = np.column_stack(((metadata["onset"].to_numpy() * sfreq).astype(int),
                              np.zeros(len(metadata), dtype=int),
                              metadata["event_id"].to_numpy().astype(int)))
    events_dict = {cond: ind for ind, cond in enumerate(conditions)}
    # Convert the data to an epochs:
    epochs = mne.EpochsArray(data, info, events=events, event_id=events_dict, tmin=times[0])
    epochs.metadata = metadata

    return epochs


def compute_epochs_stat(epochs, tmin, tmax, metric="mean", channel="E77", cond_1="cond_1", cond_2="cond_2",
                        stat="tstat"):
    """

    :param epochs:
    :param tmin:
    :param tmax:
    :param metric:
    :param channel:
    :param cond_1:
    :param cond_2:
    :param stat:
    :return:
    """
    if metric.lower() == "mean":
        metric_fun = np.mean
    elif metric.lower() == "peak":
        metric_fun = np.max
    else:
        raise Exception("You have passed {} as a metric, but only mean and peak supported!".format(metric))
    # Convert the data to data frame:
    df_epochs = epochs.copy().pick(channel).crop(tmin=tmin, tmax=tmax).to_data_frame(long_format=True)
    # Compute the dependent variable
    dependent_var_df = \
        df_epochs.groupby(['channel', 'epoch'], group_keys=False)[
            "value"].aggregate(metric_fun).reset_index()
    dependent_var_df["condition"] = epochs.metadata["condition"]
    # Compute the required statistic:
    if stat.lower() == "tstat":
        stat, p_value = ttest_ind(dependent_var_df.loc[dependent_var_df["condition"] == cond_1, "value"],
                                  dependent_var_df.loc[dependent_var_df["condition"] == cond_2, "value"])
    elif stat.lower() == "fsize":
        stat = compute_fsize(dependent_var_df.loc[dependent_var_df["condition"] == cond_1, "value"],
                             dependent_var_df.loc[dependent_var_df["condition"] == cond_2, "value"])
    else:
        raise Exception("You have passed {} as a stat, but only fsize and tstat supported!".format(metric))
    return stat


def epochs_fsize(epochs, tmin, tmax, metric="mean", channel="E77", cond_1="cond_1", cond_2="cond_2"):
    """

    :param epochs:
    :param tmin:
    :param tmax:
    :param metric:
    :param channel:
    :param cond_1:
    :param cond_2:
    :return:
    """
    if metric.lower() == "mean":
        func = np.mean
    elif metric.lower() == "peak":
        func = np.max

    # Convert the data to data frame:
    df_epochs = epochs.copy().pick(channel).crop(tmin=tmin, tmax=tmax).to_data_frame(long_format=True)
    # Compute the dependent variable
    dependent_var_df = \
        df_epochs.groupby(['channel', 'epoch'], group_keys=False)[
            "value"].aggregate(func).reset_index()
    # Compute the t-statistic between both conditions:
    stat, p_value = ttest_ind(dependent_var_df.loc[dependent_var_df["condition"] == cond_1, "value"],
                              dependent_var_df.loc[dependent_var_df["condition"] == cond_2, "value"])
    return stat


def compute_t_stat(data_1, data_2, axis=0):
    """
    This function computes a zscore along a specific dimension of a matrix
    :param data_2:
    :param data_1:
    :param axis: (int) which axis along which to compute the zscore for the null distribution
    :return: zscore
    """
    assert isinstance(data_1, np.ndarray) and isinstance(data_2, np.ndarray), "data must be a numpy arrays!"
    # assert data_1.shape == data_2.shape, "data_1 and data_2 must have the same dimensions!"
    # Compute the different values:
    p1 = np.mean(data_1, axis=axis)
    p2 = np.mean(data_2, axis=axis)
    sigma_1 = np.std(data_1, axis=axis)
    sigma_2 = np.std(data_2, axis=axis)
    try:
        return (p1 - p2) / np.sqrt((sigma_1 ** 2) / data_1.shape[0] + (sigma_2 ** 2) / data_2.shape[0])
    except ZeroDivisionError:
        Exception("y standard deviation is equal to 0, can't compute zscore!")


def compute_fsize(data_1, data_2):
    """
    This function computes effect sizes.
    :param data_1:
    :param data_2:
    :return:
    """
    return (np.mean(np.array(data_1)) - np.mean(np.array(data_2))) / np.std(np.concatenate([data_1, data_2]))
