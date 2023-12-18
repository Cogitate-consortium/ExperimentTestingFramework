import numpy as np
from scipy.stats import ttest_ind, sem, t
import statistics
import mne
import pandas as pd


def find_nearest_ind(array, value):
    return (np.abs(array - value)).argmin()


def create_epochs(data, ch_names, ch_types, sfreq, conditions, times, n_trials_per_cond):
    """
    This function creates an MNE epochs object.
    :param data: (np array) trial x channel x time, data to populate the epochs with
    :param ch_names: (list of strings) name of the channels. len(ch_names) must be equal to data.shape[1]
    :param ch_types: (list of strings) type of each channel. len(ch_types) must be equal to data.shape[1]
    :param sfreq: (int) sampling frequency of the signal
    :param conditions: (string) name of the condition of that set of epochs
    :param times: (np array) time vector of the epoch data
    :param n_trials_per_cond: (int) number of trials per condition
    :return:
    """
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
    This function computes the t-statistics of the difference between two conditions in a given time window on MNE
    epochs data
    :param epochs: (mne epoch object) contains the data on which to compute the t-stat
    :param tmin: (float) start of time window
    :param tmax: (float) end of time window in which to compute the t-stat
    :param metric: (string) name of the metric to use to compute the aggregate in the time window. Can be "mean" or
    "peak"
    :param channel: (string or list) name of the channel to use to compute the data
    :param cond_1: (string) name of the first condition
    :param cond_2: (string) name of the second condition
    :param stat: (string) name of the statistics to compute. Should be either t-stat or fsize
    :return:
    """
    if metric.lower() == "mean":
        metric_fun = "mean"
    elif metric.lower() == "peak":
        metric_fun = np.max
    else:
        raise Exception("You have passed {} as a metric, but only mean and peak supported!".format(metric))
    # Convert the data to data frame:
    df_epochs = epochs.copy().pick(channel).crop(tmin=tmin, tmax=tmax).to_data_frame(long_format=True)
    # Compute the dependent variable
    dependent_var_df = \
        df_epochs.groupby(['channel', 'epoch'], group_keys=False, observed=False)[
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


def compute_t_stat(data_1, data_2, axis=0):
    """
    This function computes a zscore along a specific dimension of a matrix
    :param data_1: (np array) data of the first condition
    :param data_2: (np array) data of the second condition
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
