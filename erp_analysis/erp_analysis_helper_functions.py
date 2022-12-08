import json
import mne
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
from general_utilities.path_helper_function import load_epochs
from general_utilities.simulate_malfunction import jitter_trials, shuffle_triggers
import numpy as np
params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)


class MidpointNormalize(Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_heatmap(df, index, column, values, xlabel="", ylabel="", zlabel="", title="", cmap="RdYlBu_r", midpoint=1.96):
    """

    :param df:
    :param index:
    :param column:
    :param values:
    :param xlabel:
    :param ylabel:
    :param title:
    :return:
    """
    # Convert long to wide table to generate a heatmap:
    avg_effect_size = df.pivot(index=index, columns=column, values=values)
    # Add color map
    norm = MidpointNormalize(vmin=np.min(avg_effect_size.to_numpy()), vmax=np.max(avg_effect_size.max()),
                             midpoint=midpoint)
    # Generate a heatmap:
    fig, ax = plt.subplots(1)
    sns.heatmap(avg_effect_size, ax=ax, cmap=cmap, norm=norm,
                cbar_kws={'label': 'colorbar title'})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_3d(ax, x, y, z, label, xlabel="", y_label="", zlabel="", alpha=.2):
    """

    :param ax:
    :param x:
    :param y:
    :param z:
    :param label:
    :param xlabel:
    :param y_label:
    :param zlabel:
    :param alpha:
    :return:
    """
    # Add the scatter:
    ax.scatter(x, y, z, label=label)
    # Add Surface:
    ax.plot_trisurf(x, y, z, alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(y_label)
    ax.set_zlabel(zlabel)
    return ax


def subject_erp_wrapper(subject, path_info, jitter_durations, jitter_trial_props, jitter_tails, components_dict,
                        conditions, shuffle_props):
    results = pd.DataFrame()
    for task in path_info["tasks"]:
        # Load the epochs:
        epochs = load_epochs(path_info["bids_root"], subject,
                             path_info["ses"], path_info["data_type"], path_info["preprocess_folder"],
                             path_info["signal"], path_info["preprocess_steps"], task)
        # Load the parameters:
        sim_param_file = Path(path_info["bids_root"], "derivatives", "preprocessing",
                              "sub-" + subject, "ses-" + path_info["ses"],
                              "eeg", "epoching", path_info["signal"], path_info["preprocess_steps"],
                              "sub-{}_ses-{}_task-{}_desc-config.json".format(subject,
                                                                              path_info["ses"],
                                                                              task))
        with open(sim_param_file) as f:
            sim_param = json.load(f)

        for jitter_duration in jitter_durations:
            for jitter_proportion in jitter_trial_props:
                # Mix up the trials:
                epochs_jitter = jitter_trials(epochs.copy(), refresh_rate=jitter_duration,
                                              trials_proportion=jitter_proportion,
                                              tail=jitter_tails, max_jitter=jitter_duration,
                                              exact_jitter=True)

                # Loop through each component:
                for component in components_dict.keys():
                    # Compute the peak:
                    ch, latency, peak = \
                        epochs_jitter.copy().crop(components_dict[component]["t0"],
                                                  components_dict[component]["tmax"]).average().get_peak(
                            ch_type='eeg',
                            return_amplitude=True)
                    # Compute the effect size at the single subject level:
                    peaks_fsize, avgs_fsize, peaks_tstat, avgs_tstat, latency_std = \
                        single_trials_fsizes(epochs_jitter, components_dict[component]["t0"],
                                             components_dict[component]["tmax"])
                    # Loop through each conditions:
                    cond_means = {"{}-mean".format(cond): None for cond in conditions}
                    for cond in conditions:
                        cond_means["{}-mean".format(cond)] = \
                            np.mean(np.squeeze(
                                epochs_jitter[cond].copy().crop(components_dict[component]["t0"],
                                                                components_dict[component][
                                                                    "tmax"]).get_data()), axis=(1, 0))
                    # Compute the zscore difference between two conditions:
                    evk_diff_zscore = diff_zscore(epochs_jitter)
                    # Compute the evoked response for each condition:
                    evk = epochs_jitter.average(by_event_type=True)
                    # Append to the results:
                    results = results.append(pd.DataFrame({
                        "task": task,
                        "effect_size": sim_param["effect_size"][component],
                        "subject": subject,
                        "jitter_duration": jitter_duration,
                        "jitter_prop": jitter_proportion,
                        "shuffle_proportion": 0,
                        "component": component,
                        "peaks_fsize": peaks_fsize,
                        "avgs_fsize": avgs_fsize,
                        "peaks_tstat": peaks_tstat,
                        "avgs_tstat": avgs_tstat,
                        "latency_std": latency_std,
                        "peak-amp": peak,
                        "peak-latency": latency,
                        "evoked": [evk],
                        "evoked_diff": evk_diff_zscore,
                        **cond_means
                    }, index=[0]))
        # Looping through the label shuffle:
        for shuffle_proportion in shuffle_props:
            epochs_shuffle = shuffle_triggers(epochs.copy(), trials_proportion=shuffle_proportion)
            # Loop through each component:
            for component in components_dict.keys():
                # Compute the peak:
                ch, latency, peak = \
                    epochs_shuffle.copy().crop(components_dict[component]["t0"],
                                               components_dict[component]["tmax"]).average().get_peak(
                        ch_type='eeg',
                        return_amplitude=True)
                # Compute the effect size at the single subject level:
                peaks_fsize, avgs_fsize, peaks_tstat, avgs_tstat, latency_std = \
                    single_trials_fsizes(epochs_shuffle, components_dict[component]["t0"],
                                         components_dict[component]["tmax"])
                # Loop through each conditions:
                cond_means = {"{}-mean".format(cond): None for cond in conditions}
                for cond in conditions:
                    cond_means["{}-mean".format(cond)] = \
                        np.mean(np.squeeze(
                            epochs_shuffle[cond].copy().crop(components_dict[component]["t0"],
                                                             components_dict[component][
                                                                 "tmax"]).get_data()), axis=(1, 0))
                # Compute the zscore difference between two conditions:
                evk_diff_zscore = diff_zscore(epochs_shuffle)
                evk = epochs_shuffle.average(by_event_type=True)
                # Append to the results:
                results = results.append(pd.DataFrame({
                    "task": task,
                    "effect_size": sim_param["effect_size"][component],
                    "subject": subject,
                    "jitter_duration": 0,
                    "jitter_prop": 0,
                    "shuffle_proportion": shuffle_proportion,
                    "component": component,
                    "peaks_fsize": peaks_fsize,
                    "avgs_fsize": avgs_fsize,
                    "peaks_tstat": peaks_tstat,
                    "avgs_tstat": avgs_tstat,
                    "latency_std": latency_std,
                    "peak-amp": peak,
                    "peak-latency": latency,
                    "evoked": [evk],
                    "evoked_diff": evk_diff_zscore,
                    **cond_means
                }, index=[0]))
    return results


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


def diff_zscore(epochs):
    """
    This function z scores the epochs per condition and then takes the difference
    :param epochs: (mne epochs object)
    :return:
    """
    data = []
    # Looping through each condition:
    for condition in epochs.event_id.keys():
        data.append(np.squeeze(epochs[condition].get_data()))
    # Now compute the difference between the two conditions:
    if len(data) != 2:
        raise Exception("There were either more or less than two conditions! Not supported")
    else:
        zscore_diff = compute_t_stat(*data, axis=0)
    # Convert to evoked:
    return mne.EvokedArray(np.expand_dims(zscore_diff, axis=0), epochs.info, tmin=epochs.times[0],
                           nave=epochs.events.shape[0], comment="zscore")


def compute_fsize(data_1, data_2):
    """
    This function computes effect sizes.
    :param data_1:
    :param data_2:
    :return:
    """
    return (np.mean(np.array(data_1)) - np.mean(np.array(data_2))) / np.std(np.concatenate([data_1, data_2]))


def single_trials_fsizes(epochs, tmin, tmax):
    # Get the conditions:
    conds = list(set(epochs.events[:, 2]))
    if len(conds) > 2:
        raise Exception("There are more than two conditions. Not supported yet!")
    latencies, peaks, avgs = {str(cond): [] for cond in conds}, \
                             {str(cond): [] for cond in conds}, \
                             {str(cond): [] for cond in conds}
    # Looping through each trial:
    for ix, trial in enumerate(epochs.iter_evoked()):
        ch, latency, peak = trial.get_peak(ch_type='eeg', return_amplitude=True,
                                           tmin=tmin, tmax=tmax)
        avg = np.mean(trial.copy().crop(tmin, tmax).data)
        latencies[trial.comment].append(latency)
        peaks[trial.comment].append(peak)
        avgs[trial.comment].append(avg)
    # Convert to list of lists:
    latencies = [np.array(values) for values in latencies.values()]
    peaks = [np.array(values) for values in peaks.values()]
    avgs = [np.array(values) for values in avgs.values()]
    # Compute the variables of interest:
    # Compute the latency standard deviation:
    latency_std = np.std(np.concatenate(latencies))
    # Compute the peaks and average effect sizes:
    peaks_fsize = compute_fsize(peaks[0], peaks[1])
    avgs_fsize = compute_fsize(avgs[0], avgs[1])
    # Compute the t statistics:
    peaks_tstat = compute_t_stat(peaks[0], peaks[1], axis=0)
    avgs_tstat = compute_t_stat(avgs[0], avgs[1], axis=0)

    return peaks_fsize, avgs_fsize, peaks_tstat, avgs_tstat, latency_std