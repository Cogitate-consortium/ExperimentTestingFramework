import argparse
import json
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import statistics
import mne
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from general_utilities.path_helper_function import find_files, path_generator, list_subjects, load_epochs
from general_utilities.simulate_malfunction import jitter_trials, shuffle_triggers
from general_utilities.data_helper_function import mean_confidence_interval
import numpy as np

test_subs = ["1", "2"]
show_plots = False
debug = False


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
                    peaks_diff_fsize, avgs_diff_fsize, latency_std = \
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
                    evk_diff_zscore = diff_zscore(epochs)
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
                        "peaks_diff_fsize": peaks_diff_fsize,
                        "avgs_diff_fsize": avgs_diff_fsize,
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
                peaks_diff_fsize, avgs_diff_fsize, latency_std = \
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
                evk_diff_zscore = diff_zscore(epochs)
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
                    "peaks_diff_fsize": peaks_diff_fsize,
                    "avgs_diff_fsize": avgs_diff_fsize,
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
    :param x: (float) a single number for which to compute the zscore with respect ot the y distribution to the
    :param h0: (1d array) distribution of data with which to compute the std and mean:
    :param axis: (int) which axis along which to compute the zscore for the null distribution
    :return: zscore
    """
    assert isinstance(data_1, np.ndarray) and isinstance(data_2, np.ndarray), "data must be a numpy arrays!"
    assert data_1.shape == data_2.shape, "data_1 and data_2 must have the same dimensions!"
    # Compute the different values:
    p1 = np.mean(data_1, axis=axis)
    p2 = np.mean(data_2, axis=axis)
    sigma_1 = np.std(data_1, axis=axis)
    sigma_2 = np.std(data_2, axis=axis)
    try:
        tstat = (p1 - p2) / np.sqrt((sigma_1 ** 2) / data_1.shape[0] + (sigma_2 ** 2) / data_2.shape[0])
    except ZeroDivisionError:
        Exception("y standard deviation is equal to 0, can't compute zscore!")

    return tstat


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
        ch, latency, peak = trial.copy().crop(tmin, tmax).get_peak(ch_type='eeg', return_amplitude=True)
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
    # Compute the mean peaks difference:
    peaks_diff_mean = np.mean(np.array(peaks[0])) - np.mean(np.array(peaks[1]))
    # Compute the peaks pooled standard deviation:
    peaks_stdev = np.std(np.concatenate(peaks))
    # Same for the mean:
    avgs_diff_mean = np.mean(np.array(avgs[0])) - np.mean(np.array(avgs[1]))
    avg_stdev = np.std(np.concatenate(avgs))
    # Compute effect sizes:
    peaks_diff_fsize = peaks_diff_mean / peaks_stdev
    avgs_diff_fsize = avgs_diff_mean / avg_stdev

    return peaks_diff_fsize, avgs_diff_fsize, latency_std


def erp_analysis():
    # ============================================================================================
    # Parsing input and getting the configs:
    parser = argparse.ArgumentParser(description="Arguments for simulate_epochs")
    parser.add_argument('--config', type=str, default=None,
                        help="Config file for analysis parameters (file name + path)")
    args = parser.parse_args()
    if args.config is None:
        if os.name == "nt":
            configs = find_files(Path(os.getcwd(), "erp_analysis_configs_local"), naming_pattern="*", extension=".json")
        else:
            configs = find_files(Path(os.getcwd(), "configs_hpc"), naming_pattern="*", extension=".json")
    else:
        configs = [args.config]

    # ==================================================================================================================
    # Looping through each config:
    for config in configs:
        # Load the config
        with open(config) as f:
            param = json.load(f)
        # Generate the dir to save the results:
        save_root = Path(param["bids_root"], "derivatives", param["analysis"], "population")
        results_save_root = path_generator(save_root, analysis=param["name"],
                                           preprocessing_steps=param["preprocess_steps"],
                                           fig=False, results=True, data=False)
        fig_save_root = path_generator(save_root, analysis=param["name"],
                                       preprocessing_steps=param["preprocess_steps"],
                                       fig=False, results=True, data=False)
        # List the subjects:
        subjects_list = list_subjects(Path(param["bids_root"], "derivatives", "preprocessing"), prefix="sub-")
        path_info = {
            "bids_root": param["bids_root"],
            "data_type": param["data_type"],
            "ses": param["ses"],
            "preprocess_steps": param["preprocess_steps"],
            "preprocess_folder": param["preprocess_folder"],
            "signal": param["signal"],
            "tasks": param["tasks"]
        }
        results = Parallel(n_jobs=param["njobs"])(delayed(subject_erp_wrapper)(
            subjects_list[i], path_info,
            param["jitter_durations"],
            param["jittered_trials_proportion"],
            param["tail"],
            param["components"],
            param["conditions"],
            param["shuffled_trials_proportion"]
        ) for i in tqdm(range(len(subjects_list))))
        results = pd.concat(results, ignore_index=True)
        # Save the results to file:
        results_to_save = results.loc[:, results.columns != "evoked"]
        results_to_save = results_to_save.loc[:, results_to_save.columns != "evoked_diff"]
        results_to_save.to_csv(Path(results_save_root, "components_results.csv"))

        # ================================================================================================
        # Plot the effect sizes:
        # Compute the effect sizes of the difference between the two conditions:
        results["cond_diff"] = results["cond_1-mean"] - results["cond_2-mean"]
        effect_sizes_df = pd.DataFrame()
        # Looping through each component again to plot the effect of jitter:
        for component in results["component"].unique():
            for effect_size in results["effect_size"].unique():
                for jitter_prop in results["jitter_prop"].unique():
                    for jitter_duration in results["jitter_duration"].unique():
                        # Extract all the data:
                        diff_avg = results.loc[(results["component"] == component)
                                               & (results["effect_size"] == effect_size)
                                               & (results["jitter_prop"] == jitter_prop)
                                               & (results["jitter_duration"] == jitter_duration), "avgs_diff_fsize"]. \
                            to_numpy()
                        diff_peak = results.loc[(results["component"] == component)
                                                & (results["effect_size"] == effect_size)
                                                & (results["jitter_prop"] == jitter_prop)
                                                & (results["jitter_duration"] == jitter_duration), "peaks_diff_fsize"]. \
                            to_numpy()
                        latencies_std = results.loc[(results["component"] == component)
                                                    & (results["effect_size"] == effect_size)
                                                    & (results["jitter_prop"] == jitter_prop)
                                                    & (results["jitter_duration"] == jitter_duration), "latency_std"]. \
                            to_numpy()
                        # Compute the effect size:
                        effect_sizes_df = effect_sizes_df.append(pd.DataFrame({
                            "sim_effect_size": effect_size,
                            "component": component,
                            "jitter_prop": jitter_prop,
                            "jitter_duration": jitter_duration,
                            "avg_fsize": np.mean(diff_avg),
                            "peak_fsize": np.mean(diff_peak),
                            "latencies_std": np.mean(latencies_std)
                        }, index=[0]))
                effect_sizes_df = effect_sizes_df.reset_index(drop=True)

            # ============================================================================================
            # Plot effect of jitter:
            # First plot effect on peak amplitude:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            for sim_effect_size in effect_sizes_df["sim_effect_size"].unique():
                ax.scatter(effect_sizes_df.loc[(effect_sizes_df["component"] == component) &
                                               (effect_sizes_df[
                                                    "sim_effect_size"] == sim_effect_size),
                                               "jitter_prop"].to_numpy(),
                           effect_sizes_df.loc[(effect_sizes_df["component"] == component) &
                                               (effect_sizes_df[
                                                    "sim_effect_size"] == sim_effect_size),
                                               "jitter_duration"].to_numpy(),
                           effect_sizes_df.loc[(effect_sizes_df["component"] == component) &
                                               (effect_sizes_df[
                                                    "sim_effect_size"] == sim_effect_size),
                                               "peak_fsize"].to_numpy(),
                           label="Simulated effect size = {}".format(sim_effect_size)
                           )
                ax.set_xlabel('Jitter proportion')
                ax.set_ylabel('Jitter duration')
                ax.set_zlabel('Observed effect size')
            plt.title("Observed jitter effect on peak effect size")
            plt.legend()
            plt.tight_layout()
            plt.savefig(Path(fig_save_root, "{}_peak_effect_sizes_3d.png".format(component)))
            if show_plots:
                plt.show()
            else:
                plt.close()

            # Same for the mean amplitude:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            for sim_effect_size in effect_sizes_df["sim_effect_size"].unique():
                ax.scatter(effect_sizes_df.loc[(effect_sizes_df["component"] == component) &
                                               (effect_sizes_df[
                                                    "sim_effect_size"] == sim_effect_size),
                                               "jitter_prop"].to_numpy(),
                           effect_sizes_df.loc[(effect_sizes_df["component"] == component) &
                                               (effect_sizes_df[
                                                    "sim_effect_size"] == sim_effect_size),
                                               "jitter_duration"].to_numpy(),
                           effect_sizes_df.loc[(effect_sizes_df["component"] == component) &
                                               (effect_sizes_df[
                                                    "sim_effect_size"] == sim_effect_size),
                                               "avg_fsize"].to_numpy(),
                           label="Simulated effect size = {}".format(sim_effect_size)
                           )
                ax.set_xlabel('Jitter proportion')
                ax.set_ylabel('Jitter duration')
                ax.set_zlabel('Observed effect size')
            plt.title("Observed jitter effect on average effect size")
            plt.legend()
            plt.tight_layout()
            plt.savefig(Path(fig_save_root, "{}_average_effect_sizes_3d.png".format(component)))
            if show_plots:
                plt.show()
            else:
                plt.close()

            # Finally for the latency:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            for sim_effect_size in effect_sizes_df["sim_effect_size"].unique():
                ax.scatter(effect_sizes_df.loc[(effect_sizes_df["component"] == component) &
                                               (effect_sizes_df[
                                                    "sim_effect_size"] == sim_effect_size),
                                               "jitter_prop"].to_numpy(),
                           effect_sizes_df.loc[(effect_sizes_df["component"] == component) &
                                               (effect_sizes_df[
                                                    "sim_effect_size"] == sim_effect_size),
                                               "jitter_duration"].to_numpy(),
                           effect_sizes_df.loc[(effect_sizes_df["component"] == component) &
                                               (effect_sizes_df[
                                                    "sim_effect_size"] == sim_effect_size),
                                               "latencies_std"].to_numpy(),
                           label="Simulated effect size = {}".format(sim_effect_size)
                           )
                ax.set_xlabel('Jitter proportion')
                ax.set_ylabel('Jitter duration')
                ax.set_zlabel('Observed {} peak latency std'.format(component))
            plt.title("Observed jitter effect on {} peak latency std".format(component))
            plt.legend()
            plt.tight_layout()
            plt.savefig(Path(fig_save_root, "{}_peak_latency_std_3d.png".format(component)))
            if show_plots:
                plt.show()
            else:
                plt.close()

        # ============================================================================================
        # Plot effect of shuffle:
        for component in results["component"].unique():
            effect_sizes_df = pd.DataFrame()
            for effect_size in results["effect_size"].unique():
                for shuffle_prop in results["shuffle_proportion"].unique():
                    # Extract all the data:
                    diff_avg = results.loc[(results["component"] == component)
                                           & (results["effect_size"] == effect_size)
                                           & (results["shuffle_proportion"] == shuffle_prop),
                                           "avgs_diff_fsize"]. \
                        to_numpy()
                    diff_peak = results.loc[(results["component"] == component)
                                            & (results["effect_size"] == effect_size)
                                            & (results["shuffle_proportion"] == shuffle_prop),
                                            "peaks_diff_fsize"]. \
                        to_numpy()
                    # Compute the effect size:
                    effect_sizes_df = effect_sizes_df.append(pd.DataFrame({
                        "sim_effect_size": effect_size,
                        "component": component,
                        "shuffle_proportion": shuffle_prop,
                        "avg_fsize": np.mean(diff_avg),
                        "peak_fsize": np.mean(diff_peak)
                    }, index=[0]))
                effect_sizes_df = effect_sizes_df.reset_index(drop=True)
            # Convert long to wide table to generate a heatmap:
            avg_effect_size = effect_sizes_df.loc[effect_sizes_df["component"] == component].pivot(
                index="shuffle_proportion",
                columns="sim_effect_size",
                values="avg_fsize")
            # Generate a heatmap:
            fig, ax = plt.subplots(1)
            sns.heatmap(avg_effect_size, ax=ax)
            ax.set_ylabel("Proportion of shuffle trials")
            ax.set_xlabel("Simulated Effect sizes")
            ax.set_title("Average effect size as a function of label shuffles"
                         "\n comp={}".format(component))
            plt.savefig(Path(fig_save_root, "{}_shuffle_average.png".format(component)))
            if show_plots:
                plt.show()
            else:
                plt.close()
            # Convert long to wide table to generate a heatmap:
            avg_effect_size = effect_sizes_df.loc[effect_sizes_df["component"] == component].pivot(
                index="shuffle_proportion",
                columns="sim_effect_size",
                values="peak_fsize")
            # Generate a heatmap:
            fig, ax = plt.subplots(1)
            sns.heatmap(avg_effect_size, ax=ax)
            ax.set_ylabel("Proportion of shuffle trials")
            ax.set_xlabel("Simulated Effect sizes")
            ax.set_title("Peak effect size as a function of label shuffles"
                         "\n comp={}".format(component))
            plt.savefig(Path(fig_save_root, "{}_shuffle_peak.png".format(component)))
            if show_plots:
                plt.show()
            else:
                plt.close()

        # ================================================================================================
        # Plot time series:
        for component in results["component"].unique():
            # Get all the data for this component:
            comp_results = results.loc[results["component"] == component]
            # Extract the data for plotting jitter without shuffle and the other way around:
            jitter_data = comp_results.loc[comp_results["shuffle_proportion"] == 0]
            shuffle_data = comp_results.loc[comp_results["jitter_prop"] == 0]

            for sim_effect_size in jitter_data["effect_size"].unique():
                # Extract these data:
                jitter_f_size_df = jitter_data.loc[jitter_data["effect_size"] == sim_effect_size]
                shuffle_f_size_df = shuffle_data.loc[shuffle_data["effect_size"] == sim_effect_size]

                # ============================================================================================
                # Printing the evoked response differences:

                # =========================================
                # Plot effect of jitter on evoked difference:
                for jitter_dur in jitter_f_size_df["jitter_duration"].unique():
                    jitter_dur_df = jitter_f_size_df.loc[
                        jitter_f_size_df["jitter_duration"] == jitter_dur]
                    fig, ax = plt.subplots()
                    for jitter_prop in jitter_dur_df["jitter_prop"].unique():
                        evks = jitter_dur_df.loc[
                            jitter_dur_df["jitter_prop"] == jitter_prop, "evoked_diff"].to_list()
                        times = [evk.times for evk in evks]
                        evks_list = [np.squeeze(evk.get_data()) for evk in evks]
                        # Depending on the jittere distribution, the subjects arrays might have different counts of data
                        # points. Adjusting for that:
                        times_intersect = np.sort(np.array(list(set.intersection(*map(set, times)))))
                        for ind_evk, evk in enumerate(evks_list):
                            # Find the overlap:
                            sub_inter, ind_1, ind_2 = np.intersect1d(times[ind_evk], times_intersect,
                                                                     return_indices=True)
                            evks_list[ind_evk] = evks_list[ind_evk][ind_2]
                        # Compute the mean and CIs:
                        avg, low_ci, up_ci = mean_confidence_interval(np.array(evks_list), confidence=0.95, axis=0)
                        ax.plot(times_intersect, avg, label="jitter={}".format(jitter_prop))
                        ax.fill_between(times_intersect, up_ci, low_ci, alpha=.2)
                    ax.set_xlim([times_intersect[0], times_intersect[-1]])
                    ax.set_ylabel("T statistic")
                    ax.set_xlabel("Time (sec)")
                    plt.legend()
                    plt.savefig(Path(fig_save_root, "{}_evoked_fsize_{}_jitter_dur_{}.png".format(component,
                                                                                                  sim_effect_size,
                                                                                                  jitter_dur)))
                    if show_plots:
                        plt.show()
                    else:
                        plt.close()

                # =========================================
                # Plot effect of shuffle on evoked difference:
                fig, ax = plt.subplots()
                for shuffle_prop in shuffle_f_size_df["shuffle_proportion"].unique():
                    evks = shuffle_f_size_df.loc[shuffle_f_size_df["shuffle_proportion"] == shuffle_prop,
                                                 "evoked_diff"].to_list()
                    times = evks[0].times
                    evks = np.array([np.squeeze(evk.get_data()) for evk in evks])
                    # Compute the mean and CIs:
                    avg, low_ci, up_ci = mean_confidence_interval(evks, confidence=0.95, axis=0)
                    ax.plot(times, avg, label="shuffle={}".format(shuffle_prop))
                    ax.fill_between(times, up_ci, low_ci, alpha=.2)
                ax.set_ylabel("T statistic")
                ax.set_xlabel("Time (sec)")
                ax.set_xlim([times_intersect[0], times_intersect[-1]])
                plt.legend()
                plt.savefig(Path(fig_save_root, "{}_evoked_fsize_{}_shuffle.png".format(component, sim_effect_size)))
                if show_plots:
                    plt.show()
                else:
                    plt.close()

                # ============================================================================================
                # Plotting the evoked responses per condition
                for jitter_duration in jitter_f_size_df["jitter_duration"].unique():
                    jitter_dur_df = jitter_f_size_df.loc[jitter_f_size_df["jitter_duration"] == jitter_duration]
                    # Looping through each jitter proportion:
                    for jitter_prop in jitter_dur_df["jitter_prop"].unique():
                        fig, ax = plt.subplots()
                        evks = jitter_dur_df.loc[jitter_dur_df["jitter_prop"] == jitter_prop,
                                                 "evoked"].to_list()
                        # Combine subjects data:
                        sub_evks = [[], []]
                        for evk in evks:
                            sub_evks[0].append(evk[0])
                            sub_evks[1].append(evk[1])
                        times = [evk[0].times for evk in evks]
                        # Depending on the jittere distribution, the subjects arrays might have different counts of data
                        # points. Adjusting for that:
                        times_intersect = np.sort(np.array(list(set.intersection(*map(set, times)))))
                        for ind, evk in enumerate(sub_evks):
                            # Find the overlap:
                            sub_inter, ind_1, ind_2 = np.intersect1d(times[ind], times_intersect, return_indices=True)
                            sub_evks[ind][0].data = sub_evks[ind][0].data[:, ind_2]
                            sub_evks[ind][1].data = sub_evks[ind][1].data[:, ind_2]
                        # Compute the mean and CIs:
                        avg, low_ci, up_ci = mean_confidence_interval(
                            np.array([np.squeeze(evk.data) for evk in sub_evks[0]]),
                            confidence=0.95, axis=0)
                        ax.plot(times_intersect, avg, label="cond_1")
                        ax.fill_between(times_intersect, up_ci, low_ci, alpha=.2)
                        avg, low_ci, up_ci = mean_confidence_interval(
                            np.array([np.squeeze(evk.data) for evk in sub_evks[1]]),
                            confidence=0.95, axis=0)
                        ax.plot(times_intersect, avg, label="cond_2")
                        ax.fill_between(times_intersect, up_ci, low_ci, alpha=.2)
                        ax.set_xlim([times_intersect[0], times_intersect[-1]])
                        ax.set_ylabel("Amplitude (mV)")
                        ax.set_xlabel("Time (sec)")
                        plt.legend()
                        plt.savefig(Path(fig_save_root, "{}_evoked_fsize_{}_jitter_dur_{}_jitter_prop_{}.png".format(
                            component,
                            sim_effect_size,
                            jitter_duration,
                            jitter_prop)))
                        if show_plots:
                            plt.show()
                        else:
                            plt.close()

        print("Finished!")


if __name__ == "__main__":
    erp_analysis()
