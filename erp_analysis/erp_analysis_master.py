import argparse
import json
import os

import mne
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from general_utilities.path_helper_function import find_files, path_generator, list_subjects, load_epochs
from general_utilities.simulate_malfunction import jitter_trials, shuffle_triggers
from general_utilities.data_helper_function import mean_confidence_interval
import numpy as np

test_subs = ["1", "2"]


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

        # Generate the name of the columns to store everything:
        results = pd.DataFrame()
        for subject in subjects_list:
            start = time()
            # if subject not in test_subs:
            #     continue
            print("Analyzing sub-{}".format(subject))
            for task in param["tasks"]:
                # Load the epochs:
                epochs = load_epochs(param["bids_root"], subject,
                                     param["ses"], param["data_type"], param["preprocess_folder"],
                                     param["signal"], param["preprocess_steps"], task)
                # Load the parameters:
                sim_param_file = Path(param["bids_root"], "derivatives", "preprocessing",
                                      "sub-" + subject, "ses-" + param["ses"],
                                      "eeg", "epoching", param["signal"], param["preprocess_steps"],
                                      "sub-{}_ses-{}_task-{}_desc-config.json".format(subject,
                                                                                      param["ses"],
                                                                                      task))
                with open(sim_param_file) as f:
                    sim_param = json.load(f)
                for jitter_proportion in param["jittered_trials_proportion"]:
                    # Mix up the trials:
                    epochs = jitter_trials(epochs, refresh_rate=param["refresh_rate"],
                                           trials_proportion=jitter_proportion,
                                           tail=param["tail"], max_jitter=param["max_jitter"])
                    # Looping through the label shuffle:
                    for shuffle_proportion in param["shuffled_trials_proportion"]:
                        epochs = shuffle_triggers(epochs, trials_proportion=shuffle_proportion)
                        # Loop through each component:
                        for component in param["components"].keys():
                            # Compute the peak:
                            ch, latency, peak = \
                                epochs.copy().crop(param["components"][component]["t0"],
                                                   param["components"][component]["tmax"]).average().get_peak(
                                    ch_type='eeg',
                                    return_amplitude=True)
                            # Loop through each conditions:
                            cond_means = {"{}-mean".format(cond): None for cond in param["conditions"]}
                            for cond in param["conditions"]:
                                cond_means["{}-mean".format(cond)] = \
                                    np.mean(np.squeeze(epochs[cond].copy().crop(param["components"][component]["t0"],
                                                                                param["components"][component][
                                                                                    "tmax"]).get_data()), axis=(1, 0))
                            # Compute the evoked difference between two conditions:
                            evk_diff = mne.combine_evoked(epochs.average(by_event_type=True), weights=[1, -1])
                            # Append to the results:
                            results = results.append(pd.DataFrame({
                                "task": task,
                                "effect_size": sim_param["effect_size"][component],
                                "subject": subject,
                                "jitter_prop": jitter_proportion,
                                "shuffle_proportion": shuffle_proportion,
                                "component": component,
                                "peak-amp": peak,
                                "peak-latency": latency,
                                "evoked_diff": evk_diff,
                                **cond_means
                            }, index=[0]))
            print("Time for a subject={:.2f}".format(time() - start))

        # Save the results to file:
        results_to_save = results.loc[:, results.columns != "evoked"]
        results_to_save.to_csv(Path(results_save_root, "components_results.csv"))

        # ================================================================================================
        # Plot the effect sizes:
        # Compute the effect sizes of the difference between the two conditions:
        results["cond_diff"] = results["cond_1-mean"] - results["cond_2-mean"]
        effect_sizes_df = pd.DataFrame()
        # Looping through each component again:
        for component in results["component"].unique():
            for effect_size in results["effect_size"].unique():
                for jitter_prop in results["jitter_prop"].unique():
                    for shuffle_prop in results["shuffle_proportion"].unique():
                        # Extract all the data:
                        diff = results.loc[(results["component"] == component)
                                           & (results["effect_size"] == effect_size)
                                           & (results["jitter_prop"] == jitter_prop)
                                           & (results["shuffle_proportion"] == shuffle_prop), "cond_diff"]. \
                            to_numpy()
                        # Compute the effect size:
                        effect_sizes_df = effect_sizes_df.append(pd.DataFrame({
                            "sim_effect_size": effect_size,
                            "component": component,
                            "jitter_prop": jitter_prop,
                            "shuffle_prop": shuffle_prop,
                            "obs_effect_size": np.mean(diff) / np.std(diff)
                        }, index=[0]))
                effect_sizes_df = effect_sizes_df.reset_index(drop=True)
                # Get the data of this component:
                comp_effect_size = effect_sizes_df.loc[
                    (effect_sizes_df["component"] == component) &
                    (effect_sizes_df["sim_effect_size"] == effect_size)].pivot(
                    index="jitter_prop",
                    columns="shuffle_prop",
                    values="obs_effect_size")
                # Generate a heatmap:
                fig, ax = plt.subplots(1)
                sns.heatmap(comp_effect_size, ax=ax)
                ax.set_ylabel("Jittered trials proportion")
                ax.set_xlabel("Label shuffle proportion")
                ax.set_title("Effect size as a function of trial jitters and shuffle"
                             "\n comp={}, fsize={}, jitter={}, shuffle={}".format(component, effect_size,
                                                                                  jitter_prop, shuffle_prop))
                plt.savefig(Path(fig_save_root, "{}_effect_size_{}.png".format(component, effect_size)))
                plt.show()
            # Generating a three d plot:
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
                                               "shuffle_prop"].to_numpy(),
                           effect_sizes_df.loc[(effect_sizes_df["component"] == component) &
                                               (effect_sizes_df[
                                                    "sim_effect_size"] == sim_effect_size),
                                               "obs_effect_size"].to_numpy(),
                           label="Simulated effect size = {}".format(sim_effect_size)
                           )
                ax.set_xlabel('Jitter proportion')
                ax.set_ylabel('Shuffle proportion')
                ax.set_zlabel('Effect size')
            plt.title("Observed effect size as a function of simulated effect size, shuffle and "
                      "jitter proportions")
            plt.legend()
            plt.tight_layout()
            plt.savefig(Path(fig_save_root, "{}_effect_sizes_3d.png".format(component)))
            plt.show()
        # ================================================================================================
        # Plot the peak and latency:
        # Looping through each component again:
        for component in results["component"].unique():
            # Get all the data for this component:
            comp_results = results.loc[results["component"] == component]
            # Extract the data for plotting jitter without shuffle and the other way around:
            jitter_data = comp_results.loc[comp_results["shuffle_proportion"] == 0]
            shuffle_data = comp_results.loc[comp_results["jitter_prop"] == 0]

            # Plot the peak latency and width for each jitter:
            fig, ax = plt.subplots(2)
            sns.boxplot(data=jitter_data, x="jitter_prop", y="peak-amp", orient="v", ax=ax[0])
            sns.boxplot(data=jitter_data, x="jitter_prop", y="peak-latency", orient="v", ax=ax[1])
            ax[0].set_ylabel("Peak amplitude (mV)")
            ax[1].set_ylabel("Peak latency (ms)")
            ax[1].set_xlabel("Jitter proportion")
            ax[0].set_title("{} changes as a function of jitter".format(component))
            plt.tight_layout()
            plt.savefig(Path(fig_save_root, "{}_peaks_jitter.png".format(component)))
            plt.show()
            fig, ax = plt.subplots(2)
            sns.boxplot(data=shuffle_data, x="shuffle_proportion", y="peak-amp", orient="v", ax=ax[0])
            sns.boxplot(data=shuffle_data, x="shuffle_proportion", y="peak-latency", orient="v", ax=ax[1])
            ax[0].set_ylabel("Peak amplitude (mV)")
            ax[1].set_ylabel("Peak latency (ms)")
            ax[1].set_xlabel("Shuffle proportion")
            ax[0].set_title("{} changes as a function of jitter".format(component))
            plt.savefig(Path(fig_save_root, "{}_peaks_shuffle.png".format(component)))
            plt.show()

            # Printing the evoked response:
            for sim_effect_size in jitter_data["effect_size"].unique():
                # Extract these data:
                jitter_f_size_df = jitter_data.loc[jitter_data["effect_size"] == sim_effect_size]
                shuffle_f_size_df = shuffle_data.loc[shuffle_data["effect_size"] == sim_effect_size]
                fig, ax = plt.subplots()
                for jitter_prop in jitter_f_size_df["jitter_prop"].unique():
                    evks = jitter_f_size_df.loc[jitter_f_size_df["jitter_prop"] == jitter_prop, "evoked_diff"].to_list()
                    times = [evk.times for evk in evks]
                    evks_list = [np.squeeze(evk.get_data()) for evk in evks]
                    # Depending on the jittere distribution, the subjects arrays might have different counts of data
                    # points. Adjusting for that:
                    times_intersect = np.sort(np.array(list(set.intersection(*map(set, times)))))
                    for ind, evk in enumerate(evks_list):
                        # Find the overlap:
                        sub_inter, ind_1, ind_2 = np.intersect1d(times[ind], times_intersect, return_indices=True)
                        evks_list[ind] = evks_list[ind][ind_2]
                    # Compute the mean and CIs:
                    avg, low_ci, up_ci = mean_confidence_interval(np.array(evks_list), confidence=0.95, axis=0)
                    ax.plot(times_intersect, avg, label="jitter={}".format(jitter_prop))
                    ax.fill_between(times_intersect, up_ci, low_ci, alpha=.2)
                ax.set_xlim([times_intersect[0], times_intersect[-1]])
                ax.set_ylabel("Amplitude (mV)")
                ax.set_xlabel("Time (sec)")
                plt.legend()
                plt.savefig(Path(fig_save_root, "{}_evoked_fsize_{}_jitter.png".format(component, sim_effect_size)))
                plt.show()

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
                ax.set_ylabel("Amplitude (mV)")
                ax.set_xlabel("Time (sec)")
                ax.set_xlim([times_intersect[0], times_intersect[-1]])
                plt.legend()
                plt.savefig(Path(fig_save_root, "{}_evoked_fsize_{}_shuffle.png".format(component, sim_effect_size)))
                plt.show()
        print("A")
        # Loop through all the


if __name__ == "__main__":
    erp_analysis()
