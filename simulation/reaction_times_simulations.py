import argparse
import json
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from general_utilities.path_helper_function import path_generator
from general_utilities.simulate_malfunction import jitter_trials, shuffle_triggers
from general_utilities.data_helper_function import find_nearest_ind, create_epochs, compute_t_stat, compute_fsize

from simulation_n.plotter_functions import plot_jitters_sims, plot_shuffle_sims

show_results = False


def simulate_reaction_times(data_type="beh", signal="reaction_time", mean_rt=450, sigma_rt=20):
    # ============================================================================================
    # Parsing input and getting the configs:
    parser = argparse.ArgumentParser(description="Arguments for simulate_epochs")
    parser.add_argument('--config', type=str, default=None,
                        help="Config file for analysis parameters (file name + path)")
    args = parser.parse_args()
    config = args.config

    # Load the config
    with open(config) as f:
        param = json.load(f)
    # Prepare path stuff:
    save_root = Path(param["simulation_bids"]["bids_root"], "derivatives", signal, "population")
    results_save_root = path_generator(save_root,
                                       preprocessing_steps=param["simulation_bids"]["preprocess_steps"],
                                       fig=False, results=True, data=False)
    # Prepare a pandas data frame to store the final results:
    jitters_results = pd.DataFrame(columns=["fsize", "jitter duration", "jitter proportion",
                                            "observed f size", "t statistic"])
    shuffle_results = pd.DataFrame(columns=["fsize", "shuffle proportion",
                                            "observed f size", "t statistic"])

    # Generate time axis:
    times = np.linspace(param["t0"], param["tmax"],
                        int(param["sfreq"] * (param["tmax"] - param["t0"])))
    # Unroll the malfunc parameters:
    jitter_proportions = np.arange(**param["jitter_trial_props"]).round(decimals=2)
    jitter_durations = np.arange(**param["jitter_durations"]).round(decimals=2)
    shuffle_proportions = np.arange(**param["shuffled_trials_proportion"]).round(decimals=2)
    # Looping through effect sizes:
    effect_sizes = np.arange(**param["effect_sizes"]).round(decimals=2)
    for fsize in effect_sizes:
        # Preallocate array for data:
        data = []
        # Convert the effect size into the delay difference required to achieve this effect size:
        delay_diff = fsize * sigma_rt
        # Generate response time stamps from both conditions:
        for ind, condition in enumerate(param["conditions"]):
            data.append(np.zeros([param["n_trials_per_cond"], 1, times.shape[0]]))
            print((mean_rt + delay_diff * ind) * 10 ** -3)
            # Generate single trials rt:
            cond_rt = np.random.lognormal(mean=np.log((mean_rt + delay_diff * ind) * 10 ** -3),
                                          sigma=sigma_rt * 10 ** -3,
                                          size=param[
                                              "n_trials_per_cond"])
            # Set a 1 at each rt inds:
            for trial in range(data[ind].shape[0]):
                data[ind][trial, 0, find_nearest_ind(times, cond_rt[trial])] = 1
        if show_results:
            plt.hist(times[np.where(data[0] == 1)[2]])
            plt.hist(times[np.where(data[1] == 1)[2]])
            plt.show()
        # Convert data to array:
        data = np.concatenate(data)
        # Convert to an mne epochs (as this is what the different jittering functions require:
        epochs = create_epochs(data, ["response_box"], "eeg", param["sfreq"],
                               param["conditions"], times,
                               param["n_trials_per_cond"])

        # Save the data to file just in case we ever need them again:
        print("=" * 40)
        print("Saving the results")
        save_root = Path(param["simulation_bids"]["bids_root"], "derivatives", "preprocessing",
                         "sub-" + str(1), "ses-" + param["simulation_bids"]["ses"],
                         data_type, "epoching", signal,
                         param["simulation_bids"]["preprocess_steps"])
        if not os.path.isdir(save_root):
            os.makedirs(save_root)
        fname = "sub-{}_ses-{}_task-{}_desc-epoch_beh_-epo.fif".format(1, param["simulation_bids"]["ses"],
                                                                       param["simulation_bids"]["task"])
        epochs.save(Path(save_root, fname), overwrite=True)

        # Loop through the different jitters parameters:
        for jitter_duration in jitter_durations:
            for jitter_proportion in jitter_proportions:
                for i in range(param["n_repeats"]):
                    # Mix up the trials:
                    epochs_jitter = jitter_trials(epochs.copy(), refresh_rate=jitter_duration,
                                                  trials_proportion=jitter_proportion,
                                                  tail=param["jitter_tails"],
                                                  max_jitter=jitter_duration,
                                                  exact_jitter=True)
                    # Get the reaction time for each trial:
                    rt = {}
                    for condition in list(epochs_jitter.metadata["condition"].unique()):
                        inds = np.where(epochs_jitter[condition].get_data() == 1)[2]
                        # Convert those to log transformed reaction times:
                        rt[condition] = np.log(np.array([times[ind] for ind in inds]))

                    # Compute the effect size and the t-statistic:
                    obs_f_size = np.abs(compute_fsize(rt["cond_1"], rt["cond_2"]))
                    t_stat = np.abs(compute_t_stat(rt["cond_1"], rt["cond_2"], axis=0))
                    jitters_results = jitters_results.append(pd.DataFrame({
                        "fsize": fsize,
                        "jitter duration": jitter_duration,
                        "jitter proportion": jitter_proportion,
                        "observed f size": obs_f_size,
                        "t statistic": t_stat
                    }, index=[0])).reset_index(drop=True)
                    print("fsize: {}".format(fsize))
                    print("     jitter_duration = {}".format(jitter_duration))
                    print("     jitter_proportion = {}".format(jitter_proportion))
                    print("         obs fsize = {}".format(obs_f_size))
                    print("         obs t_stat = {}".format(t_stat))

        # Now loop through the different shuffle:
        for shuffle_proportion in shuffle_proportions:
            for i in range(param["n_repeats"]):
                # Shuffle triggers:
                epochs_shuffle = shuffle_triggers(epochs.copy(), trials_proportion=shuffle_proportion)
                # Get the reaction time for each trial:
                rt = {}
                for condition in list(epochs_shuffle.metadata["condition"].unique()):
                    inds = np.where(epochs_shuffle[condition].get_data() == 1)[2]
                    # Convert those to reaction times:
                    rt[condition] = np.array([times[ind] for ind in inds])

                # Compute the effect size and the t-statistic:
                obs_f_size = np.abs(compute_fsize(rt["cond_1"], rt["cond_2"]))
                t_stat = np.abs(compute_t_stat(rt["cond_1"], rt["cond_2"], axis=0))
                shuffle_results = shuffle_results.append(pd.DataFrame({
                    "fsize": fsize,
                    "shuffle proportion": shuffle_proportion,
                    "label shuffle proportion": 0,
                    "observed f size": obs_f_size,
                    "t statistic": t_stat
                }, index=[0])).reset_index(drop=True)
                print("fsize: {}".format(fsize))
                print("     shuffle_proportion = {}".format(shuffle_proportion))
                print("         obs fsize = {}".format(obs_f_size))
                print("         obs t_stat = {}".format(t_stat))
    # Save the results of the jitter and shuffle procedures:
    jitter_results_file = Path(results_save_root, "jitter_results.csv")
    shuffle_results_file = Path(results_save_root, "shuffle_results.csv")
    shuffle_results.to_csv(Path(results_save_root, "shuffle_results.csv"))
    jitters_results.to_csv(Path(results_save_root, "jitter_results.csv"))
    return param, jitter_results_file, shuffle_results_file


if __name__ == "__main__":
    # Simulate the reaction time with jitters and label shuffles:
    run_param, jitter_results_path, shuffle_results_path = \
        simulate_reaction_times(data_type="beh", signal="reaction_time", mean_rt=450, sigma_rt=20)
    # Plot the results of the jitter:
    plot_jitters_sims(jitter_results_path, run_param["simulation_bids"]["bids_root"], "reaction_time",
                      run_param["simulation_bids"]["preprocess_steps"])
    # Plot the results of the jitter:
    plot_shuffle_sims(shuffle_results_path, run_param["simulation_bids"]["bids_root"], "reaction_time",
                      run_param["simulation_bids"]["preprocess_steps"])
