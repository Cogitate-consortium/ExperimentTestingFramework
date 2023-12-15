import mne
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from general_utilities.path_helper_function import list_subjects, load_epochs
from general_utilities.simulate_malfunction import jitter_trials, shuffle_triggers
from general_utilities.data_helper_function import compute_epochs_stat
from simulation.erp_helper_function import generate_single_trials, erp_gaussian_model

np.random.seed(0)


def erp_simulations(config, verbose=False):
    """

    :param config:
    :param verbose:
    :return:
    """
    # ============================================================================================
    # Loading the config:
    with open(config) as f:
        param = json.load(f)
    if verbose:
        print("=" * 40)
        print("Welcome to gmm_from_data")
    # ==================================================================================================================
    # Loading the subjects data:
    # List the subjects:
    subjects_list = list_subjects(Path(param["bids_root"]), prefix="sub-")
    # Loading single subjects data and performing some operations:
    evks = []
    for subject in subjects_list:
        if verbose:
            print("loading sub-{} data".format(subject))
        # Load this subject epochs:
        epochs = load_epochs(param["bids_root"], subject,
                             param["ses"],
                             param["data_type"],
                             param["preprocess_folder"],
                             param["signal"],
                             param["preprocessing_steps"],
                             param["task"])
        # Average the data in each sensor:
        evks.append(epochs.crop(0.05, 0.2).average())

    # Average the evks across subjects:
    grand_avg = mne.grand_average(evks)

    # ==================================================================================================================
    # Fitting the gaussian mixture model:
    if verbose:
        print("")
        print("Fitting the gaussian mixture model: ")
    # Run model fitting with optimization:
    posteriors = erp_gaussian_model(grand_avg, param["optimization_parameters"],
                                    channel=param["channel"], verbose=True, plot=True)

    # =======================================================================================================
    # Simulate single trials according to the passed configs:
    # But first, create tables to store the results:
    jitters_results = []
    shuffle_results = []

    # Prepare path to save the results:
    results_save_root = Path(param["bids_root"], "derivatives", param["signal"])
    if not os.path.isdir(results_save_root):
        os.makedirs(results_save_root)
    # Unroll the malfunc parameters:
    jitter_proportions = np.arange(**param["jitter_trial_props"]).round(decimals=2)
    jitter_durations = np.arange(**param["jitter_durations"]).round(decimals=2)
    shuffle_proportions = np.arange(**param["shuffled_trials_proportion"]).round(decimals=2)

    # Generate time axis:
    times = np.linspace(param["t0"], param["tmax"],
                        int(param["sfreq"] * (param["tmax"] - param["t0"])))

    # Looping through the effect sizes:
    effect_sizes = np.arange(**param["effect_sizes"]).round(decimals=2)
    for fsize in effect_sizes:
        # Generate single trials ERPs:
        epochs = generate_single_trials(times, fsize, posteriors, param["channel"],
                                        param["n_trials_per_cond"],
                                        param["conditions"],
                                        param["sigma"],
                                        param["sfreq"])

        # ==============================================================================================================
        # Apply jitter:
        for jitter_duration in jitter_durations:
            for jitter_proportion in jitter_proportions:
                for i in range(param["n_repeats"]):
                    # Mix up the trials:
                    epochs_jitter = jitter_trials(epochs.copy(), refresh_rate=jitter_duration,
                                                  trials_proportion=jitter_proportion,
                                                  tail=param["jitter_tails"], max_jitter=jitter_duration,
                                                  exact_jitter=True)
                    # Compute the effect size at the single subject level:
                    tstat = compute_epochs_stat(epochs_jitter,
                                                param["components_time_win"]["P1"]["t0"],
                                                param["components_time_win"]["P1"]["tmax"],
                                                metric="mean", channel=param["channel"],
                                                cond_1="cond_1", cond_2="cond_2", stat="tstat")
                    obs_fsize = compute_epochs_stat(epochs_jitter,
                                                    param["components_time_win"]["P1"]["t0"],
                                                    param["components_time_win"]["P1"]["tmax"],
                                                    metric="mean", channel=param["channel"],
                                                    cond_1="cond_1", cond_2="cond_2", stat="fsize")

                    # Compute the effect size and the t-statistic:
                    jitters_results.append(pd.DataFrame({
                        "fsize": fsize,
                        "jitter duration": jitter_duration,
                        "jitter proportion": jitter_proportion,
                        "observed f size": obs_fsize,
                        "t statistic": tstat
                    }, index=[0]))
                    print("fsize: {}".format(fsize))
                    print("     jitter_duration = {}".format(jitter_duration))
                    print("     jitter_proportion = {}".format(jitter_proportion))
                    print("         obs fsize = {}".format(tstat))
                    print("         obs t_stat = {}".format(fsize))

        # ==============================================================================================================
        # Apply label shuffle:
        for shuffle_proportion in shuffle_proportions:
            # Shuffle triggers:
            for i in range(param["n_repeats"]):
                epochs_shuffle = shuffle_triggers(epochs.copy(), trials_proportion=shuffle_proportion)
                # Compute difference between both conditions in P1:
                tstat = compute_epochs_stat(epochs_shuffle,
                                            param["components_time_win"]["P1"]["t0"],
                                            param["components_time_win"]["P1"]["tmax"],
                                            metric="mean", channel=param["channel"],
                                            cond_1="cond_1", cond_2="cond_2", stat="tstat")
                obs_fsize = compute_epochs_stat(epochs_shuffle,
                                                param["components_time_win"]["P1"]["t0"],
                                                param["components_time_win"]["P1"]["tmax"],
                                                metric="mean", channel=param["channel"],
                                                cond_1="cond_1", cond_2="cond_2", stat="fsize")
                shuffle_results.append(pd.DataFrame({
                    "fsize": fsize,
                    "shuffle proportion": shuffle_proportion,
                    "label shuffle proportion": 0,
                    "observed f size": obs_fsize,
                    "t statistic": tstat
                }, index=[0]))
                print("fsize: {}".format(fsize))
                print("     shuffle_proportion = {}".format(shuffle_proportion))
                print("         obs fsize = {}".format(tstat))
                print("         obs t_stat = {}".format(fsize))

    # Concatenate the data frames:
    shuffle_results = pd.concat(shuffle_results).reset_index(drop=True)
    jitters_results = pd.concat(jitters_results).reset_index(drop=True)
    # Save the results of the jitter and shuffle procedures:
    jitter_results_file = Path(results_save_root, "jitter_results.csv")
    shuffle_results_file = Path(results_save_root, "shuffle_results.csv")
    shuffle_results.to_csv(shuffle_results_file)
    jitters_results.to_csv(jitter_results_file)

    return param, jitter_results_file, shuffle_results_file, posteriors


if __name__ == "__main__":
    # Path to the config:
    config_file = "general_config.json"
    components_parameters_file = "components_priors.json"
    # Launch ERP simulation pipeline:
    run_param, jitter_results_path, shuffle_results_path, run_posteriors = \
        erp_simulations("01_simulation_erp_config.json", verbose=False)
