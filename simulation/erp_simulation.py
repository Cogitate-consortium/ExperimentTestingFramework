import argparse
import mne
import json
from pathlib import Path
import numpy as np
import pandas as pd
from general_utilities.path_helper_function import list_subjects, load_epochs, path_generator
from general_utilities.simulate_malfunction import jitter_trials, shuffle_triggers
from general_utilities.data_helper_function import compute_epochs_stat
from simulation_n.erp_helper_function import generate_single_trials, erp_gaussian_model
from simulation_n.plotter_functions import plot_jitters_sims, plot_shuffle_sims


def erp_simulations(components_priors, sigma=0.5, channels=None, verbose=False, data_type="eeg", signal="erp",
                    components_time_win=None):
    """

    :param components_priors:
    :param sigma:
    :param channels:
    :param verbose:
    :param data_type:
    :param signal:
    :param components_time_win:
    :return:
    """
    # ============================================================================================
    # Parsing input and getting the configs:
    parser = argparse.ArgumentParser(description="Arguments for simulate_epochs")
    parser.add_argument('--config', type=str, default=None,
                        help="Config file for analysis parameters (file name + path)")
    args = parser.parse_args()
    config = args.config
    # Loading the config:
    with open(config) as f:
        param = json.load(f)
    if channels is None:
        channels = "E77"
    if components_time_win is None:
        components_time_win = {"P1": {
            "t0": 0.08,
            "tmax": 0.140
        }}
    if verbose:
        print("=" * 40)
        print("Welcome to gmm_from_data")
    # ==================================================================================================================
    # Loading the subjects data:
    # List the subjects:
    subjects_list = list_subjects(Path(param["object_processing_bids"]["bids_root"]), prefix="sub-")
    # Loading single subjects data and performing some operations:
    evks = []
    for subject in subjects_list:
        if verbose:
            print("loading sub-{} data".format(subject))
        # Load this subject epochs:
        epochs = load_epochs(param["object_processing_bids"]["bids_root"], subject,
                             param["object_processing_bids"]["ses"],
                             data_type,
                             param["object_processing_bids"]["preprocess_folder"],
                             signal,
                             param["object_processing_bids"]["preprocessing_steps"],
                             param["object_processing_bids"]["task"])
        # Average the data in each sensor:
        evks.append(epochs.crop(0.05, 0.2).average())
    # Average the evks across subjects:
    grand_avg = mne.grand_average(evks)

    # ==================================================================================================================
    # Fitting the gaussian mixture model:
    if verbose:
        print("")
        print("Fitting the gaussian mixture model: ")
    # Load the priors:
    if not isinstance(components_priors, dict):
        if isinstance(components_priors, str):
            with open(components_priors) as f:
                components_priors = json.load(f)
        else:
            raise Exception("The component priors must be either a dictionary or a json file string!")
    posteriors = erp_gaussian_model(grand_avg, components_priors, channel=channels, verbose=True, plot=False)

    # =======================================================================================================
    # Simulate single trials according to the passed configs:
    # But first, create tables to store the results:
    jitters_results = pd.DataFrame(columns=["fsize", "jitter duration", "jitter proportion",
                                            "observed f size", "t statistic"])
    shuffle_results = pd.DataFrame(columns=["fsize", "shuffle proportion",
                                            "observed f size", "t statistic"])
    # Load the config
    # Prepare path stuff:
    save_root = Path(param["simulation_bids"]["bids_root"], "derivatives", signal, "population")
    results_save_root = path_generator(save_root,
                                       preprocessing_steps=param["simulation_bids"]["preprocess_steps"],
                                       fig=False, results=True, data=False)
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
        # Generate the ERP:
        epochs = generate_single_trials(times, fsize, posteriors, "E77", param["n_trials_per_cond"],
                                        param["conditions"], sigma,
                                        param["sfreq"])

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
                                                components_time_win["P1"]["t0"],
                                                components_time_win["P1"]["tmax"],
                                                metric="mean", channel="E77",
                                                cond_1="cond_1", cond_2="cond_2", stat="tstat")
                    obs_fsize = compute_epochs_stat(epochs_jitter,
                                                    components_time_win["P1"]["t0"],
                                                    components_time_win["P1"]["tmax"],
                                                    metric="mean", channel="E77",
                                                    cond_1="cond_1", cond_2="cond_2", stat="fsize")

                    # Compute the effect size and the t-statistic:
                    jitters_results = jitters_results.append(pd.DataFrame({
                        "fsize": fsize,
                        "jitter duration": jitter_duration,
                        "jitter proportion": jitter_proportion,
                        "observed f size": obs_fsize,
                        "t statistic": tstat
                    }, index=[0])).reset_index(drop=True)
                    print("fsize: {}".format(fsize))
                    print("     jitter_duration = {}".format(jitter_duration))
                    print("     jitter_proportion = {}".format(jitter_proportion))
                    print("         obs fsize = {}".format(tstat))
                    print("         obs t_stat = {}".format(fsize))

        # Now loop through the different shuffle:
        for shuffle_proportion in shuffle_proportions:
            # Shuffle triggers:
            for i in range(param["n_repeats"]):
                epochs_shuffle = shuffle_triggers(epochs.copy(), trials_proportion=shuffle_proportion)
                # Compute difference between both conditions in P1:
                tstat = compute_epochs_stat(epochs_shuffle,
                                            components_time_win["P1"]["t0"],
                                            components_time_win["P1"]["tmax"],
                                            metric="mean", channel="E77",
                                            cond_1="cond_1", cond_2="cond_2", stat="tstat")
                obs_fsize = compute_epochs_stat(epochs_shuffle,
                                                components_time_win["P1"]["t0"],
                                                components_time_win["P1"]["tmax"],
                                                metric="mean", channel="E77",
                                                cond_1="cond_1", cond_2="cond_2", stat="fsize")
                shuffle_results = shuffle_results.append(pd.DataFrame({
                    "fsize": fsize,
                    "shuffle proportion": shuffle_proportion,
                    "label shuffle proportion": 0,
                    "observed f size": obs_fsize,
                    "t statistic": tstat
                }, index=[0])).reset_index(drop=True)
                print("fsize: {}".format(fsize))
                print("     shuffle_proportion = {}".format(shuffle_proportion))
                print("         obs fsize = {}".format(tstat))
                print("         obs t_stat = {}".format(fsize))
    # Save the results of the jitter and shuffle procedures:
    jitter_results_file = Path(results_save_root, "jitter_results.csv")
    shuffle_results_file = Path(results_save_root, "shuffle_results.csv")
    shuffle_results.to_csv(Path(results_save_root, "shuffle_results.csv"))
    jitters_results.to_csv(Path(results_save_root, "jitter_results.csv"))
    return param, jitter_results_file, shuffle_results_file, posteriors


if __name__ == "__main__":
    run_param, jitter_results_path, shuffle_results_path, run_posteriors = \
        erp_simulations("config\components_priors.json", channels=None, verbose=False, data_type="eeg", signal="erp",
                        components_time_win={"P1": {
                            "t0": 0.08,
                            "tmax": 0.140
                        }})
    # Plot the results of the jitter:
    plot_jitters_sims(jitter_results_path, run_param["simulation_bids"]["bids_root"], "erp",
                      run_param["simulation_bids"]["preprocess_steps"])
    # Plot the results of the jitter:
    plot_shuffle_sims(shuffle_results_path, run_param["simulation_bids"]["bids_root"], "erp",
                      run_param["simulation_bids"]["preprocess_steps"])
