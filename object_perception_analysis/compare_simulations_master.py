import argparse
from pathlib import Path
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from general_utilities.path_helper_function import find_files, path_generator
from general_utilities.data_helper_function import mean_confidence_interval

t0 = 0
tmax = 0.5
fig_size = [15, 20]
SMALL_SIZE = 22
MEDIUM_SIZE = 24
BIGGER_SIZE = 26
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the fi


def compare_simulated_jitters():
    parser = argparse.ArgumentParser(description="Arguments for MVPA analysis across all subjects")
    parser.add_argument('--config', type=str, default=None,
                        help="Config file for analysis parameters (file name + path)")
    args = parser.parse_args()
    if args.config is None:
        if os.name == "nt":
            configs = find_files(Path(os.getcwd(), "configs_local"), naming_pattern="*", extension=".json")
        else:
            configs = find_files(Path(os.getcwd(), "configs_hpc"), naming_pattern="*", extension=".json")
    else:
        configs = [args.config]

    results = {}
    configs_df = pd.DataFrame()
    # Looping through all config:
    for config in configs:
        # Load the config:
        with open(config) as f:
            config = json.load(f)

        # Get the results for this particular config:
        save_root = Path(config["bids_root"], "derivatives", config["analysis"], "population")
        results_save_root = path_generator(save_root, analysis=config["name"],
                                           preprocessing_steps=config["preprocess_steps"],
                                           fig=False, results=True, data=False)
        with open(Path(results_save_root, "sub-population_decoding_scores.pkl"), 'rb') as f:
            results[config["name"]] = pickle.load(f)
        # Get the maximal decoding values observed:
        max_decoding = np.max([np.max(results[config["name"]][key], axis=1) for key in results[config["name"]].keys()])
        # Extract the config relevant info as a dataframe:
        if config["trigger_jitter_parameter"] is not None:
            configs_df = configs_df.append(pd.DataFrame({
                "name": config["name"],
                "jitter_amp_ms": config["trigger_jitter_parameter"]["jitter_amp_ms"],
                "trials_proportion": config["trigger_jitter_parameter"]["trials_proportion"],
                "max_decoding": max_decoding
            }, index=[0]))
        else:
            configs_df = configs_df.append(pd.DataFrame({
                "name": config["name"],
                "jitter_amp_ms": 0,
                "trials_proportion": 0,
                "max_decoding": max_decoding
            }, index=[0]))
    configs_df = configs_df.reset_index(drop=True)

    # Create path to save the results:
    save_root = Path(config["bids_root"], "derivatives", config["analysis"], "simulation_comparisons")
    if not os.path.isdir(save_root):
        os.makedirs(save_root)

    # =======================================================================
    # 1. Plot the decoding results averaged across labels:
    fig, ax = plt.subplots(figsize=fig_size)
    for ind, key in enumerate(results.keys()):
        # Get the parameter:
        param = configs_df.loc[configs_df["name"] == key, ["jitter_amp_ms", "trials_proportion"]]
        label = "{}ms, {}%trials".format(param["jitter_amp_ms"].values[0], param["trials_proportion"].values[0] * 100)
        # Average across all the different labels:
        data = np.mean(np.array([results[key][label] for label in results[key].keys()]), axis=0)
        avg, low_ci, up_ci = mean_confidence_interval(data)
        times = np.linspace(t0, tmax, num=avg.shape[-1])
        ax.plot(times, avg, label=label)
        ax.fill_between(times, up_ci, low_ci, alpha=.2)
        ax.set_xlabel('Times (s)')
        ax.set_ylabel('Accuracy')  # Area Under the Curve
    ax.legend()
    ax.axvline(.0, color='k', linestyle='-')
    ax.set_title('Average decoding accuracy')
    # Save the figure to a file:
    plt.savefig(Path(save_root, "average_decoding_scores.png"))

    # =======================================================================
    # 2. Plot the decoding results separately for each label:
    labels = list(results[list(results.keys())[0]].keys())
    for label in labels:
        fig, ax = plt.subplots(figsize=fig_size)
        for ind, key in enumerate(results.keys()):
            # Average across all the different labels:
            param = configs_df.loc[configs_df["name"] == key, ["jitter_amp_ms", "trials_proportion"]]
            line_label = "{}ms, {}%trials".format(param["jitter_amp_ms"].values[0],
                                                  param["trials_proportion"].values[0] * 100)
            avg, low_ci, up_ci = mean_confidence_interval(results[key][label])
            times = np.linspace(t0, tmax, num=avg.shape[-1])
            ax.plot(times, avg, label=line_label)
            ax.fill_between(times, up_ci, low_ci, alpha=.2)
            ax.set_xlabel('Times (s)')
            ax.set_ylabel('Accuracy')  # Area Under the Curve
        ax.legend()
        ax.axvline(.0, color='k', linestyle='-')
        ax.set_title('{} decoding accuracy'.format(label))
        # Save the figure to a file:
        plt.savefig(Path(save_root, "{}_decoding_scores.png".format(label)))

    # =======================================================================
    # 3. Plotting the max decoding accuracy as a function of proportion of jitter separately for each jitter duration:
    for jitter_dur in configs_df["jitter_amp_ms"].unique():
        # Extract only the relevant data:
        jitter_dur_df = configs_df.loc[configs_df["jitter_amp_ms"] == jitter_dur]
        # Plot the max decoding accuracy as a function of proportion of trials affected:
        fig, ax = plt.subplots(figsize=fig_size)
        ax.scatter(jitter_dur_df["trials_proportion"], jitter_dur_df["max_decoding"])
        ax.plot(jitter_dur_df["trials_proportion"], jitter_dur_df["max_decoding"])
        ax.set_xlabel('Trials proportion')
        ax.set_ylabel('Peak accuracy')
        # Save the figure to a file:
        plt.savefig(Path(save_root, "{}ms_peak_decoding_scores.png".format(jitter_dur)))

    print("DONE!")


if __name__ == "__main__":
    compare_simulated_jitters()
