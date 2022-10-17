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


def compare_simulated_jitters():
    parser = argparse.ArgumentParser(description="Arguments for MVPA analysis across all subjects")
    parser.add_argument('--config', type=str, default=None,
                        help="Config file for analysis parameters (file name + path)")
    args = parser.parse_args()
    if args.config is None:
        configs = find_files(Path(os.getcwd(), "configs"), naming_pattern="*", extension=".json")
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
        with open(Path(results_save_root, "sub-population_decoding_scores.pkl"), 'wb') as f:
            results[config["name"]] = pickle.load(f)
        # Get the maximal decoding values observed:
        max_decoding = np.max([np.max(results[config["name"]][key]) for key in config["name"].keys()])
        # Extract the config relevant info as a dataframe:
        configs_df = pd.append(pd.DataFrame({
            "name": config["name"],
            "jitter_amp_ms": config["trigger_jitter_parameter"]["jitter_amp_ms"],
            "trials_proportion": config["trigger_jitter_parameter"]["trials_proportion"],
            "max_decoding": max_decoding
        }, index=[0]))
    configs_df = configs_df.reset_index(drop=True)

    # =======================================================================
    # 1. Plot the decoding results averaged across labels:
    save_root = Path(config["bids_root"], "derivatives", config["analysis"])
    fig, ax = plt.subplots()
    for ind, key in enumerate(results.keys()):
        # Average across all the different labels:
        data = np.mean(np.array([results[key][label] for label in results[key].keys()]), axis=0)
        avg, low_ci, up_ci = mean_confidence_interval(data)
        ax.plot(avg, label=key)
        ax.fill_between(range(avg.shape[-1]), up_ci, low_ci, alpha=.2)
        ax.set_xlabel('Times')
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
        fig, ax = plt.subplots()
        for ind, key in enumerate(results.keys()):
            # Average across all the different labels:
            avg, low_ci, up_ci = mean_confidence_interval(results[key][label])
            ax.plot(avg, label=key)
            ax.fill_between(range(avg.shape[-1]), up_ci, low_ci, alpha=.2)
            ax.set_xlabel('Times')
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
        fig, ax = plt.subplots()
        ax.scatter(jitter_dur_df["trials_proportion"], jitter_dur_df["max_decoding"])
        ax.plot(jitter_dur_df["trials_proportion"], jitter_dur_df["max_decoding"])
        ax.set_xlabel('Trials proportion')
        ax.set_ylabel('Peak accuracy')
        # Save the figure to a file:
        plt.savefig(Path(save_root, "{}ms_peak_decoding_scores.png".format(jitter_dur)))

    print("DONE!")


if __name__ == "__main__":
    compare_simulated_jitters()
