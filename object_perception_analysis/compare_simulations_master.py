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
        configs_df = configs_df.append(pd.DataFrame({
            "name": config["name"],
            "refresh_rate": config["trigger_jitter_parameter"]["refresh_rate"],
            "jitter_trials_proportion": config["trigger_jitter_parameter"]["trials_proportion"],
            "jitter_tails": config["trigger_jitter_parameter"]["tail"],
            "jitter_max": config["trigger_jitter_parameter"]["max_jitter"],
            "shuffle_trials_proportion": config["trigger_shuffle_parameter"]["trials_proportion"],
            "max_decoding": max_decoding
        }, index=[0]))
    configs_df = configs_df.reset_index(drop=True)

    # Create path to save the results:
    save_root = Path(config["bids_root"], "derivatives", config["analysis"], "simulations")
    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    # ==================================================================================================================
    # 1. Comparing between jitter proportion with everything else constant:
    no_shuffle_save_root = Path(config["bids_root"], "derivatives", config["analysis"], "simulations",
                                "jitter_no_shuffle")
    if not os.path.isdir(no_shuffle_save_root):
        os.makedirs(no_shuffle_save_root)
    # Only interested with no shuffle in a first time:
    no_shuffle_config = configs_df.loc[configs_df["shuffle_trials_proportion"] == 0]
    # Looping through the tail parameters:
    for tail in list(no_shuffle_config["jitter_tails"].unique()):
        config_tail = no_shuffle_config.loc[no_shuffle_config["jitter_tails"] == tail]
        # Looping through the max jitter:
        for max_jitter in list(config_tail["jitter_max"].unique()):
            config_max_jitter = config_tail.loc[config_tail["jitter_max"] == max_jitter]
            config_max_jitter = config_max_jitter.sort_values("jitter_trials_proportion")
            fig, ax = plt.subplots(figsize=fig_size)
            ax.scatter(config_max_jitter["jitter_trials_proportion"].to_numpy(),
                       config_max_jitter["max_decoding"].to_numpy())
            ax.plot(config_max_jitter["jitter_trials_proportion"].to_numpy(),
                    config_max_jitter["max_decoding"].to_numpy())
            ax.set_xlabel('jitter trials proportion')
            ax.set_ylabel('Shuffled labels peak accuracy')
            # Save the figure:
            if max_jitter is not None:
                plt.savefig(Path(no_shuffle_save_root,
                                 "peak_decoding_per_jitter_proportion_{}_max{}ms.png".format(tail, max_jitter)))
            else:
                plt.savefig(Path(no_shuffle_save_root,
                                 "peak_decoding_per_jitter_proportion_{}.png".format(tail)))

            # Plot the max time resolved decoding results under each jitter conditions:
            fig, ax = plt.subplots(figsize=fig_size)
            for name in list(config_tail["name"].unique()):
                # Average across all the different labels:
                param = configs_df.loc[configs_df["name"] == name]
                label = "{}ms, {}%trials".format(param["refresh_rate"].values[0],
                                                 param["jitter_trials_proportion"].values[0] * 100)
                data = np.max(np.array([results[name][label] for label in results[name].keys()]), axis=0)
                avg, low_ci, up_ci = mean_confidence_interval(data)
                times = np.linspace(t0, tmax, num=avg.shape[-1])
                ax.plot(times, avg, label=label)
                ax.scatter(times, avg)
                ax.fill_between(times, up_ci, low_ci, alpha=.2)
            ax.legend()
            ax.axvline(.0, color='k', linestyle='-')
            ax.set_title('Average decoding accuracy')
            ax.set_xlabel('Times (s)')
            ax.set_ylabel('Accuracy')  # Area Under the Curve
            if max_jitter is not None:
                plt.savefig(Path(no_shuffle_save_root,
                                 "time_resolved_decoding_per_jitter_proportion_{}_max{}ms.png".format(tail,
                                                                                                      max_jitter)))
            else:
                plt.savefig(Path(no_shuffle_save_root,
                                 "time_resolved_decoding_per_jitter_proportion_{}.png".format(tail)))

    print("DONE!")


if __name__ == "__main__":
    compare_simulated_jitters()
