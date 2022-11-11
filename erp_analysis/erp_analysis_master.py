import argparse
import json
import os

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy.signal import peak_widths
from general_utilities.path_helper_function import find_files, path_generator, list_subjects, load_epochs
from general_utilities.simulate_malfunction import jitter_trials, shuffle_triggers
import numpy as np

test_subs = ["1", "2", "3", "4"]


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
                        # epochs = shuffle_triggers(epochs, trials_proportion=shuffle_proportion)
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
                            # Append to the results:
                            results = results.append(pd.DataFrame({
                                "task": task,
                                "effect_size": sim_param["effect_size"][component],
                                "subject": subject,
                                "jitter_prop": str(jitter_proportion),
                                "shuffle_proportion": str(shuffle_proportion),
                                "component": component,
                                "peak-amp": peak,
                                "peak-latency": latency,
                                **cond_means
                            }, index=[0]))

        # Save the results to file:
        results.to_csv(Path(results_save_root, "components_results.csv"))

        # ================================================================================================
        # Plot the effect sizes:
        # Compute the effect sizes of the difference between the two conditions:
        results["cond_diff"] = results["cond_1-mean"] - results["cond_2-mean"]
        effect_sizes = pd.DataFrame()
        # Looping through each component again:
        for component in results["component"]:
            for task in results["task"].unique():
                for jitter_prop in results["jitter_prop"]:
                    for shuffle_prop in results["shuffle_proportion"]:
                        # Extract all the data:
                        diff = comp_results.loc[(results["component"] == component)
                                                and (results["task"] == task)
                                                and (comp_results["jitter_prop"] == jitter_prop)
                                                and (results["shuffle_prop"] == shuffle_prop), "cond_diff"]. \
                            to_numpy()
                        # Compute the effect size:
                        effect_sizes = effect_sizes.append(pd.DataFrame({
                            "task": task,
                            "jitter_prop": jitter_prop,
                            "shuffle_prop": shuffle_prop,
                            "effect_size": np.mean(diff) / np.std(diff)
                        }, index=[0]))
            # Get the data of this component:
            comp_effect_size = effect_sizes.loc[effect_sizes["component"] == component].pivot(index="jitter_prop",
                                                                                              columns="shuffle_prop",
                                                                                              values="effect_size")
            # Generate a heatmap:
            fig, ax = plt.subplots(1)
            sns.heatmap(comp_effect_size, ax=ax)
            ax.set_ylabel("Label shuffle proportion")
            ax.set_xlabel("Jittered trials proportion")
            ax.set_title("Effect size as a function of trial jitters and shuffle")
            plt.savefig(Path(fig_save_root, "{}_effect_size.png".format(component)))

        # ================================================================================================
        # Plot the peak and latency:
        # Looping through each component again:
        for component in results["component"]:
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

            fig, ax = plt.subplots(2)
            sns.boxplot(shuffle_data, x="peak-amp", y="shuffle_proportion", orient="v", ax=ax[1])
            sns.boxplot(shuffle_data, x="peak-latency", y="shuffle_proportion", orient="v", ax=ax[1])
            ax[0].set_ylabel("Peak amplitude (mV)")
            ax[1].set_ylabel("Peak latency (ms)")
            ax[1].set_xlabel("Shuffle proportion")
            ax[0].set_title("{} changes as a function of jitter".format(component))
            plt.savefig(Path(fig_save_root, "{}_peaks_shuffle.png".format(component)))

        print("A")
        # Loop through all the


if __name__ == "__main__":
    erp_analysis()
