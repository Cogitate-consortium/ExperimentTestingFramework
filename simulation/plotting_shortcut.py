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


def plot_shortcut(config):
    with open(config) as f:
        param = json.load(f)

    # ========================================================================================
    # Plot RT results:
    save_root = Path(param["simulation_bids"]["bids_root"], "derivatives", "reaction_time", "population")
    results_save_root = path_generator(save_root,
                                       preprocessing_steps=param["simulation_bids"]["preprocess_steps"],
                                       fig=False, results=True, data=False)
    # Save the results of the jitter and shuffle procedures:
    jitter_results_file = Path(results_save_root, "jitter_results.csv")
    shuffle_results_file = Path(results_save_root, "shuffle_results.csv")
    # Plot the results of the jitter:
    plot_jitters_sims(jitter_results_file, param["simulation_bids"]["bids_root"], "reaction_time",
                      param["simulation_bids"]["preprocess_steps"])
    # Plot the results of the jitter:
    plot_shuffle_sims(shuffle_results_file, param["simulation_bids"]["bids_root"], "reaction_time",
                      param["simulation_bids"]["preprocess_steps"])

    # ========================================================================================
    # Plot ERP results:
    save_root = Path(param["simulation_bids"]["bids_root"], "derivatives", "erp", "population")
    results_save_root = path_generator(save_root,
                                       preprocessing_steps=param["simulation_bids"]["preprocess_steps"],
                                       fig=False, results=True, data=False)
    # Save the results of the jitter and shuffle procedures:
    jitter_results_file = Path(results_save_root, "jitter_results.csv")
    shuffle_results_file = Path(results_save_root, "shuffle_results.csv")
    # Plot the results of the jitter:
    plot_jitters_sims(jitter_results_file, param["simulation_bids"]["bids_root"], "erp",
                      param["simulation_bids"]["preprocess_steps"])
    # Plot the results of the jitter:
    plot_shuffle_sims(shuffle_results_file, param["simulation_bids"]["bids_root"], "erp",
                      param["simulation_bids"]["preprocess_steps"])


if __name__ == "__main__":
    plot_shortcut("config\general_config.json")
