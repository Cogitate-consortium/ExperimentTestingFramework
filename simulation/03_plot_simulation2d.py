import json
from pathlib import Path
from general_utilities.path_helper_function import path_generator
from simulation.plotter_functions import plot_jitters_sims, plot_shuffle_sims


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
