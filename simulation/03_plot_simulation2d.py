import json
import os
from pathlib import Path
from simulation.plotter_functions import plot_jitters_sims, plot_shuffle_sims


def plot_results2d(config_erp, config_rt):

    # ========================================================================================
    # Load config:
    # Load the ERP config:
    with open(config_erp) as f:
        param_erp = json.load(f)
    # Load the RT config:
    with open(config_rt) as f:
        param_rt = json.load(f)

    # ========================================================================================
    # Plot RT results:
    results_save_root = Path(param_rt["bids_root"], "derivatives", param_rt["signal"])
    if not os.path.isdir(results_save_root):
        raise FileNotFoundError("The reaction time simulation files were not found. Make sure to run "
                                "02_simulation_rt.py first!")

    # Save the results of the jitter and shuffle procedures:
    jitter_results_file = Path(results_save_root, "jitter_results.csv")
    shuffle_results_file = Path(results_save_root, "shuffle_results.csv")
    # Plot the results of the jitter:
    plot_jitters_sims(jitter_results_file, param_rt["bids_root"], param_rt["signal"])
    # Plot the results of the jitter:
    plot_shuffle_sims(shuffle_results_file, param_rt["bids_root"], param_rt["signal"])

    # ========================================================================================
    # Plot ERP results:
    results_save_root = Path(param_erp["bids_root"], "derivatives", "erp", "population")
    if not os.path.isdir(results_save_root):
        raise FileNotFoundError("The ERP simulation files were not found. Make sure to run "
                                "01_simulation_erp.py first!")
    # Save the results of the jitter and shuffle procedures:
    jitter_results_file = Path(results_save_root, "jitter_results.csv")
    shuffle_results_file = Path(results_save_root, "shuffle_results.csv")
    # Plot the results of the jitter:
    plot_jitters_sims(jitter_results_file, param_erp["bids_root"], param_erp["signal"])
    # Plot the results of the jitter:
    plot_shuffle_sims(shuffle_results_file, param_erp["bids_root"], param_erp["signal"])


if __name__ == "__main__":
    plot_results2d("01_simulation_erp_config.json", "02_simulation_rt_config.json")
