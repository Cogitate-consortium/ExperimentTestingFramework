import argparse
from pathlib import Path
import os
import json
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)
import matplotlib.pyplot as plt

from general_utilities.path_helper_function import find_files, list_subjects, path_generator, load_epochs
from general_utilities.data_helper_function import mean_confidence_interval
from general_utilities.jitter_simulation import generate_jitter


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
        results[config["name"]] = np.load(Path(results_save_root, "population_decoding_scores.npy"))

    # Loop through each set of results:
    fig, ax = plt.subplots()
    save_root = Path(config["bids_root"], "derivatives", config["analysis"])
    for key in results.keys():
        avg, low_ci, up_ci = mean_confidence_interval(results[key])
        ax.plot(avg, label=key)
        ax.fill_between(up_ci, low_ci, alpha=.2)
        ax.axhline(.5, color='k', linestyle='--', label='chance')
        ax.set_xlabel('Times')
        ax.set_ylabel('Accuracy')  # Area Under the Curve
    ax.legend()
    ax.axvline(.0, color='k', linestyle='-')
    ax.set_title('Population decoding')
    # Save the figure to a file:
    plt.savefig(Path(save_root, "population" + "_decoding_scores.png"))

    print("DONE!")


if __name__ == "__main__":
    compare_simulated_jitters()
