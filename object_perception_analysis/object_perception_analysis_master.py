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


def single_subject_mvpa(subject, epochs, config, conditions=None, labels_condition=None, classifier="svm", n_cv=5,
                        n_jobs=8):
    """

    :param subject:
    :param epochs:
    :param config:
    :param conditions:
    :param labels_condition:
    :param classifier:
    :param n_cv:
    :param n_jobs:
    :return:
    """
    # =========================================================================================
    # Housekeeping
    save_root = Path(config["bids_root"], "derivatives", config["analysis"], "sub-" + subject)
    fig_save_root = path_generator(save_root, analysis=config["name"],
                                   preprocessing_steps=config["preprocess_steps"],
                                   fig=True, results=False, data=False)
    results_save_root = path_generator(save_root, analysis=config["name"],
                                       preprocessing_steps=config["preprocess_steps"],
                                       fig=False, results=True, data=False)
    # Saving the config where the data will be saved:
    with open(Path(fig_save_root, 'config.json'), 'w') as f:
        json.dump(config, f)
    with open(Path(results_save_root, 'config.json'), 'w') as f:
        json.dump(config, f)

    # =========================================================================================
    # Prepare the data:
    if conditions is not None:
        epochs = epochs[conditions]
    # Extract the data:
    data = np.squeeze(epochs.get_data())
    # Extract the labels:
    labels = epochs.metadata[labels_condition].to_numpy()

    # Prepare the classifier:
    if classifier.lower() == "svm":
        clf = make_pipeline(
            StandardScaler(),
            LinearSVC()
        )
    elif classifier.lower() == "logisticregression":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(solver='liblinear')
        )
    else:
        raise Exception("The classifier passed was not recogized! Only SVM or logisticregression supported")

    # Performing the decoding in a time resolved fashion:
    time_decod = SlidingEstimator(clf, n_jobs=n_jobs, scoring='accuracy', verbose=True)
    # Scoring the classifier:
    scores = cross_val_multiscore(time_decod, data, labels, cv=n_cv, n_jobs=n_jobs)

    # Saving the results to file:
    np.save(scores, Path(results_save_root, "sub-" + subject + "_decoding_scores.npy"))

    # Average scores across cross-validation splits
    scores = np.mean(scores, axis=0)
    # Plot the results:
    fig, ax = plt.subplots()
    ax.plot(epochs.times, scores, label='score')
    ax.axhline(.5, color='k', linestyle='--', label='chance')
    ax.set_xlabel('Times')
    ax.set_ylabel('Accuracy')  # Area Under the Curve
    ax.legend()
    ax.axvline(.0, color='k', linestyle='-')
    ax.set_title('Sensor space decoding')
    # Save the figure to a file:
    plt.savefig(Path(fig_save_root, "sub-" + subject + "_decoding_scores.png"))
    plt.close()
    return scores


def mvpa_manager():
    parser = argparse.ArgumentParser(description="Arguments for MVPA analysis across all subjects")
    parser.add_argument('--config', type=str, default=None,
                        help="Config file for analysis parameters (file name + path)")
    args = parser.parse_args()
    if args.config is None:
        configs = find_files(Path(os.getcwd(), "configs"), naming_pattern="*", extension=".json")
    else:
        configs = [args.config]

    # Looping through all config:
    for config in configs:
        # Load the config:
        with open(config) as f:
            config = json.load(f)
        # List the subjects:
        subjects_list = list_subjects(Path(config["bids_root"], "derivatives", "preprocessing"), prefix="sub-")

        # Looping through each subject to launch the analysis across all:
        scores = []
        for subject in subjects_list:
            # Load this subject epochs:
            epochs = load_epochs(config["bids_root"], subject,
                                 config["ses"], config["data_type"], config["preprocess_folder"],
                                 config["signal"], config["preprocess_steps"], config["task"])
            # Run the decoding on this subject:
            scores.append(single_subject_mvpa(subject, epochs, config,
                                              conditions=config["conditions"],
                                              labels_condition=config["labels_condition"],
                                              classifier=config["classifier"],
                                              n_cv=config["n_cv"],
                                              n_jobs=config["n_jobs"]))

        # Generate the path to save the population results:
        save_root = Path(config["bids_root"], "derivatives", config["analysis"], "population")
        fig_save_root = path_generator(save_root, analysis=config["name"],
                                       preprocessing_steps=config["preprocess_steps"],
                                       fig=True, results=False, data=False)
        results_save_root = path_generator(save_root, analysis=config["name"],
                                           preprocessing_steps=config["preprocess_steps"],
                                           fig=False, results=True, data=False)
        # Saving the results to file:
        np.save(scores, Path(results_save_root, "population_decoding_scores.npy"))
        scores = np.array(scores)
        # Compute the mean and ci of the decoding:
        avg, up_ci, low_ci = mean_confidence_interval(scores)
        # Plot the results:
        fig, ax = plt.subplots()
        ax.plot(epochs.times, avg)
        ax.fill_between(epochs.times, up_ci, low_ci, alpha=.2)
        ax.axhline(.5, color='k', linestyle='--', label='chance')
        ax.set_xlabel('Times')
        ax.set_ylabel('Accuracy')  # Area Under the Curve
        ax.legend()
        ax.axvline(.0, color='k', linestyle='-')
        ax.set_title('Population decoding')
        # Save the figure to a file:
        plt.savefig(Path(fig_save_root, "population" + "_decoding_scores.png"))


if __name__ == "__main__":
    mvpa_manager()
