import argparse
from pathlib import Path
import os
import json
import itertools
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

from general_utilities.path_helper_function import find_files, list_subjects, path_generator, load_epochs
from general_utilities.data_helper_function import mean_confidence_interval
from general_utilities.jitter_simulation import generate_jitter


def single_subject_mvpa(subject, epochs, config, conditions=None, labels_condition=None, classifier="svm", n_cv=5,
                        n_features=30, n_jobs=8):
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
    # Mess the triggers timing if needed:
    if config["trigger_jitter_parameter"] is not None:
        epochs = generate_jitter(epochs, **config["trigger_jitter_parameter"])

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
    elif classifier.lower() == "lda":
        clf = make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis()
        )
    else:
        raise Exception("The classifier passed was not recogized! Only SVM or logisticregression supported")

    # Extract the data:
    data = np.squeeze(epochs.get_data())
    # Extract the labels:
    y = epochs.metadata[labels_condition].to_numpy()
    labels = list(set(y))
    # Perform cross validation:
    skf = StratifiedKFold(n_splits=n_cv)
    confusion_matrices = np.zeros((len(labels), len(labels),
                                   np.arange(0, data.shape[-1] - config["n_sample_window"] + 1,
                                             config["n_sample_steps"]).shape[0],
                                   n_cv))
    fold_ctr = 0
    # Looping through folds:
    for train_index, test_index in skf.split(data, y):
        print("Training and testing classifier")
        print("Fold: {}".format(fold_ctr + 1))
        sample_ctr = 0
        # Looping through the time windows:
        for i in range(0, data.shape[-1] - config["n_sample_window"] + 1, config["n_sample_steps"]):
            x_train, x_test = data[train_index, :, i:i + config["n_sample_window"]], \
                              data[test_index, :, i:i + config["n_sample_window"]]
            y_train, y_test = y[train_index], y[test_index]
            # Reshape the x arays:
            x_train, x_test = x_train.reshape(x_train.shape[0], -1), x_test.reshape(x_test.shape[0], -1)
            # Fit the classifier on the train data:
            clf.fit(X=x_train, y=y_train)
            # Predict:
            y_pred = clf.predict(x_test)
            # Generate a confusion matrix:
            confusion_matrices[:, :, sample_ctr, fold_ctr] = confusion_matrix(y_test, y_pred,
                                                                              labels=labels, normalize="true")
            sample_ctr += 1
        fold_ctr += 1
    # Average across the CV:
    confusion_matrices = np.mean(confusion_matrices, axis=-1)
    # Loop through the labels:
    scores = {}
    for ind, label in enumerate(labels):
        scores[label] = confusion_matrices[ind, ind, :]
        np.save(Path(results_save_root, "sub-" + subject + "_"
                     + label + "_decoding_scores.npy"), scores[label])

    # Plot the results:
    fig, ax = plt.subplots()
    for label in scores.keys():
        times = np.linspace(epochs.times[0], epochs.times[-1], num=scores[label].shape[0])
        ax.plot(times, scores[label], label=label)
    ax.axhline(1 / len(labels), color='k', linestyle='--', label='chance')
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
        t0 = None
        tmax = None
        for subject in subjects_list:
            # Load this subject epochs:
            epochs = load_epochs(config["bids_root"], subject,
                                 config["ses"], config["data_type"], config["preprocess_folder"],
                                 config["signal"], config["preprocess_steps"], config["task"])
            t0, tmax = epochs.times[0], epochs.times[-1]
            # Run the decoding on this subject:
            scores.append(single_subject_mvpa(subject, epochs, config,
                                              conditions=config["conditions"],
                                              labels_condition=config["labels_condition"],
                                              classifier=config["classifier"],
                                              n_cv=config["n_cv"],
                                              n_jobs=config["n_jobs"],
                                              n_features=config["n_features"]))

        # Generate the path to save the population results:
        save_root = Path(config["bids_root"], "derivatives", config["analysis"], "population")
        fig_save_root = path_generator(save_root, analysis=config["name"],
                                       preprocessing_steps=config["preprocess_steps"],
                                       fig=True, results=False, data=False)
        results_save_root = path_generator(save_root, analysis=config["name"],
                                           preprocessing_steps=config["preprocess_steps"],
                                           fig=False, results=True, data=False)
        # Combinining participants results:
        population_scores = {key: None for key in scores[0].keys()}
        for label in population_scores.keys():
            population_scores[label] = np.array([score[label] for score in scores])
            np.save(Path(results_save_root, "sub-population_"
                         + label + "_decoding_scores.npy"), population_scores[label])

        # Plot the results:
        fig, ax = plt.subplots()
        for label in population_scores.keys():
            # Compute the mean and ci of the decoding:
            avg, low_ci, up_ci = mean_confidence_interval(population_scores[label])
            times = np.linspace(t0, tmax, num=avg.shape[0])
            ax.scatter(times, avg)
            ax.plot(times, avg, label=label, linewidth=0.5)
            ax.fill_between(times, low_ci, up_ci, alpha=.2)
        ax.set_xlabel('Times')
        ax.set_ylabel('Accuracy')  # Area Under the Curve
        ax.legend()
        ax.set_title('Population decoding')
        # Save the figure to a file:
        plt.savefig(Path(fig_save_root, "population" + "_decoding_scores.png"))
        print("DONE!")


if __name__ == "__main__":
    mvpa_manager()
