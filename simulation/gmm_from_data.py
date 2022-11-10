
import json
import scipy

from pathlib import Path
import numpy as np
from general_utilities.path_helper_function import list_subjects, load_epochs
from simulation.simulation_helper_functions import gaussian_5_comp


def gmm_from_data(components_priors,
                  bids_root="C:\\Users\\alexander.lepauvre\\Documents\\PhD\\Experimental_testing_framework\\data\\"
                            "object_processing\\bids",  session="1", data_type="eeg",
                  preprocessing_folder="epoching", signal="erp", preprocessing_steps="unknown",
                  task="object_processing", verbose=False):
    """
    This function loads preprocessed data and fits a gaussian mixture model to the data. The fitted components are then
    returned to be able to simulate data from them
    :param components_prior: (string or dict) either path to the component prior json file or dictionary containing
    the priors
    :param bids_root: (string or path object) path to the bids root
    :param session: (string) name of the session
    :param data_type: (string) type of data to be loaded
    :param preprocessing_folder: (string) preprocessing folder from which to load the data. Depends on the pipeline
    :param signal: (string) name of the signal to load
    :param preprocessing_steps: (string) subfolder specifying the preprocessing steps taken on the data
    :param task: (string) name of the data
    :param verbose: (bool) Whether or not to print to the command line
    :return:
    """
    if verbose:
        print("=" * 40)
        print("Welcome to gmm_from_data")
    # ==================================================================================================================
    # Loading the subjects data:
    # List the subjects:
    subjects_list = list_subjects(Path(bids_root, "derivatives", "preprocessing"), prefix="sub-")
    # Loading single subjects data and performing some operations:
    evk = {subject: None for subject in subjects_list}
    for subject in subjects_list:
        if verbose:
            print("loading sub-{} data".format(subject))
        # Load this subject epochs:
        epochs = load_epochs(bids_root, subject,
                             session, data_type, preprocessing_folder,
                             signal, preprocessing_steps, task)
        # Average the data in each sensor:
        evk[subject] = epochs.average()
    # Average the evk across subjects:
    data = np.array([evk[subject].data for subject in evk.keys()])
    grand_avg = np.mean(data, axis=0)

    # ==================================================================================================================
    # Fitting the gaussian mixture model:
    if verbose:
        print("")
        print("Fitting the gaussian mixture model: ")
    # Load the priors:
    if not isinstance(components_priors, dict):
        if isinstance(components_priors, str):
            with open(components_priors) as f:
                components_priors = json.load(f)
        else:
            raise Exception("The component priors must be either a dictionary or a json file string!")
    # Convert the priors to lists:
    amplitudes = [components_priors[comp]["amplitude"] for comp in components_priors.keys()]
    latencies = [components_priors[comp]["latency"] for comp in components_priors.keys()]
    sigmas = [components_priors[comp]["sigma"] for comp in components_priors.keys()]
    bounds = [[], []]
    for parameter in ["amplitude_limits", "latency_limits", "sigma_limits"]:
        for component in components_priors.keys():
            bounds[0].append(components_priors[component][parameter][0])
            bounds[1].append(components_priors[component][parameter][1])
    # Fit the model to the data to extract the parameters:
    popt, pcov = scipy.optimize.curve_fit(gaussian_5_comp, epochs.times, np.squeeze(grand_avg[0, :].T),
                                          p0=amplitudes + latencies + sigmas,
                                          bounds=bounds)
    # Convert the fitted parameters to a dict:
    components_posteriors = {component: {
            "amplitude": None,
            "latency": None,
            "sigma": None
        }
        for component in components_priors.keys()
    }
    for ind, component in enumerate(components_posteriors.keys()):
        components_posteriors[component]["amplitude"] = popt[ind]
        components_posteriors[component]["latency"] = popt[ind + len(components_posteriors)]
        components_posteriors[component]["sigma"] = popt[ind + len(components_posteriors) * 2]
    with open('components_posteriors.json', 'w') as f:
        json.dump(components_posteriors, f)
    if verbose:
        for component in components_posteriors:
            print("{} :".format(component))
            print("   Amplitude: {}".format(components_posteriors[component]["amplitude"]))
            print("   Latency: {}".format(components_posteriors[component]["latency"]))
            print("   Sigma: {}".format(components_posteriors[component]["sigma"]))

    return components_posteriors


if __name__ == "__main__":
    gmm_from_data("components_priors.json", verbose=True)