import argparse
from parameters_class import AnalysisParametersClass
from utilities import find_files, baseline_scaling, path_generator, file_name_generator
from pathlib import Path
import mne
import matplotlib.pyplot as plt


def compare_conditions():
    """
    This function enables comparing different conditions from the epochs data. This script therefore selects
    specific trials of the epochs data matching conditions set in the evoked_configs. Then, the activation to these different
    conditions is plotted
    :return:
    """
    # Get the parameters:
    parser = argparse.ArgumentParser(
        description="Implements analysis of EDFs for experiment1")
    parser.add_argument('--AnalysisParametersFile', type=str, default=None,
                        help="Config file for analysis parameters (file name + path)")
    parser.add_argument('--subjectID', type=str, default=None,
                        help="Directory and name of the subject info json")
    args = parser.parse_args()

    # Create the parameters file
    parameters_object = AnalysisParametersClass("compare_evoked", args.AnalysisParametersFile, args.subjectID)

    # Looping through the different analyses configured:
    for analysis_name, analysis_parameters in parameters_object.analysis_parameters.items():
        # ==============================================================================================================
        # Prepare path parameters for saving data and results:
        # ==============================================================================================================
        # Get the file to be used for this analysis:
        # Loading the file
        data_file = find_files(str(Path(parameters_object.input_file_root,
                                        analysis_parameters["signal"], parameters_object.preprocess_steps)),
                               naming_pattern="*-epo", extension=".fif")
        # Adding the input file to the parameters object:
        parameters_object.input_file = data_file[0]

        # Prepare the path to where the data should be saved:
        save_path_fig = path_generator(parameters_object.save_root,
                                       analysis=analysis_name,
                                       preprocessing_steps=parameters_object.preprocess_steps,
                                       fig=True, results=False, data=False)
        save_path_results = path_generator(parameters_object.save_root,
                                           analysis=analysis_name,
                                           preprocessing_steps=parameters_object.preprocess_steps,
                                           fig=False, results=True, data=False)
        save_path_data = path_generator(parameters_object.save_root,
                                        analysis=analysis_name,
                                        preprocessing_steps=parameters_object.preprocess_steps,
                                        fig=False, results=False, data=True)
        # Saving the different configs to the different directories:
        parameters_object.save_parameters(save_path_fig)
        parameters_object.save_parameters(save_path_results)
        parameters_object.save_parameters(save_path_data)
        # prepare the different file names:
        epochs_file_name = file_name_generator(save_path_data, parameters_object.files_prefix,
                                               "data-epo", ".fif", data_type="eeg")
        evoked_file_name = file_name_generator(save_path_results, parameters_object.files_prefix,
                                               "data-ave", ".fif", data_type="eeg")
        comparison_files_name = str(file_name_generator(save_path_fig, parameters_object.files_prefix,
                                                        "{0}_compare", ".png", data_type="eeg"))

        # ==============================================================================================================
        # Preparing the data:
        # ==============================================================================================================
        epochs = mne.read_epochs(data_file[0],
                                 verbose='error', preload=True)
        # Adding the input file to the parameters object:
        parameters_object.input_file = data_file[0]

        # Do baseline correction if needed:
        if analysis_parameters["do_baseline_correction"]:
            baseline_scaling(
                epochs, correction_method=analysis_parameters["baseline_correction_method"])
        # Save the epochs data:
        epochs.save(epochs_file_name, overwrite=True)
        # Selecting the conditions of interest:
        if analysis_parameters["conditions"] is not None:
            cond_epochs = {cond: None for cond in analysis_parameters["conditions"].keys()}
            for cond in analysis_parameters["conditions"]:
                cond_epochs[cond] = epochs[analysis_parameters["conditions"][cond]]
        else:
            raise Exception("You cannot use the compare_conditions.py script without passing different conditions "
                            "\nto compare!")
        # Compute the evoked for each condition:
        cond_evoked = {cond: cond_epochs[cond].average() for cond in cond_epochs.keys()}
        # These can now be saved:
        mne.write_evokeds(evoked_file_name, [cond_evoked[cond] for cond in cond_evoked.keys()])

        # Prepare the data for each condition:
        evokeds = {cond: list(cond_epochs[cond].iter_evoked()) for cond in cond_epochs.keys()}
        # Looping through the different components:
        for component in analysis_parameters["components"].keys():
            # Create the title:
            cond_n_trials = {cond: len(evokeds[cond]) for cond in evokeds}
            title = " ".join([component, "component", " vs ".join([cond + " (N={0})".format(cond_n_trials[cond])
                                                                   for cond in cond_n_trials.keys()])])
            # Plotting the comparison for these two conditions:
            mne.viz.plot_compare_evokeds(evokeds, combine='mean', picks=analysis_parameters["components"][component],
                                         title=title, show=False)
            plt.savefig(comparison_files_name.format(component), transparent=True)

        return None


if __name__ == "__main__":
    compare_conditions()
