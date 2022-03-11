import argparse
from paramters_class import EvokedParametersClass
from utilities import find_files, baseline_scaling, path_generator, file_name_generator
from pathlib import Path
import mne
import matplotlib.pyplot as plt


def compute_evoked():
    # Get the parameters:
    parser = argparse.ArgumentParser(
        description="Implements analysis of EDFs for experiment1")
    parser.add_argument('--AnalysisParametersFile', type=str, default=None,
                        help="Config file for analysis parameters (file name + path)")
    parser.add_argument('--subjectID', type=str, default=None,
                        help="Directory and name of the subject info json")
    args = parser.parse_args()

    # Create the parameters file
    parameters_object = EvokedParametersClass(args.AnalysisParametersFile, args.subjectID)

    # Looping through the different analyses configured:
    for analysis_name, analysis_parameters in parameters_object.analysis_parameters.items():
        # ==============================================================================================================
        # Preparing the data:
        # ==============================================================================================================
        # Loading the file
        data_file = find_files(str(Path(parameters_object.input_file_root,
                                        analysis_parameters["signal"], parameters_object.preprocess_steps)),
                               naming_pattern="*-epo", extension=".fif")
        epochs = mne.read_epochs(data_file[0],
                                 verbose='error', preload=True)
        # Adding the input file to the parameters object:
        parameters_object.input_file = data_file[0]

        # Prepare saving directories:
        # Prepare the path to saving the data according to the conventions:
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
        # Dumping the parameters there:
        parameters_object.save_parameters(save_path_fig)
        parameters_object.save_parameters(save_path_results)
        parameters_object.save_parameters(save_path_data)
        # Generate data file names:
        epochs_file_name = file_name_generator(save_path_data, parameters_object.files_prefix,
                                               "data-epo", ".fif", data_type="eeg")
        comp_epo_files_name = str(file_name_generator(save_path_data, parameters_object.files_prefix,
                                                      "{0}_data-epo", ".fif", data_type="eeg"))
        # Generate results (i.e. evoked responses file names):
        evoked_file_name = file_name_generator(save_path_results, parameters_object.files_prefix,
                                               "data-ave", ".fif", data_type="eeg")
        comp_evo_files_name = str(file_name_generator(save_path_results,
                                                      parameters_object.files_prefix,
                                                      "{0}_data-ave", ".fif", data_type="eeg"))
        # Generate figures names:
        image_files_name = str(file_name_generator(save_path_fig, parameters_object.files_prefix,
                                                   "{0}_image", ".png", data_type="eeg"))
        joint_files_name = str(file_name_generator(save_path_fig, parameters_object.files_prefix,
                                                   "{0}_joint", ".png", data_type="eeg"))
        topo_files_name = str(file_name_generator(save_path_fig, parameters_object.files_prefix,
                                                  "{0}_topo", ".png", data_type="eeg"))

        # Selecting the conditions of interest:
        if analysis_parameters["conditions"] is not None:
            epochs = epochs[analysis_parameters["conditions"]]
        # Do baseline correction if needed:
        if analysis_parameters["do_baseline_correction"]:
            baseline_scaling(
                epochs, correction_method=analysis_parameters["baseline_correction_method"])

        # Compute the evoked response:
        evoked = epochs.average()
        # Save the epochs and evoked after the conditions selection and baseline correction:
        epochs.save(epochs_file_name, overwrite=True)
        evoked.save(evoked_file_name)

        # Plot all the electrodes:
        evoked.plot_joint(times="peaks", title="Evoked responses", show=False, picks=parameters_object.data_type)
        plt.savefig(joint_files_name.format("all"), transparent=True)
        plt.close()
        print(joint_files_name)
        # Plotting the topomaps:
        evoked.plot_topomap(times="auto", ch_type=parameters_object.data_type, show=False)
        plt.savefig(topo_files_name.format("all"), transparent=True)
        plt.close()

        # Looping through the different channels groups:
        for component in analysis_parameters["components"].keys():
            # -----------------------------------------------------
            # Data handling:
            # Extracting the data for these specific channels:
            comp_epochs = epochs.copy().pick(analysis_parameters["components"][component])
            comp_evoked = evoked.copy().pick(analysis_parameters["components"][component])
            # Saving the data:
            comp_epochs.save(comp_epo_files_name.format(component), overwrite=True)
            comp_evoked.save(comp_evo_files_name.format(component))

            # ----------------------------------------------------
            # Plotting:
            # Image:
            epochs.plot_image(picks=analysis_parameters["components"][component], combine="mean", show=False)
            plt.savefig(image_files_name.format(component), transparent=True)
            plt.close()
            # Joint:
            comp_evoked.plot_joint(times="peaks", title=component + " evoked responses", show=False,
                                   picks=parameters_object.data_type)
            plt.savefig(joint_files_name.format(component), transparent=True)
            plt.close()

    return None


if __name__ == "__main__":
    compute_evoked()
