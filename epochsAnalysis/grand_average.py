import argparse
from parameters_class import ExtractComponentClass
from utilities import path_generator, file_name_generator, list_subjects
from pathlib import Path
import mne
import matplotlib.pyplot as plt


def grand_average():
    # Get the parameters:
    parser = argparse.ArgumentParser(
        description="Implements analysis of EDFs for experiment1")
    parser.add_argument('--AnalysisParametersFile', type=str, default=None,
                        help="Config file for analysis parameters (file name + path)")
    sub_id = "grand_average"
    args = parser.parse_args()

    # Create the parameters file
    parameters_object = ExtractComponentClass(args.AnalysisParametersFile, sub_id)

    # Looping through the different analyses configured:
    for analysis_name, analysis_parameters in parameters_object.analysis_parameters.items():
        # Prepare saving directories:
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
        # Prepare the different file names for saving:
        evoked_file_name = Path(save_path_data, file_name_generator(save_path_data, parameters_object.files_prefix,
                                                                    "data-ave", ".fif", data_type="eeg"))
        comp_evo_files_name = str(Path(save_path_data, file_name_generator(save_path_data,
                                                                           parameters_object.files_prefix,
                                                                           "{0}_data-ave", ".fif", data_type="eeg")))
        joint_files_name = str(Path(save_path_fig, file_name_generator(save_path_fig, parameters_object.files_prefix,
                                                                       "{0}_joint", ".png", data_type="eeg")))
        topo_files_name = str(Path(save_path_fig, file_name_generator(save_path_fig, parameters_object.files_prefix,
                                                                      "{0}_topo", ".png", data_type="eeg")))

        # Generate the names of the files to load for each subject:
        subject_data_root = Path(Path(parameters_object.BIDS_root, "derivatives", "components", "sub-{0}"))
        subject_evoked_file = str(Path(save_path_data, file_name_generator(subject_data_root,
                                                                          parameters_object.files_prefix,
                                                                          "data-ave", ".fif", data_type="eeg")))

        # Listing all the subjects:
        subjects_list = list_subjects(Path(parameters_object.BIDS_root, "derivatives", "component"))
        # Loading all subjects evoked data:
        sub_evo = []
        for sub in subjects_list:
            sub_evo.append(mne.read_evokeds(subject_evoked_file.format(sub)))
        # Perform the grand average:
        grand_avg = mne.grand_average(sub_evo)
        # Save the grand average data:
        grand_avg.save(evoked_file_name)

        # Plot the grand average:
        grand_avg.plot_joint(times="peaks", title="Grand average evoked responses",
                             show=False, picks=parameters_object.data_type)
        plt.savefig(joint_files_name.format("all"), transparent=True)
        plt.close()
        print(joint_files_name)
        # Plotting the topomaps:
        grand_avg.plot_topomap(times="auto", ch_type=parameters_object.data_type, show=False)
        plt.savefig(topo_files_name.format("all"), transparent=True)
        plt.close()

        # Looping through the components:
        for component in analysis_parameters["components"].keys():
            comp_evoked = grand_avg.copy().pick(analysis_parameters["components"][component])
            # Save to file:
            comp_evoked.save(comp_evo_files_name.format(component))
            # Plot the joint:
            comp_evoked.plot_joint(times="peaks", title="Grand average evoked responses", show=False,
                                   picks=parameters_object.data_type)
            plt.savefig(joint_files_name.format(component), transparent=True)
            plt.close()


if __name__ == "__main__":
    grand_average()
