import argparse
from parameters_class import AnalysisParametersClass
from utilities import path_generator, file_name_generator, list_subjects
from pathlib import Path
import mne
import matplotlib.pyplot as plt


def population_evoked_comparison():
    """
    This function loads all the participants evoked data and perform grand average. The grand average is then plotted
    in different ways to make data exploration easier
    :return:
    """
    # Set the parameters:
    parser = argparse.ArgumentParser(
        description="Implements analysis of EDFs for experiment1")
    parser.add_argument('--AnalysisParametersFile', type=str, default=None,
                        help="Config file for analysis parameters (file name + path)")
    sub_id = "grand_average"
    args = parser.parse_args()

    # Create the parameters file
    parameters_object = AnalysisParametersClass("compare_evoked", args.AnalysisParametersFile, sub_id)

    # Looping through the different analyses configured:
    for analysis_name, analysis_parameters in parameters_object.analysis_parameters.items():
        # ==============================================================================================================
        # Prepare path variables:
        # ==============================================================================================================
        # Prepare saving directories:
        save_path_fig = path_generator(parameters_object.save_root,
                                       analysis=analysis_name,
                                       preprocessing_steps=parameters_object.preprocess_steps,
                                       fig=True, results=False, data=False)
        save_path_results = path_generator(parameters_object.save_root,
                                           analysis=analysis_name,
                                           preprocessing_steps=parameters_object.preprocess_steps,
                                           fig=False, results=True, data=False)

        # Prepare the different file names for saving:
        evoked_file_name = file_name_generator(save_path_results, parameters_object.files_prefix,
                                               "data-ave", ".fif", data_type="eeg")
        comparison_files_name = str(file_name_generator(save_path_fig, parameters_object.files_prefix,
                                                        "{0}_compare", ".png", data_type="eeg"))

        # Generate the names of the files to load for each subject:
        subjects_results_root = Path(parameters_object.BIDS_root,
                                     "derivatives", parameters_object.analysis_name, "sub-{0}", "results",
                                     analysis_name, parameters_object.preprocess_steps)
        sub_file_prefix = "sub-{0}_task-" + parameters_object.task_name + "_analysis-" + \
                          parameters_object.analysis_name
        subject_evoked_file = str(file_name_generator(subjects_results_root,
                                                      sub_file_prefix,
                                                      "data-ave", ".fif", data_type="eeg"))

        # ------------------------------------------------------------------------------------------
        # Load all subjects data and perform grand average per condition
        # List all subjects:
        subjects_list = list_subjects(Path(parameters_object.BIDS_root, "derivatives",
                                           parameters_object.analysis_name))
        # Loading the data of all subjects per condition:
        grand_average = {cond: None for cond in analysis_parameters["conditions"]}
        for cond in analysis_parameters["conditions"]:
            # Fetch of all participants for the given condition:
            cond_evoked = []
            for sub in subjects_list:
                if sub != sub_id:
                    cond_evoked.append(mne.read_evokeds(subject_evoked_file.format(sub),
                                                        verbose='warning', condition=cond))
            # Performing the grand average for this condition:
            grand_average[cond] = mne.grand_average(cond_evoked)
        del cond_evoked
        # Convert grand average to list for saving:
        grand_average_list = [grand_average[cond] for cond in grand_average.keys()]
        mne.write_evokeds(evoked_file_name, grand_average_list)
        del grand_average_list

        # ------------------------------------------------------------------------------------------
        # Extract each components:
        for component in analysis_parameters["components"].keys():
            # Create the title:
            title = " ".join([component, "grand average", " vs ".join([cond for cond in grand_average.keys()]),
                              "(N-sub={0})".format(len(subjects_list)-1)])
            # Plotting the comparison for these two conditions:
            mne.viz.plot_compare_evokeds(grand_average, combine='mean',
                                         picks=analysis_parameters["components"][component],
                                         title=title, show=False)
            plt.savefig(comparison_files_name.format(component), transparent=True)

        return None


if __name__ == "__main__":
    population_evoked_comparison()
