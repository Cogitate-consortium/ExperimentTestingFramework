"""Batch runner for onset responsiveness."""
import os
from pathlib import Path
import subprocess
from general_helper_functions.pathHelperFunctions import find_files
from general_helper_functions.data_general_utilities import list_subjects


def onset_responsiveness_batch_runner(bids_root=None, analysis="responsiveness"):
    """
    This function runs jobs for each config found in the current config folder!
    :return:
    """
    # Getting the current dir
    pwd = os.getcwd()
    # Set path to the current configs:
    if analysis == "responsiveness":
        current_config_dir = os.path.join(pwd, "configs")
    elif analysis == "conjunctive":
        current_config_dir = os.path.join(pwd, "configs_conjunctive_analysis")
    else:
        raise Exception("The analysis you passed is not supported! Must be either conjunctive or responsiveness")
    # Find all the json files in the current config:
    config_files = find_files(
        current_config_dir, naming_pattern="*", extension=".json")
    # Launching a job for each:
    for config in config_files:
        # Listing all the subjects we have so far:
        subjects_list = list_subjects(
            Path(bids_root, "derivatives", "preprocessing"))
        for subject in subjects_list:
            if analysis == "responsiveness":
                run_command = "sbatch " + "visual_responsivness_job.sh" \
                              + " --analysis_parameters_file=" \
                              + '"{}"'.format(config) \
                              + " --participant_id=" + '"{}"'.format(subject)
            else:
                run_command = "sbatch " + "conjunctive_analysis_job.sh" \
                              + " --analysis_parameters_file=" \
                              + '"{}"'.format(config) \
                              + " --participant_id=" + '"{}"'.format(subject)
            subprocess.Popen(run_command, shell=True)


if __name__ == "__main__":
    onset_responsiveness_batch_runner(
        bids_root="/mnt/beegfs/XNAT/COGITATE/ECoG/phase_2/processed/bids",
        analysis="responsiveness")
