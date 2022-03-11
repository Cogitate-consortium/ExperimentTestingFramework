"""Batch runner for epochs_analysis."""
import os
from pathlib import Path
import subprocess
from utilities import find_files, list_subjects


def batch_runner(bids_root=None, analysis="epochs_analysis"):
    """
    This function runs jobs for each config found in the current config folder!
    :return:
    """
    # Getting the current dir
    pwd = os.getcwd()
    # Set path to the current configs:
    if analysis == "epochs_analysis":
        current_config_dir = os.path.join(pwd, "configs")
    else:
        raise Exception("The analysis you passed is not supported!")
    # Find all the json files in the current config:
    config_files = find_files(
        current_config_dir, naming_pattern="*", extension=".json")
    # Launching a job for each:
    for config in config_files:
        # Listing all the subjects we have so far:
        subjects_list = list_subjects(
            Path(bids_root, "derivatives", "preprocessing"))
        for subject in subjects_list:
            run_command = "sbatch " + "compute_evoked_slurm.sh" \
                          + " --analysis_parameters_file=" \
                          + '"{}"'.format(config) \
                          + " --participant_id=" + '"{}"'.format(subject)
            subprocess.Popen(run_command, shell=True)


if __name__ == "__main__":
    batch_runner(
        bids_root="/mnt/beegfs/XNAT/workspace/001_testing_framework/data/bids",
        analysis="epochs_analysis")
