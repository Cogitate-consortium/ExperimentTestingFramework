import json
import subprocess
import shutil

from pathlib import Path

from utilities import find_files


class EpochAnalysisClass:
    """
    This class creates the analysis parameters object, based on a json file.
    :return:
    """

    def __init__(self, json_file, sub_id=None):

        # Loading the json dict:
        with open(json_file, 'r') as fp:
            json_dict = json.load(fp)

        # First things first, adding the git id and url to keep track of things. But if git is not installed, unknown:
        try:
            self.github_repo_url = str(subprocess.check_output(
                ['git', 'config', '--get', 'remote.origin.url']))
            self.git_commit_id = str(
                subprocess.check_output(['git', 'rev-parse', 'HEAD']))
        except FileNotFoundError:
            self.github_repo_url = "unknown"
            self.git_commit_id = "unknown"

        # --------------------------------------------------------------------------------------------------------------
        # Paths to the raw data and to the BIDS root
        self.raw_root = json_dict['raw_root']
        self.BIDS_root = json_dict['BIDS_root']
        # --------------------------------------------------------------------------------------------------------------
        # File naming parameters for BIDS
        self.session = json_dict['session']
        self.task_name = json_dict['task_name']
        self.data_type = json_dict['data_type']
        # --------------------------------------------------------------------------------------------------------------
        # Info about which data to load
        self.preprocessing_folder = json_dict['preprocessing_folder']
        self.preprocess_steps = json_dict['preprocess_steps']

        # --------------------------------------------------------------------------------------------------------------
        # Getting the analysis parameters
        self.analysis_parameters = json_dict['analysis_parameters']

        # Variable to store which file was taken in
        self.input_file = ""

        # Add the participant ID:
        self.SUBJ_ID = sub_id

        # Setting the root for the produced files:
        self.save_root = str(Path(self.BIDS_root, "derivatives", "components"))
        # Adding the root to the input file:
        self.input_file_root = str(Path(self.BIDS_root, "derivatives", "preprocessing", "sub-" + self.SUBJ_ID,
                                        self.data_type, self.preprocessing_folder))
        # Creating the files prefix
        self.files_prefix = "sub-" + self.SUBJ_ID + "_task-" + self.task_name \
                            + "_analysis-vis_resp_"

    def save_parameters(self, save_path):
        """
        This function saves the analysis parameters to a json file in the specified directory. It also fetches the
        preprocessing parameters file and the subject info and copies it in the specified directory
        :param save_path: (string or Path object) Path where the data should be saved
        :return:
        """

        # Converting the object to dict:
        obj_dict = self.__dict__

        # Setting the full file name:
        full_file = Path(save_path, "sub-" + self.SUBJ_ID + "_ses-" + self.session + "_task-" + self.task_name
                         + "_desc-" + "visual_responsivness_analysis_parameters.json")
        # Dumping to a json file
        with open(full_file, "w") as fp:
            json.dump(obj_dict, fp, indent=4)

        # Fetching the preprocessing parameters from the loaded file:
        # In the analysis parameters, we always save the parameters file along side the saved data. Therefore, we
        # search the directory from which we loaded the data for the json parameters file:
        preprocessing_parameters_file = find_files(Path(self.input_file).parents[0],
                                                   naming_pattern="*Preprocessing_parameters", extension=".json")[0]
        # Get the name of the parameters file to copy it to the right place with the right name:
        preprocessing_parameters_file_name = Path(
            preprocessing_parameters_file).name
        # Then copying that file to the set directory:
        shutil.copy(preprocessing_parameters_file, Path(
            save_path, preprocessing_parameters_file_name))
        # Same for the subject info file:
        subject_info_file = find_files(Path(self.BIDS_root, "derivatives", "preprocessing", "sub-" + self.SUBJ_ID),
                                       naming_pattern="*SubjectInfo", extension=".json")[0]
        # Get the name of the parameters file to copy it to the right place with the right name:
        subject_info_file_name = Path(subject_info_file).name

        # Then copying that file to the set directory:
        shutil.copy(subject_info_file, Path(save_path, subject_info_file_name))
