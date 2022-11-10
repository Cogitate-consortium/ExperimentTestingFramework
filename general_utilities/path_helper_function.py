import os
import glob
import numpy as np
from pathlib import Path
import mne


def find_files(root, naming_pattern=None, extension=None):
    """
    This function finds files matching a specific naming pattern recursively in directory from root
    :param root: root of the directory among which to search. Must be a string
    :param naming_pattern: string the files must match to be returned
    :param extension: Extension of the file to search for
    :return:
    """
    if extension is None:
        extension = '.*'
    if naming_pattern is None:
        naming_pattern = '*'

    matches = []
    for sub_folder, dirnames, filenames in os.walk(root):
        for filename in glob.glob(sub_folder + os.sep + '*' + naming_pattern + '*' + extension):
            matches.append(os.path.join(sub_folder, filename))
    # XNAT will mess up the order of files in case there was an abortion, because it will put things in folders called
    # ABORTED. Therefore, the files need to be sorted based on the file names:
    # Getting the file names:
    matches_file_names = [file.split(os.sep)[-1] for file in matches]
    files_order = np.argsort(matches_file_names)
    matches = [matches[ind] for ind in files_order]
    [print(match) for match in matches]

    return matches


def list_subjects(root, prefix="sub-"):
    """
    This function lists all the "subjects" found in a given folder.
    :param root: (string or Pathlib object) root to where the subjects are found
    :param prefix: (string) prefix to the subjects ID
    :return: (list of strings) list of the subjects ID
    """
    list_folders = os.listdir(root)
    subject_list = [folder.split("-")[1]
                    for folder in list_folders if prefix in folder]

    return subject_list


def path_generator(root, analysis=None, preprocessing_steps=None, fig=False, results=False, data=False):
    """
    Generate the path to where the data should be saved
    :param root: (string or pathlib path object) root of where the data should be saved
    :param analysis: (string) name of the analysis. The highest level folder for the saving will be called accordingly
    :param preprocessing_steps: (string) description of the preprocessing steps used to generate the used data to keep
    track of things
    :param fig: (boolean) whether or not the path is for saving figures. If set to false, the stats should be set to
    true
    :param results: (boolean) whether or not the path is for saving statistics. If set to false, the fig should be set
    to true
    :param data: (boolean) whether or not the path is for saving data. If set to false, the fig should be set
    to true
    :return: save_path (Pathlib path object) path to where the data should be saved
    """

    if fig is True and results is False and data is False:
        save_path = Path(root, "figure", analysis, preprocessing_steps)
    elif fig is False and results is True and data is False:
        save_path = Path(root, "results", analysis, preprocessing_steps)
    elif fig is False and results is False and data is True:
        save_path = Path(root, "data", analysis, preprocessing_steps)
    else:
        raise Exception("You attempted to generate a path to save the analysis specifying that it'S neither stats nor "
                        "figure. that doesn't work. Make sure that only one of the two is true")
    # Generating the directory if that doesn't exist
    if not os.path.isdir(save_path):
        # Creating the directory:
        os.makedirs(save_path)

    return save_path


def load_epochs(bids_root, subject, session, data_type, preprocessing_folder, signal, preprocessing_steps, task):
    # Generate file name:
    epoch_file = Path(bids_root, "derivatives", "preprocessing", "sub-" + subject, "ses-" + session, data_type,
                      preprocessing_folder, signal, preprocessing_steps,
                      "sub-{}_ses-{}_task-{}_desc-epoching_{}-epo.fif".format(subject, session, task, data_type))
    # Return the read epochs:
    return mne.read_epochs(epoch_file, verbose="ERROR")
