import glob
import os
from mne.baseline import rescale
from pathlib import Path


def find_files(root, naming_pattern=None, extension=None):
    """
    This function finds files matching a specific naming pattern recursively in directory from root
    :param root: root of the directory among which to search. Must be a string
    :param naming_pattern: string the files must match to be returned
    :param extension: Extension of the file to search for
    :return:
    """
    print("-"*40)
    print("Welcome to find files")
    if extension is None:
        extension = '.*'
    if naming_pattern is None:
        naming_pattern = '*'

    matches = []
    for sub_folder, dirnames, filenames in os.walk(root):
        for filename in glob.glob(sub_folder + os.sep + '*' + naming_pattern + '*' + extension):
            matches.append(os.path.join(sub_folder, filename))
    print("The following files were found: ")
    [print(match) for match in matches]

    return matches


def baseline_scaling(epochs, correction_method="ratio", baseline=(None, 0), picks=None, n_jobs=1):
    """
    This function performs baseline correction on the data. The default is to compute the mean over the entire baseline
    and dividing each data points in the entire epochs by it. Another option is to substract baseline from each time
    point
    :param epochs: (mne epochs object) epochs on which to perform the baseline correction
    :param correction_method: (string) options to do the baseline correction. Options are:
        mode : 'mean' | 'ratio' | 'logratio' | 'percent' | 'zscore' | 'zlogratio'
        Perform baseline correction by
        - subtracting the mean of baseline values ('mean')
        - dividing by the mean of baseline values ('ratio')
        - dividing by the mean of baseline values and taking the log
          ('logratio')
        - subtracting the mean of baseline values followed by dividing by
          the mean of baseline values ('percent')
        - subtracting the mean of baseline values and dividing by the
          standard deviation of baseline values ('zscore')
        - dividing by the mean of baseline values, taking the log, and
          dividing by the standard deviation of log baseline values
          ('zlogratio')
          source: https://github.com/mne-tools/mne-python/blob/main/mne/baseline.py
    :param baseline: (tuple) which bit to take as the baseline
    :param picks: (None or list of int or list of strings) indices or names of the channels on which to perform the
    correction. If none, all channels are used
    :param n_jobs: (int) number of jobs to use to run the function. Can be ran in parallel
    :return: none, the data are modified in place
    """
    epochs.apply_function(rescale, times=epochs.times, baseline=baseline, mode=correction_method,
                          picks=picks, n_jobs=n_jobs, )

    return None


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
    :param data: (boolean) whether or not to generate the path for the data:
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


def file_name_generator(save_path, file_prefix, description, file_extension, data_type="ieeg"):
    """
    This function generates full file names according to the cogitate naming conventions:
    :param save_path: (pathlib path object or path string) root path to where the data should be saved
    :param file_prefix: (string) prfix of the file name
    :param description: (string) what some after teh prefix in the file name
    :param file_extension: (string) what comes after the description, if anything
    :param data_type: (string) data type the data are from
    :return: full_file_name (string) file path + file name
    """
    full_file_name = os.path.join(
        save_path, file_prefix + data_type + "_" + description + file_extension)

    return full_file_name


def list_subjects(root, prefix="sub-"):
    """
    This function lists all the "subjects" found in a given folder.
    :param root:
    :param prefix:
    :return:
    """
    list_folders = os.listdir(root)
    subject_list = [folder.split("-")[1]
                    for folder in list_folders if prefix in folder]

    return subject_list