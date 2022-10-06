import os


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