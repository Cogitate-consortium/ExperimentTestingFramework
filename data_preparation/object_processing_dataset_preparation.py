from general_utilities.path_helper_function import list_subjects
from pathlib import Path
import os
import scipy.io
import mne
import numpy as np
import pandas as pd

ses = "1"
data_type = "eeg"
task = "object_processing"

category_dict = {
    1: "human_body",
    2: "human_face",
    3: "animal_body",
    4: "animal_face",
    5: "fruit_vegetable",
    6: "inanimate_object"
}
trial_duration = 0.5


def mat2mne(matfile):
    """
    This function converts the .mat file epoch eeg_mat from this eeg_mat set: https://purl.stanford.edu/bq914sc3730
    into mne epochs
    :param matfile: (Path) full path to the math file
    :return:
    """
    print("=" * 40)
    print("Converting the file: {}".format(matfile))
    print("To an mne epoch")
    eeg_mat = scipy.io.loadmat(matfile)
    # Extract the relevant info:
    n_channels = eeg_mat["X_3D"].shape[0]
    sfreq = eeg_mat["Fs"][0][0]
    ch_names = [f'EEG{n:02}' for n in range(1, n_channels + 1)]
    ch_types = "eeg"
    # Create the info:
    info = mne.create_info(ch_names, sfreq, ch_types=ch_types)
    info["description"] = "Object processing dataset"

    # Create annotations:
    onset_samp = []
    description = []
    trials_categories = np.squeeze(eeg_mat["categoryLabels"])
    trials_examplar = np.squeeze(eeg_mat["exemplarLabels"])
    samp_ctr = 0
    for i in range(eeg_mat["categoryLabels"].shape[1]):
        # Generate trial desc:
        description.append("{}/examplar_{}".format(category_dict[trials_categories[i]], trials_examplar[i]))
        # We are here considering that the trials were happening directly after another:
        onset_samp.append(samp_ctr)
        samp_ctr += eeg_mat["X_3D"].shape[1]
    # Convert to numpy arrays:
    onset_samp = np.array(onset_samp)
    description = np.array(description)
    # Extract unique descriptions from the bunch:
    unique_desc = np.unique(description)
    # Generate a UID dict:
    event_dict = {val: i for i, val in enumerate(unique_desc)}
    # Create the events:
    events_uid = [int(event_dict[val]) for val in description]
    events = np.column_stack((onset_samp,
                              np.zeros(eeg_mat["X_3D"].shape[2], dtype=int),
                              np.array(events_uid)))

    # Extract the data:
    data = np.reshape(eeg_mat["X_3D"], (eeg_mat["X_3D"].shape[2], eeg_mat["X_3D"].shape[0], eeg_mat["X_3D"].shape[1]))

    # Generate the epochs:
    epochs = mne.EpochsArray(data, info, tmin=0, events=events, event_id=event_dict)

    # Adding metadata:
    metadata = pd.DataFrame(data=[val.split("/") for val in description], columns=["category", "examplar"],
                            index=range(len(epochs)))
    epochs.metadata = metadata

    return epochs


def conversion_manager(root):
    # List all the subjects:
    sub_list = list_subjects(root, prefix="sub-")
    # For each subject, converting the data to mne epoch object:
    for sub in sub_list:
        sub_epochs = Path(root, "sub-" + sub, sub + ".mat")
        mne_epochs = mat2mne(str(sub_epochs))
        # Set the path to save this subject data:
        save_path = Path(root, "derivatives", "preprocessing", "sub-" + sub, "ses-" + ses, data_type, "epoching",
                         "erp", "unknown")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Generate the file name:
        file_name = "sub-{}_ses-{}_task-{}_desc-epoching_{}-epo.fif".format(sub, ses, task, data_type)
        # Save the epoch:
        mne_epochs.save(Path(save_path, file_name))


if __name__ == "__main__":
    conversion_manager(root=r"C:\Users\alexander.lepauvre\Documents\PhD\Experimental_testing_framework\data\object_processing\bids")
