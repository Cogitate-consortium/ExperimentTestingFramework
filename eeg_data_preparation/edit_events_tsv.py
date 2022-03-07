import os
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne_bids import BIDSPath

SUB_LIST = ["01"]
session = None
datatype = "eeg"
task_name = "rsvp"
relevant_events = "Event/E  1"


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


def check_diff_diff(ts1, ts2, save_root, ts1_name="", ts2_name=""):
    """
    This function performs the diff of diff to ensure that the alignment between the two time lines is reasonable!
    :param ts1: (np array of floats) time stamps of the first time series
    :param ts2: (np array) time stamp of the second time series
    :param save_root: (string or pathlib object) root to where the data should be saved
    :param ts1_name: (string) name of the first time series, useful for plotting and printing, to know what's what
    :param ts2_name: (string) name of the second time series, useful for plotting and printing, to know what's what
    :return: None
    """
    print("-"*40)
    print("Welcome to check diff of diff!")
    # Check that the numbers match:
    if len(ts1) != len(ts2):
        raise Exception("The number of events in {0} {1} "
                        "\nThe number of events in ts2 {2}, {3}!"
                        "\nAlignment is not possible! You need to "
                        "review the events TSV and clean it!".format(ts1_name, len(ts1), ts2_name, len(ts2)))
    # Check that the diff of diff is within an acceptable range:
    ts1_diff = np.diff(ts1)
    ts2_diff = np.diff(ts2)
    # Compute the diff of diff:
    diff_of_diff = ts1_diff - ts2_diff
    # Print summary of the results to the command line:
    print("Max diff: {:.3f}".format(max(diff_of_diff)))
    print("Mean diff: {:.3f}".format(np.mean(diff_of_diff)))
    # Plot the two time series diff on top of another for later inspections:
    plt.plot(ts1_diff, color="r", label=ts1_name)
    plt.plot(ts2_diff, color="b", label=ts2_name)
    plt.legend()
    plt.title("Interval between time stamps in \n{0} and {1}".format(ts1_name, ts2_name))
    # Saving the results:
    file_name = os.path.join(save_root, "triggers_alignment_plot.png")
    plt.savefig(file_name, transparent=True)
    plt.close()
    # Plot the distribution:
    plt.hist(diff_of_diff)
    plt.title("Distribution of jitter {0} - {1}".format(ts1_name, ts2_name))
    # Saving the results:
    file_name = os.path.join(save_root, "triggers_alignment_distribution.png")
    plt.savefig(file_name, transparent=True)
    plt.close()

    return None


def align_events(bids_root, sub_id):
    """
    This script reads in the events tsv as well as the eeg data to align the time stamp of the tsv to the triggers found
    in the eeg file. In the openneuro ds003825 dataset, the events tsv time line is from the experimental computer.
    In order to be able to analyze the data with their respective events, alignment needs to occur. To that end, the
    events tsv from the bids data set gets updated such that the events time stamps are in the eeg timeline
    :param bids_root: (string or pathlib object) root to the bids data set
    :param sub_id: (string) id of a given subject
    :return:
    events_df: (pd dataframe) updated events time stamps
    events_tsv_file: (string) full path to the events tsv object, to overwrite it
    """
    print("-"*40)
    print("Welcome to align_events!")
    print("Aligning events tsv for sub-{0}".format(sub_id))
    # prepare the bids path:
    bids_path = BIDSPath(root=bids_root, subject=sub_id,
                         session=session,
                         datatype=datatype,
                         task=task_name)
    # Loading the raw files (without using the mne bids read raw, because as of now the events tsv is not aligned
    # to the signal)
    raw = mne.io.read_raw(bids_path)
    # Now, fetching the tsv file:
    events_tsv_file = glob.glob("{0}/{1}".format(bids_path.directory, "*events.tsv"))
    # Perform a few sanity checks
    if len(events_tsv_file) == 0:
        raise Exception("No events tsv files were found for this subject!")
    if len(events_tsv_file) > 1:
        raise Exception("Several events tsv were found for this subject!")
    # Loading the events tsv:
    events_df = pd.read_csv(events_tsv_file[0], sep='\t')  # Loading the info
    # Now, aligning the time stamps of the events tsv to the raw:
    signal_events_ind = np.where(raw.annotations.description == relevant_events)[0]
    signal_timestamps_sec = raw.annotations.onset[signal_events_ind]
    events_tsv_ts_sec = events_df["onset"].values / 1000

    # Plotting the diff of diff to make sure that alignment is reasonable:
    check_diff_diff(signal_timestamps_sec, events_tsv_ts_sec,
                    save_root=bids_path.directory,
                    ts1_name="signal_timestamps_sec",
                    ts2_name="event_tsv_timestamps")
    # Replacing the onset in the events df by the time stamps found in the signal:
    events_df["onset"] = signal_timestamps_sec
    # Setting any temporal information to seconds to ensure that units are consistent between mne object and logs:
    events_df["duration"] = events_df["duration"].values / 1000

    return events_df, events_tsv_file[0]


def edit_events_desc(events_new_ts):
    """

    :param events_new_ts:
    :return:
    """
    # Get the column of interest:
    col = [col for col in events_new_ts.columns if col not in ["onset", "duration"]]
    # Convert rows from the col of interest to forward slash separated:
    new_evt_desc = ["/".join(row[col].apply(str).to_list()) for ind, row in events_new_ts.iterrows()]
    # Now recreating the new events:
    new_events_df = pd.DataFrame({
        "onset": events_new_ts["onset"].values,
        "duration": events_new_ts["onset"].values,
        "trial_type": new_evt_desc
    })

    return new_events_df


def manage_edit_events_tsv(bids_root, overwrite=False):
    """
    This function loops through the participants found in the bids data set, to update the events tsv file such that
    the time in them stems from the signal
    :return: None
    """
    # Listing all the participants in the bids root:
    subject_list = list_subjects(bids_root, prefix="sub-")
    # Looping through each:
    for sub in subject_list:
        # First, align the events from the signal to the log files:
        events_new_ts, tsv_file = align_events(bids_root, sub)
        # Then, rework the events descriptions to be more readily compliant with out preprocessing pipeline:
        events_new_desc = edit_events_desc(events_new_ts)
        # And now, overwriting the events tsv
        if overwrite:
            events_new_desc.to_csv(tsv_file, sep="\t", index=False)
    return None


if __name__ == "__main__":
    manage_edit_events_tsv(bids_root="/mnt/beegfs/XNAT/workspace/001_testing_framework/data/bids", overwrite=True)
