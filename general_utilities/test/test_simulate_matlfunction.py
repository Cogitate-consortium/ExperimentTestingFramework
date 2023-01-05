import unittest
import mne
import numpy as np
from general_utilities.simulate_malfunction import jitter_trials


def create_onset_epochs(sfreq=1000, onset_time=0, offset_time=0.1, n_trials=500, tmin=-0.2, tmax=2.0):
    """
    This function creates an epochs object with an onset and offset at precise time points to allow testing whether
    the introduction of a jitter worked as intended.
    :param sfreq:
    :param onset_time:
    :param offset_time:
    :param n_trials:
    :param tmin:
    :param tmax:
    :return:
    """
    # Create the info for the channels objects:
    info = mne.create_info(ch_names=['ch1'],
                           ch_types=['eeg'],
                           sfreq=sfreq)
    # Generate the data:
    data = np.zeros([n_trials, 1, int((tmax - tmin) * sfreq)])
    # Convert the onset and offset to samples:
    onset_samp, offset_samp = int((onset_time - tmin) * sfreq), int((offset_time - tmin) * sfreq)
    # Add the one to this time window
    data[:, :, onset_samp:offset_samp] = 1
    # Create epochs:
    epochs = mne.EpochsArray(data, info, tmin=tmin)

    return epochs


class TestJitterTrials(unittest.TestCase):

    def test_jitter_positive(self):
        # Set the epochs parameters:
        sfreq = 1000
        onset_time, offset_time = 0, 0.1
        n_trials = 500
        tmin, tmax = -0.2, 2.0
        # Create the epochs:
        epochs = create_onset_epochs(sfreq=sfreq, onset_time=onset_time, offset_time=offset_time,
                                     n_trials=n_trials, tmin=tmin, tmax=tmax)
        # Set the jitter parameters:
        refresh_rate = 16
        trials_proportion = 0.1
        # Jitter the epochs
        epochs = jitter_trials(epochs, refresh_rate=refresh_rate, trials_proportion=trials_proportion,
                               tail="upper", max_jitter=16)

        # Convert the refresh rate into data samples:
        onset_sec_jittered = onset_time + refresh_rate * 10**-3
        # Find the onset of each trial:
        jittered_onsets_samp = np.squeeze(np.argmax(epochs.get_data() == 1, axis=2))
        # Check what the time these onsets corresponds to:
        jittered_onsets_time = np.squeeze(np.array([epochs.times[samp] for samp in jittered_onsets_samp]))
        # Find if there are any samples that are not equal to either 0 or 16ms:
        self.assertTrue(len(np.where([jitter not in [0, onset_sec_jittered]
                                      for jitter in jittered_onsets_time])[0]) == 0)
        # Make sure that the number of trials that are jittered roughly matches expectations
        obs_jitter = len(np.where(jittered_onsets_time >= onset_sec_jittered)[0]) / len(jittered_onsets_time)
        self.assertTrue(obs_jitter - trials_proportion < 0.1)

    def test_jitter_negative(self):
        # Set the epochs parameters:
        sfreq = 1000
        onset_time, offset_time = 0, 0.1
        n_trials = 500
        tmin, tmax = -0.2, 2.0
        # Create the epochs:
        epochs = create_onset_epochs(sfreq=sfreq, onset_time=onset_time, offset_time=offset_time,
                                     n_trials=n_trials, tmin=tmin, tmax=tmax)
        # Set the jitter parameters:
        refresh_rate = 16
        trials_proportion = 0.1
        # Jitter the epochs
        epochs = jitter_trials(epochs, refresh_rate=refresh_rate, trials_proportion=trials_proportion,
                               tail="lower", max_jitter=16)

        # Convert the refresh rate into data samples:
        onset_sec_jittered = onset_time - refresh_rate * 10**-3
        # Find the onset of each trial:
        jittered_onsets_samp = np.squeeze(np.argmax(epochs.get_data() == 1, axis=2))
        # Check what the time these onsets corresponds to:
        jittered_onsets_time = np.squeeze(np.array([epochs.times[samp] for samp in jittered_onsets_samp]))
        # Find if there are any samples that are not equal to either 0 or 16ms:
        self.assertTrue(len(np.where([jitter not in [0, onset_sec_jittered]
                                      for jitter in jittered_onsets_time])[0]) == 0)
        # Make sure that the number of trials that are jittered roughly matches expectations
        obs_jitter = len(np.where(jittered_onsets_time <= onset_sec_jittered)[0]) / len(jittered_onsets_time)
        self.assertTrue(obs_jitter - trials_proportion < 0.1)

    def test_jitter_two_tails(self):
        # Set the epochs parameters:
        sfreq = 1000
        onset_time, offset_time = 0, 0.1
        n_trials = 500
        tmin, tmax = -0.2, 2.0
        # Create the epochs:
        epochs = create_onset_epochs(sfreq=sfreq, onset_time=onset_time, offset_time=offset_time,
                                     n_trials=n_trials, tmin=tmin, tmax=tmax)
        # Set the jitter parameters:
        refresh_rate = 16
        trials_proportion = 0.1
        # Jitter the epochs
        epochs = jitter_trials(epochs, refresh_rate=refresh_rate, trials_proportion=trials_proportion,
                               tail="both", max_jitter=16)

        # Convert the refresh rate into data samples:
        onset_sec_jittered = [onset_time - refresh_rate * 10**-3, onset_time + refresh_rate * 10**-3]
        # Find the onset of each trial:
        jittered_onsets_samp = np.squeeze(np.argmax(epochs.get_data() == 1, axis=2))
        # Check what the time these onsets corresponds to:
        jittered_onsets_time = np.squeeze(np.array([epochs.times[samp] for samp in jittered_onsets_samp]))
        # Find if there are any samples that are not equal to either 0 or 16ms:
        self.assertTrue(len(np.where([jitter not in [0, *onset_sec_jittered]
                                      for jitter in jittered_onsets_time])[0]) == 0)
        # Make sure that the number of trials that are jittered roughly matches expectations
        obs_pos_jitter = len(np.where(jittered_onsets_time >= onset_sec_jittered[1])[0]) / len(jittered_onsets_time)
        obs_neg_jitter = len(np.where(jittered_onsets_time <= onset_sec_jittered[0])[0]) / len(jittered_onsets_time)
        self.assertTrue(obs_pos_jitter - trials_proportion / 2 < 0.1)
        self.assertTrue(obs_neg_jitter - trials_proportion / 2 < 0.1)

    def test_jitter_positive_no_max(self):
        # Set the epochs parameters:
        sfreq = 1000
        onset_time, offset_time = 0, 0.1
        n_trials = 500
        tmin, tmax = -0.2, 2.0
        # Create the epochs:
        epochs = create_onset_epochs(sfreq=sfreq, onset_time=onset_time, offset_time=offset_time,
                                     n_trials=n_trials, tmin=tmin, tmax=tmax)
        # Set the jitter parameters:
        refresh_rate = 16
        trials_proportion = 0.2
        # Jitter the epochs
        epochs = jitter_trials(epochs, refresh_rate=refresh_rate, trials_proportion=trials_proportion,
                               tail="upper", max_jitter=None)

        # Convert the refresh rate into data samples:
        onset_sec_jittered = [onset_time + refresh_rate * i * 10**-3 for i in range(6)]
        # Find the onset of each trial:
        jittered_onsets_samp = np.squeeze(np.argmax(epochs.get_data() == 1, axis=2))
        # Check what the time these onsets corresponds to:
        jittered_onsets_time = np.squeeze(np.array([epochs.times[samp] for samp in jittered_onsets_samp]))
        # Find if there are any samples that are not equal to either 0 or 16ms:
        self.assertTrue(len(np.where([jitter not in [0, *onset_sec_jittered]
                                      for jitter in jittered_onsets_time])[0]) == 0)
        # Make sure that the number of trials that are jittered roughly matches expectations
        obs_jitter = len(np.where(jittered_onsets_time >= onset_sec_jittered[1])[0]) / len(jittered_onsets_time)
        self.assertTrue(obs_jitter - trials_proportion < 0.1)

    def test_jitter_negative_no_max(self):
        # Set the epochs parameters:
        sfreq = 1000
        onset_time, offset_time = 0, 0.1
        n_trials = 500
        tmin, tmax = -0.2, 2.0
        # Create the epochs:
        epochs = create_onset_epochs(sfreq=sfreq, onset_time=onset_time, offset_time=offset_time,
                                     n_trials=n_trials, tmin=tmin, tmax=tmax)
        # Set the jitter parameters:
        refresh_rate = 16
        trials_proportion = 0.2
        # Jitter the epochs
        epochs = jitter_trials(epochs, refresh_rate=refresh_rate, trials_proportion=trials_proportion,
                               tail="lower", max_jitter=None)

        # Convert the refresh rate into data samples:
        onset_sec_jittered = [onset_time - refresh_rate * i * 10**-3 for i in range(6)]
        # Find the onset of each trial:
        jittered_onsets_samp = np.squeeze(np.argmax(epochs.get_data() == 1, axis=2))
        # Check what the time these onsets corresponds to:
        jittered_onsets_time = np.squeeze(np.array([epochs.times[samp] for samp in jittered_onsets_samp]))
        # Find if there are any samples that are not equal to either 0 or 16ms:
        self.assertTrue(len(np.where([jitter not in [0, *onset_sec_jittered]
                                      for jitter in jittered_onsets_time])[0]) == 0)
        # Make sure that the number of trials that are jittered roughly matches expectations
        obs_jitter = len(np.where(jittered_onsets_time <= onset_sec_jittered[1])[0]) / len(jittered_onsets_time)
        self.assertTrue(obs_jitter - trials_proportion < 0.1)

    def test_jitter_two_tails_no_max(self):
        # Set the epochs parameters:
        sfreq = 1000
        onset_time, offset_time = 0, 0.1
        n_trials = 500
        tmin, tmax = -0.2, 2.0
        # Create the epochs:
        epochs = create_onset_epochs(sfreq=sfreq, onset_time=onset_time, offset_time=offset_time,
                                     n_trials=n_trials, tmin=tmin, tmax=tmax)
        # Set the jitter parameters:
        refresh_rate = 16
        trials_proportion = 0.2
        # Jitter the epochs
        epochs = jitter_trials(epochs, refresh_rate=refresh_rate, trials_proportion=trials_proportion,
                               tail="both", max_jitter=None)

        # Convert the refresh rate into data samples:
        onset_sec_jittered_pos = [onset_time + refresh_rate * i * 10**-3 for i in range(6)]
        onset_sec_jittered_neg = [onset_time - refresh_rate * i * 10**-3 for i in range(6)]
        # Find the onset of each trial:
        jittered_onsets_samp = np.squeeze(np.argmax(epochs.get_data() == 1, axis=2))
        # Check what the time these onsets corresponds to:
        jittered_onsets_time = np.squeeze(np.array([epochs.times[samp] for samp in jittered_onsets_samp]))
        # Find if there are any samples that are not equal to either 0 or 16ms:
        self.assertTrue(len(np.where([jitter not in [0, *onset_sec_jittered_pos, *onset_sec_jittered_neg]
                                      for jitter in jittered_onsets_time])[0]) == 0)
        # Make sure that the number of trials that are jittered roughly matches expectations
        obs_pos_jitter = len(np.where(jittered_onsets_time >= onset_sec_jittered_pos[1])[0]) / len(jittered_onsets_time)
        obs_neg_jitter = len(np.where(jittered_onsets_time <= onset_sec_jittered_neg[1])[0]) / len(jittered_onsets_time)
        self.assertTrue(obs_pos_jitter - trials_proportion / 2 < 0.1)
        self.assertTrue(obs_neg_jitter - trials_proportion / 2 < 0.1)
