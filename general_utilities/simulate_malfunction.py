import numpy as np
import mne


# def jitter_from_distribution():


def generate_jitter(epochs, jitter_amp_ms=16, trials_proportion=0.1):
    """
    This function enables introducing temporal jitter to mne epochs object according to a few set of parameters
    :param epochs: (mne epochs object) mne epochs object for which to introduce jitter
    :param jitter_amp_ms: (int) how many milliseconds should the simulated skipped frames have (so if you say 16ms,
    a set proportion of trials will have an onset off by +-16ms)
    :param trials_proportion: (float) proportion of trials that should be affected by the jitter
    :return:
    """
    # Convert the jitter from ms to samples:
    jitter_samp = int(jitter_amp_ms * (epochs.info["sfreq"] / 1000))
    if jitter_samp == 0:
        print("WARNING: You have passed a jitter corresponds to less than a sample at this sampling rate!"
              "\nThe jitter will be set to 1 sample!")
        jitter_samp = 1

    # Get the data:
    data = epochs.get_data()
    data_new = np.zeros((data.shape[0], data.shape[1], data.shape[2] - jitter_samp))
    # Randomly pick trials for which to mess up the timing:
    trials_ind = np.random.randint(0, high=data.shape[0], size=int(trials_proportion * data.shape[0]))
    # For these trials, removing n samples from the beginning:
    data_new[trials_ind, :, :] = data[trials_ind, :, jitter_samp:]
    mask = np.ones(data.shape[0], bool)
    mask[trials_ind] = False
    data_new[mask, :, :] = data[mask, :, :-jitter_samp]

    # Recreate the epoch object:
    metadata = epochs.metadata
    epochs = mne.EpochsArray(data_new, epochs.info, tmin=epochs.times[0], events=epochs.events,
                             event_id=epochs.event_id)
    epochs.metadata = metadata
    return epochs


def shuffle_triggers(epochs, trials_proportion=0.05):
    # Get the indices of all trials:
    all_trials_ind = np.arange(0, epochs.events.shape[0])
    # Randomly pick trials indices to mix up:
    to_shuffle_ind = np.random.randint(0, high=epochs.events.shape[0],
                                       size=int(trials_proportion * epochs.events.shape[0]))
    # Randomly shuffle their order:
    shuffle_ind = np.random.randint(0, high=to_shuffle_ind.shape[0],
                                    size=to_shuffle_ind.shape[0])
    new_ind = []
    for ind in all_trials_ind:
        if ind in to_shuffle_ind:
            new_ind.append(to_shuffle_ind[shuffle_ind][np.where(to_shuffle_ind == ind)][0])
        else:
            new_ind.append(ind)

    # Now, reorganize the events and meta data accordingly:
    new_events = epochs.events[new_ind, 2]
    epochs.events[:, 2] = new_events
    new_metadata = epochs.metadata.iloc[new_ind].reset_index(drop=True)

    epochs.events = new_events
    epochs.metadata = new_metadata

    return epochs
