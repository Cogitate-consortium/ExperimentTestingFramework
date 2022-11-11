import numpy as np
import mne
import math
from scipy.stats import norm


def approximate_sigma(x, proba, loc=0):
    """

    :param x:
    :param proba:
    :param loc:
    :return:
    """
    canditate_sigma = np.linspace(0.01, 100, num=10000)
    obs_proba = np.zeros(canditate_sigma.shape)
    for ind, sig in enumerate(canditate_sigma):
        obs_proba[ind] = norm.cdf(-x, loc=loc, scale=sig)
    # Find the sigma that leads to the proba closest to what we are after:
    sigma = canditate_sigma[np.where(np.abs(obs_proba - proba) == np.min(np.abs(obs_proba - proba)))][0]
    return sigma


def generate_jitter(n_trials, refresh_rate=16, trials_proportion=0.1, tail="both", max_jitter=None):
    """

    :param n_trials:
    :param refresh_rate:
    :param trials_proportion:
    :param tail:
    :param max_jitter:
    :return:
    """
    # Compute the sigma of a null distribution fitting the given parameters:
    if tail != "both":
        sigma = approximate_sigma(refresh_rate, trials_proportion)
    else:
        sigma = approximate_sigma(refresh_rate, trials_proportion / 2)
    # Generate trials jitter:
    trials_jitter_cont = np.random.normal(loc=0, scale=sigma, size=n_trials)

    # Correct the jitters to be multiple of the refresh rate:
    trials_jitter_disc = np.array([math.floor(jitter / refresh_rate) * refresh_rate if jitter > 0
                                   else (math.floor(jitter / refresh_rate) + 1) * refresh_rate
                                   for jitter in trials_jitter_cont])
    # If there is a max jitter:
    if max_jitter is not None:
        trials_jitter_disc[np.where(trials_jitter_disc > max_jitter)] = max_jitter
        trials_jitter_disc[np.where(trials_jitter_disc < -max_jitter)] = -max_jitter
    # Handle the tails and make sure that the proportion roughly matches our expectations:
    if tail == "upper":
        trials_jitter_disc[np.where(trials_jitter_disc < 0)] = 0
    elif tail == "lower":
        trials_jitter_disc[np.where(trials_jitter_disc > 0)] = 0
    # Make sure that the proportion of trials for which there are jitter matches our expectations
    assert np.abs(trials_proportion - (len(np.where(np.abs(trials_jitter_disc) >= refresh_rate)[0]) /
                                       len(trials_jitter_disc))) < 0.1, \
        "The proportion of jittered trials does not match expectations!"

    return trials_jitter_disc


def jitter_trials(epochs, refresh_rate=16, trials_proportion=0.1, tail="both", max_jitter=None):
    """
    This function generates random jitter on the stimulus onset, simulating experiments malfunctions. The refresh rate
    specifies the amplitude of the jitter (only multiple of the refresh rate are possible). The trial proportion
    specifies the proportion of trials that has a jitter. In other words, when specifying a trial proportion of 10%,
    this means that the probability for each trial to have a jitter of 16ms or above is 10%, following a normal
    distribution.
    :param epochs: (mne epochs object) contains trials for which to inject jitter
    :param refresh_rate: (int) refresh rate of the screen used in experiment to simulate jitter. Jitter can only
    be a multiple of the refresh rate.
    :param trials_proportion: (float) proportion of trials (between 0 and 1) for which to add jitter
    :param tail: (string) tail of the distribution to simulate the jitter. If both, jitter can be both positive and
    negative. If upper, strictly positive. If lower, only negative
    :param max_jitter: (int) max jitter possible. If max jitter is 16ms, jitters cannot exceed +- 16ms.
    :return:
    """
    # Generate random jitters for each trials:
    trials_jitter_ms = generate_jitter(len(epochs), refresh_rate=refresh_rate,
                                       trials_proportion=trials_proportion,
                                       tail=tail, max_jitter=max_jitter)

    # Convert ms to samples:
    trials_jitter_samp = np.array([jitter * (epochs.info["sfreq"] / 1000) for jitter in trials_jitter_ms])

    # Get the epochs data:
    data = epochs.get_data()

    # Generate a new time axis to account for the most extreme jitters:
    min_jitter, max_jitter = np.min(trials_jitter_samp), np.max(trials_jitter_samp)
    if min_jitter < 0:
        new_times = epochs.times[int(0 + max_jitter): int(- np.abs(min_jitter))]
    else:
        new_times = epochs.times[int(0 + max_jitter):]
    # Attributing the jitter to each trial:
    trials_data_jittered = []
    trial_times = []
    # Loop through each trial:
    for ind in range(data.shape[0]):
        # Get the jitter of this particular trial:
        jitter = trials_jitter_samp[ind]
        # Here it gets a bit counter intuitive. It is the same as when changing from summer to winter hour: we are
        # advancing our clocks by one hour, therefore it is one earlier. Wait what, no it's the other way around, right?
        # Every one gets confused twice a year. Same here.
        # When we have a positive jitter, that means that the stimulus was presented later than what we initially
        # thought. That means that t0 is actually t1. This is simulated by getting rid of the last samples (shifting
        # the data to the right) while shifting time forward (i.e. shifting time to the left).
        if jitter > 0:
            trials_data_jittered.append(data[ind, :, :int(-jitter)])
            trial_times.append(epochs.times[int(jitter):])
        elif jitter < 0:  # And the opposit is true for negative jitter: the stimulus occured earlier than we think
            trials_data_jittered.append(data[ind, :, np.abs(int(jitter)):])
            trial_times.append(epochs.times[:int(jitter)])
        else:
            trials_data_jittered.append(data[ind, :, :])
            trial_times.append(epochs.times)
    # Creating an array to store the new data:
    new_data = np.zeros([len(trials_data_jittered), len(epochs.ch_names), len(new_times)])
    # Looping through each trials to get the data to the new timeline:
    for trial_ind in range(new_data.shape[0]):
        samp_ind = [ind for ind in range(len(trial_times[trial_ind])) if trial_times[trial_ind][ind] in new_times]
        new_data[trial_ind, :, :] = trials_data_jittered[trial_ind][:, samp_ind]

    # Putting everything back into an epochs object:
    metadata = epochs.metadata
    epochs = mne.EpochsArray(new_data, epochs.info, tmin=new_times[0], events=epochs.events,
                             event_id=epochs.event_id)
    epochs.metadata = metadata

    return epochs


def generate_jitter_old(epochs, jitter_amp_ms=16, trials_proportion=0.1):
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
    """
    This function randomly shuffles the triggers of a specified proportion of trials in the mne epochs object. This
    is useful to
    :param epochs:
    :param trials_proportion:
    :return:
    """
    # Get the indices of all trials:
    all_trials_inds = np.arange(0, epochs.events.shape[0])
    # Randmoly subsample a set of trials to shuffle:
    subsample_inds = np.sort(np.random.randint(0, high=epochs.events.shape[0],
                                               size=int(trials_proportion * epochs.events.shape[0])))
    # Randomly shuffle the subset:
    shuffled_subsample = np.random.permutation(subsample_inds)
    new_ind = []
    # Loop through each trial:
    for ind in all_trials_inds:
        # If the current index is one that was shuffled:
        if ind in subsample_inds:
            # Find the trial index within the subsample:
            subsample_ind = np.where(subsample_inds == ind)
            new_ind.append(shuffled_subsample[subsample_ind][0])
        else:
            new_ind.append(ind)

    # Now, reorganize the events and meta data accordingly:
    new_events = epochs.events[new_ind, 2]
    epochs.events[:, 2] = new_events
    new_metadata = epochs.metadata.iloc[new_ind].reset_index(drop=True)
    epochs.metadata = new_metadata

    return epochs
