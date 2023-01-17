import json

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from general_utilities.data_helper_function import create_epochs, compute_t_stat, compute_fsize


def generate_single_trials(times, fsize, components_dict, channel, n_trials, conditions, sigma, sfreq):
    # Preallocate array for data:
    data = np.zeros(
        [n_trials * len(conditions), 1, times.shape[0]])
    # The noise is expressed as a percentage of the variance observed in the noise free data:
    noise_mv = convert_noise(times, components_dict, sigma)
    # Adjusting the mean of each components amplitude to account for the effect size:
    cond_comp_dict = adjust_comp(components_dict, fsize, conditions,
                                 noise_mv)
    # Loop through each condition:
    ctr = 0
    for cond in cond_comp_dict.keys():
        # Generate single trials noise:
        erp = np.array([generate_erp(times, cond_comp_dict[cond],
                                     peak_noise=0)
                        for _ in range(n_trials)])
        # Generate the noise array:
        noise = np.random.normal(0,
                                 scale=noise_mv,
                                 size=erp.shape)
        erp = np.add(erp, noise)
        data[ctr:ctr + n_trials, 0, :] = erp
        ctr += n_trials
    epochs = create_epochs(data, [channel], "eeg", sfreq, conditions, times,
                           n_trials)
    return epochs


def erp_gaussian_model(grand_avg, priors, channel="E77", verbose=True, plot=True):
    """

    :param data:
    :param priors:
    :return:
    """
    # Convert the priors to lists:
    amplitudes = [priors[comp]["amplitude"] for comp in priors.keys()]
    latencies = [priors[comp]["latency"] for comp in priors.keys()]
    sigmas = [priors[comp]["sigma"] for comp in priors.keys()]
    offsets = [priors[comp]["offset"] for comp in priors.keys()]
    # Convert the bounds to list:
    bounds = [[], []]
    for parameter in ["amplitude_limits", "latency_limits", "sigma_limits", "offset_limits"]:
        for component in priors.keys():
            bounds[0].append(priors[component][parameter][0])
            bounds[1].append(priors[component][parameter][1])
    # Fit the model to each channel separately:
    posteriors = {
        component: {
            "amplitude": None,
            "latency": None,
            "sigma": None,
            "offset": None
        }
        for component in priors.keys()
    }
    # Computing the components at the population level:
    popt, pcov = curve_fit(gaussian, grand_avg.times,
                           np.squeeze(grand_avg.copy().pick(channel).data.T),
                           p0=amplitudes + latencies + sigmas + offsets,
                           bounds=bounds, method="trf")
    # Compute the R^2:
    yfit = gaussian(grand_avg.times, *popt)
    ss_res = np.sum((np.squeeze(grand_avg.copy().pick(channel).data.T) - yfit) ** 2)
    ss_tot = np.sum((np.squeeze(grand_avg.copy().pick(channel).data.T) -
                     np.mean(np.squeeze(grand_avg.copy().pick(channel).data.T))) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    # Convert the fitted parameters to a dict:
    for ind, component in enumerate(posteriors.keys()):
        posteriors[component]["amplitude"] = popt[ind]
        posteriors[component]["latency"] = popt[ind + len(posteriors)]
        posteriors[component]["sigma"] = popt[ind + len(posteriors) * 2]
        posteriors[component]["offset"] = popt[ind + len(posteriors) * 3]
    if verbose:
        for component in posteriors:
            print("Channel: {}, comp: {} :".format(channel, component))
            print("   Amplitude: {}".format(posteriors[component]["amplitude"]))
            print("   Latency: {}".format(posteriors[component]["latency"]))
            print("   Sigma: {}".format(posteriors[component]["sigma"]))
            print("   r2: {}".format(r2))
    # Save the posterior to a file:
    with open('components_posteriors.json', 'w') as f:
        json.dump(posteriors, f)

    # Plot if required:
    if plot:
        fig, ax = plt.subplots(2, figsize=[12, 8])
        # Plot the grand average:
        ax[0].plot(grand_avg.times, np.squeeze(grand_avg.copy().pick(channel).data.T),
                   label="Grand average", color="k", linewidth=3)
        ax[0].plot(grand_avg.times, gaussian(grand_avg.times, *popt), label="fit", color="r", linewidth=1)
        ax[0].set_title("Observed and fitted ERP")
        ax[0].legend()
        # Plot separately each gaussian:
        for component in posteriors.keys():
            ax[1].plot(grand_avg.times, gaussian(grand_avg.times, posteriors[component]["amplitude"],
                                                 posteriors[component]["latency"],
                                                 posteriors[component]["sigma"],
                                                 posteriors[component]["offset"]),
                       label=component)
        ax[1].legend()
        ax[1].set_title("Fitted components")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig("Fitted components.png")

    return posteriors


def generate_erp(times, comp_dict, peak_noise=0.1, latency_noise=0, sigma_noise=0):
    """

    :param times:
    :param comp_dict:
    :param peak_noise:
    :param latency_noise:
    :param sigma_noise:
    :return:
    """
    comps_list = []
    for comp in comp_dict:
        # Get the amplitude:
        comps_list.append(gaussian(times,
                                   np.random.normal(comp_dict[comp]["amplitude"], scale=peak_noise, size=1)[0],
                                   np.random.normal(comp_dict[comp]["latency"], scale=latency_noise, size=1)[0],
                                   np.random.normal(comp_dict[comp]["sigma"], scale=sigma_noise, size=1)[0],
                                   np.random.normal(comp_dict[comp]["offset"], scale=sigma_noise, size=1)[0]))
    return np.sum(np.array(comps_list), axis=0)


def convert_noise(times, comp_dict, noise):
    """
    This function convert noise expressed as a proportion of variance in the noise free data into mV. In other words,
    the noise can be expressed as 10% of the standard deviation of the noise free data. This function converts this 10%
    in actual mV to add to the data. This enables more control over the noise, as a specific noise level in mV might be
    10 times the dynamic range of the data
    :param times: (numpy array) time array over which to generate the ERP
    :param comp_dict: (dictionary) contains the parameters for each component
    :param noise: (float) noise as a proportion of std in the noise free signal
    :return:
    """
    erp = generate_erp(times, comp_dict, peak_noise=0, latency_noise=0, sigma_noise=0)
    noise_mv = np.std(erp) * noise
    return noise_mv


def adjust_comp(components_dict, fsize, conditions, noise=0.1):
    """

    :param components_dict:
    :param fsize:
    :param noise:
    :param conditions:
    :return:
    """
    assert len(conditions) == 2, "You have passed more than two conditions, not supported yet!"

    # Prepare final dictionary:
    cond_comp_dict = {cond: {comp: {} for comp in components_dict.keys()} for cond in conditions}

    # Loop through each components:
    for component in components_dict.keys():
        # Compute the amplitude difference to obtain such an effect size under this noise:
        amp_diff = (fsize * noise) / 2
        # Adding that difference to the amplitude of the first condition:
        cond_comp_dict[conditions[0]][component] = {
            "amplitude": components_dict[component]["amplitude"] + amp_diff,
            "latency": components_dict[component]["latency"],
            "sigma": components_dict[component]["sigma"],
            "offset": components_dict[component]["offset"]
        }
        # Subtracting it from the second:
        cond_comp_dict[conditions[1]][component] = {
            "amplitude": components_dict[component]["amplitude"] - amp_diff,
            "latency": components_dict[component]["latency"],
            "sigma": components_dict[component]["sigma"],
            "offset": components_dict[component]["offset"]
        }
    return cond_comp_dict


def gaussian(x, a, mu, sig, offset):
    return (a * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))) + offset


def single_trials_fsizes(epochs, tmin, tmax):
    """
    This function compute effect sizes at the single trial level
    :param epochs: (mne epochs object)
    :param tmin: (float) tmin for which to compute the effect size
    :param tmax: (float) tmax for which to compute the effect size
    :return:
    """
    # Get the conditions:
    conds = list(set(epochs.events[:, 2]))
    if len(conds) > 2:
        raise Exception("There are more than two conditions. Not supported yet!")
    latencies, peaks, avgs = {str(cond): [] for cond in conds}, \
                             {str(cond): [] for cond in conds}, \
                             {str(cond): [] for cond in conds}
    # Looping through each trial:
    for ix, trial in enumerate(epochs.iter_evoked()):
        ch, latency, peak = trial.get_peak(ch_type='eeg', return_amplitude=True,
                                           tmin=tmin, tmax=tmax)
        avg = np.mean(trial.copy().crop(tmin, tmax).data)
        latencies[trial.comment].append(latency)
        peaks[trial.comment].append(peak)
        avgs[trial.comment].append(avg)
    # Convert to list of lists:
    latencies = [np.array(values) for values in latencies.values()]
    peaks = [np.array(values) for values in peaks.values()]
    avgs = [np.array(values) for values in avgs.values()]
    # Compute the latency standard deviation:
    latency_std = np.std(np.concatenate(latencies))
    # Compute the peaks and average effect sizes:
    peaks_fsize = compute_fsize(peaks[0], peaks[1])
    avgs_fsize = compute_fsize(avgs[0], avgs[1])
    # Compute the t statistics:
    peaks_tstat = compute_t_stat(peaks[0], peaks[1], axis=0)
    avgs_tstat = compute_t_stat(avgs[0], avgs[1], axis=0)

    return peaks_fsize, avgs_fsize, peaks_tstat, avgs_tstat, latency_std
