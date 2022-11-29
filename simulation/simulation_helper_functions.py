import numpy as np


def gaussian(x, a, mu, sig):
    return a * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def gaussian_5_comp(x, a1, a2, a3, a4, a5, mu1, mu2, mu3, mu4, mu5, sig1, sig2, sig3, sig4, sig5):
    y = a1 * np.exp(-((x - mu1) / sig1) ** 2) + a2 * np.exp(-((x - mu2) / sig2) ** 2) + a3 * np.exp(
        -((x - mu3) / sig3) ** 2) \
        + a4 * np.exp(-((x - mu4) / sig4) ** 2) + a5 * np.exp(-((x - mu5) / sig5) ** 2)
    return y


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
                                   np.random.normal(comp_dict[comp]["sigma"], scale=sigma_noise, size=1)[0]))
    return np.sum(np.array(comps_list), axis=0)


def adjust_comp(components_dict, effect_size_dict, conditions, noise=0.1):
    """

    :param components_dict:
    :param effect_size_dict:
    :param noise:
    :param conditions:
    :return:
    """
    assert len(conditions) == 2, "You have passed more than two conditions, not supported yet!"

    # Prepare final dictionary:
    cond_comp_dict = {cond: {comp: {} for comp in components_dict.keys()} for cond in conditions}

    # Loop through each components:
    for component in components_dict.keys():
        effect_size = effect_size_dict[component]
        # Compute the amplitude difference to obtain such an effect size under this noise:
        amp_diff = (effect_size * noise) / 2
        # Adding that difference to the amplitude of the first condition:
        cond_comp_dict[conditions[0]][component] = {
            "amplitude": components_dict[component]["amplitude"] + amp_diff,
            "amplitude_std": components_dict[component]["amplitude_std"],
            "latency": components_dict[component]["latency"],
            "latency_std": components_dict[component]["latency_std"],
            "sigma": components_dict[component]["sigma"],
            "sigma_std": components_dict[component]["sigma_std"]
        }
        # Subtracting it from the second:
        cond_comp_dict[conditions[1]][component] = {
            "amplitude": components_dict[component]["amplitude"] - amp_diff,
            "amplitude_std": components_dict[component]["amplitude_std"],
            "latency": components_dict[component]["latency"],
            "latency_std": components_dict[component]["latency_std"],
            "sigma": components_dict[component]["sigma"],
            "sigma_std": components_dict[component]["sigma_std"]
        }
    return cond_comp_dict


def gen_intersubject_noise(cond_comp_dict, subjects, peak_noise=True, latency_noise=False, sigma_noise=False):
    """

    :param cond_comp_dict:
    :param subjects:
    :param peak_noise:
    :param latency_noise:
    :param sigma_noise:
    :return:
    """
    subjects_param_dict = {subject: {cond: {comp: {param: None for param in cond_comp_dict[cond][comp]}
                                            for comp in cond_comp_dict[cond]}
                                     for cond in cond_comp_dict.keys()} for subject in subjects}

    for subject in subjects_param_dict.keys():
        for cond in cond_comp_dict.keys():
            for comp in cond_comp_dict[cond].keys():
                if peak_noise is True:
                    subjects_param_dict[subject][cond][comp]["amplitude"] = \
                        np.random.normal(cond_comp_dict[cond][comp]["amplitude"],
                                         cond_comp_dict[cond][comp]["amplitude_std"], size=1)[0]
                else:
                    subjects_param_dict[subject][cond][comp]["amplitude"] = \
                        np.random.normal(cond_comp_dict[cond][comp]["amplitude"],
                                         0, size=1)[0]
                if latency_noise is True:
                    subjects_param_dict[subject][cond][comp]["latency"] = \
                        np.random.normal(cond_comp_dict[cond][comp]["latency"],
                                         cond_comp_dict[cond][comp]["latency_std"], size=1)[0]
                else:
                    subjects_param_dict[subject][cond][comp]["latency"] = \
                        np.random.normal(cond_comp_dict[cond][comp]["latency"],
                                         0, size=1)[0]
                if sigma_noise is True:
                    subjects_param_dict[subject][cond][comp]["sigma"] = \
                        np.random.normal(cond_comp_dict[cond][comp]["sigma"],
                                         cond_comp_dict[cond][comp]["sigma_std"], size=1)[0]
                else:
                    subjects_param_dict[subject][cond][comp]["sigma"] = \
                        np.random.normal(cond_comp_dict[cond][comp]["sigma"],
                                         0, size=1)[0]
    return subjects_param_dict
