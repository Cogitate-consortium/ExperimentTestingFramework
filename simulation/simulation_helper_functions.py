import numpy as np


def gaussian(x, a, mu, sig):
    return a * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def gaussian_5_comp(x, a1, a2, a3, a4, a5, mu1, mu2, mu3, mu4, mu5, sig1, sig2, sig3, sig4, sig5):
    y = a1 * np.exp(-((x - mu1) / sig1) ** 2) + a2 * np.exp(-((x - mu2) / sig2) ** 2) + a3 * np.exp(
        -((x - mu3) / sig3) ** 2) \
        + a4 * np.exp(-((x - mu4) / sig4) ** 2) + a5 * np.exp(-((x - mu5) / sig5) ** 2)
    return y


def generate_erp(times, comp_dict):
    """

    :param times:
    :param comp_dict:
    :return:
    """
    comps_list = []
    for comp in comp_dict:
        comps_list.append(gaussian(times, comp_dict[comp]["amp"], comp_dict[comp]["latency"],
                                   comp_dict[comp]["variance"]))
    return np.sum(np.array(comps_list), axis=0)


def adjust_comp(components_dict, effect_size_dict, sigma, conditions):
    """

    :param components_dict:
    :param effect_size_dict:
    :param sigma:
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
        amp_diff = (effect_size * sigma) / 2
        # Adding that difference to the amplitude of the first condition:
        cond_comp_dict[conditions[0]][component] = {
            "latency": components_dict[component]["latency"],
            "amp": components_dict[component]["amp"] + amp_diff,
            "variance": components_dict[component]["variance"]
        }
        # Subtracting it from the second:
        cond_comp_dict[conditions[1]][component] = {
            "latency": components_dict[component]["latency"],
            "amp": components_dict[component]["amp"] - amp_diff,
            "variance": components_dict[component]["variance"]
        }
    return cond_comp_dict

