import argparse
import json
import os
import scipy
import mne

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from general_utilities.path_helper_function import find_files
from simulation.simulation_helper_functions import adjust_comp, generate_erp


def simulate_epochs():

    # ============================================================================================
    # Parsing input and getting the configs:
    parser = argparse.ArgumentParser(description="Arguments for simulate_epochs")
    parser.add_argument('--config', type=str, default=None,
                        help="Config file for analysis parameters (file name + path)")
    args = parser.parse_args()
    if args.config is None:
        if os.name == "nt":
            configs = find_files(Path(os.getcwd(), "simulation_configs"), naming_pattern="*", extension=".json")
        else:
            configs = find_files(Path(os.getcwd(), "configs_hpc"), naming_pattern="*", extension=".json")
    else:
        configs = [args.config]

    # Loop through each config:
    for config in configs:
        # Load the config
        with open(config) as f:
            param = json.load(f)

        # Load the components:
        with open(param["components_file"]) as f:
            components_dict = json.load(f)

        # ============================================================================================
        # Simulate data:
        # Adjusting the mean of each components amplitude to account for the effect size:
        cond_comp_dict = adjust_comp(components_dict, param["effect_size"], param["noise"]["sigma"],
                                     param["conditions"])
        # Generate time axis:
        times = np.linspace(param["t0"], param["tmax"], int(param["sfreq"] * (param["tmax"] - param["t0"])))
        # Preallocate array for data:
        data = np.zeros([param["n_trials_per_cond"] * len(param["conditions"]), param["n_channels"], times.shape[0]])

        # Loop through each subject:
        for sub in range(param["n_subjects"]):
            print("Simulate sub-{} data".format(sub + 1))
            # Loop through each condition:
            ctr = 0
            fig, ax = plt.subplots()
            for cond in cond_comp_dict.keys():
                # Generate the ERP to that condition:
                erp = generate_erp(times, cond_comp_dict[cond])
                ax.plot(erp)
                # Now  replicate this component for as many trials as we have:
                single_trials = np.tile(erp, (param["n_trials_per_cond"], 1))
                # Generate the noise array:
                noise = np.random.normal(param["noise"]["mean"], scale=param["noise"]["sigma"],
                                         size=single_trials.shape)
                filt_kern = scipy.signal.boxcar(int(param["noise"]["autocorrelation_ms"] * 1000 / param["sfreq"]))
                noise = np.array([scipy.signal.convolve(noise[row, :], filt_kern, mode="same")
                                  for row in range(noise.shape[0])])
                single_trials = np.add(single_trials, noise)
                data[ctr:ctr + param["n_trials_per_cond"], 0, :] = single_trials
                ctr += param["n_trials_per_cond"]

            # ============================================================================================
            # Convert to an MNE object:
            # Create the info for the mne object:
            ch_names = [f'EEG{n:03}' for n in range(param["n_channels"])]
            ch_types = ["eeg"] * param["n_channels"]
            info = mne.create_info(ch_names, ch_types=ch_types, sfreq=param["sfreq"])
            # Generate the events:
            conditions = [[cond] * param["n_trials_per_cond"] for cond in cond_comp_dict.keys()]
            evts_ids = [[ind] * param["n_trials_per_cond"] for ind, cond in enumerate(cond_comp_dict.keys())]
            onsets = (np.linspace(0, (param["tmax"] - param["t0"]) * param["n_trials_per_cond"] * len(param["conditions"]),
                                  param["n_trials_per_cond"] * len(param["conditions"]), endpoint=False)
                      + np.abs(param["t0"]))
            # Add all of that into the metadata:
            metadata = pd.DataFrame({
                param["metadata_col"][0]: np.array([item for sublist in conditions for item in sublist]),
                param["metadata_col"][1]: np.array([item for sublist in evts_ids for item in sublist]),
                param["metadata_col"][2]: onsets,
            })
            events = np.column_stack(((metadata[param["metadata_col"][2]].to_numpy() * param["sfreq"]).astype(int),
                                      np.zeros(len(metadata), dtype=int),
                                      metadata[param["metadata_col"][1]].to_numpy().astype(int)))
            events_dict = {cond: ind for ind, cond in enumerate(cond_comp_dict.keys())}
            # Convert the data to an epochs:
            epochs = mne.EpochsArray(data, info, events=events, event_id=events_dict, tmin=param["t0"])
            epochs.metadata = metadata

            # ============================================================================================
            # Vizualize the simulated data:
            # Get the evoked data to each cond to plot them:
            evk = {
                "cond_1": epochs.copy()["cond_1"].average().data,
                "cond_2": epochs.copy()["cond_2"].average().data
            }
            # Compute the P1 effect size:
            p1_epochs = epochs.copy().crop(tmin=-0.2, tmax=0.0)
            p1_amp = {
                "cond_1": np.mean(p1_epochs.copy()["cond_1"].average().data),
                "cond_2": np.mean(p1_epochs.copy()["cond_2"].average().data)
            }
            p1_std = np.std([p1_epochs.copy()["cond_1"].average().data, p1_epochs.copy()["cond_2"].average().data])
            fig, ax = plt.subplots()
            ax.plot(times, np.squeeze(evk["cond_1"]), label="cond_1", c="blue")
            # ax.plot(times, np.squeeze(epochs.copy()["cond_1"]).T, "--", c="k", linewidth=1, alpha=0.5)
            ax.plot(times, np.squeeze(evk["cond_2"]), label="cond_2", c="red")
            # ax.plot(times, np.squeeze(epochs.copy()["cond_2"]).T, ":", c="red", linewidth=0.2, alpha=0.5)
            ax.legend()
            ax.set_ylabel("Time (s)")
            ax.set_ylabel("Amplitude mV")
            f_size = (p1_amp["cond_1"] - p1_amp["cond_2"])/p1_std
            ax.text(0.6, 0.3, "Expected  std= {:.2f}".format(param["noise"]["sigma"]))
            ax.text(0.6, 0.2, "Expected Effect size={:.2f}".format(param["effect_size"]["P1"]))
            ax.text(0.6, 0.1, "Observed P1  std= {:.2f}".format(p1_std))
            ax.text(0.6, 0, "Observed Effect size={:.2f}".format(f_size))
            # plt.show()

            # ==========================================================================================================
            do = True
            if do:
                # Save the data:
                print("=" * 40)
                print("Saving the results")
                save_root = Path(param["bids_root"], "derivatives", "preprocessing",
                                 "sub-" + str(sub + 1), "ses-" + param["ses"],
                                 "ieeg", "epoching", param["signal"], param["preprocess_steps"])
                if not os.path.isdir(save_root):
                    os.makedirs(save_root)
                fname = "sub-{}_ses-{}_task-{}_desc-epoching_ieeg-epo.fif".format(sub + 1, param["ses"],
                                                                                  param["task"])
                epochs.save(Path(save_root, fname), overwrite=True)


if __name__ == "__main__":
    simulate_epochs()