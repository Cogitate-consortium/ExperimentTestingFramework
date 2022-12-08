import argparse
import json
from joblib import Parallel, delayed
from tqdm import tqdm
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from general_utilities.path_helper_function import find_files, path_generator, list_subjects
from general_utilities.data_helper_function import mean_confidence_interval
from erp_analysis.erp_analysis_helper_functions import subject_erp_wrapper, plot_3d, plot_heatmap
import numpy as np

test_subs = ["1"]
n_repeats = 100
show_plots = False
debug = False


def erp_analysis():
    # ============================================================================================
    # Parsing input and getting the configs:
    parser = argparse.ArgumentParser(description="Arguments for simulate_epochs")
    parser.add_argument('--config', type=str, default=None,
                        help="Config file for analysis parameters (file name + path)")
    args = parser.parse_args()
    if args.config is None:
        if os.name == "nt":
            configs = find_files(Path(os.getcwd(), "erp_analysis_configs_local"), naming_pattern="*", extension=".json")
        else:
            configs = find_files(Path(os.getcwd(), "configs_hpc"), naming_pattern="*", extension=".json")
    else:
        configs = [args.config]

    # ==================================================================================================================
    # Looping through each config:
    for config in configs:
        # Load the config
        with open(config) as f:
            param = json.load(f)
        # Generate the dir to save the results:
        save_root = Path(param["bids_root"], "derivatives", param["analysis"], "population")
        results_save_root = path_generator(save_root, analysis=param["name"],
                                           preprocessing_steps=param["preprocess_steps"],
                                           fig=False, results=True, data=False)
        fig_save_root = path_generator(save_root, analysis=param["name"],
                                       preprocessing_steps=param["preprocess_steps"],
                                       fig=False, results=True, data=False)
        evk_save_root = Path(fig_save_root, "evoked")
        if not os.path.isdir(evk_save_root):
            os.makedirs(evk_save_root)

        # List the subjects:
        subjects_list = list_subjects(Path(param["bids_root"], "derivatives", "preprocessing"), prefix="sub-")
        if debug:
            subjects_list = test_subs
        path_info = {
            "bids_root": param["bids_root"],
            "data_type": param["data_type"],
            "ses": param["ses"],
            "preprocess_steps": param["preprocess_steps"],
            "preprocess_folder": param["preprocess_folder"],
            "signal": param["signal"],
            "tasks": param["tasks"]
        }
        results = Parallel(n_jobs=param["njobs"])(delayed(subject_erp_wrapper)(
            subjects_list[0], path_info,
            param["jitter_durations"],
            param["jittered_trials_proportion"],
            param["tail"],
            param["components"],
            param["conditions"],
            param["shuffled_trials_proportion"]
        ) for i in tqdm(range(param["n_repeats"])))
        results = pd.concat(results, ignore_index=True)
        # Save the results to file:
        results_to_save = results.loc[:, results.columns != "evoked"]
        results_to_save = results_to_save.loc[:, results_to_save.columns != "evoked_diff"]
        results_to_save.to_csv(Path(results_save_root, "components_results.csv"))

        # ============================================================================================
        # Plot effect of jitter:
        # Compute the effect sizes of the difference between the two conditions:
        results["cond_diff"] = results["cond_1-mean"] - results["cond_2-mean"]

        # Get the data for the jitter:
        jitter_results = results.loc[results["shuffle_proportion"] == 0]
        # Remove duplicates:
        jitter_results = jitter_results.drop_duplicates(subset=["effect_size", "subject", "jitter_duration",
                                                                "jitter_prop", "shuffle_proportion"])
        stats_df = pd.DataFrame()
        # Looping through each component again to plot the effect of jitter:
        for component in results["component"].unique():
            for effect_size in results["effect_size"].unique():
                for jitter_duration in results["jitter_duration"].unique():
                    for jitter_prop in results["jitter_prop"].unique():
                        res_df = jitter_results.loc[(results["component"] == component)
                                                    & (results["effect_size"] == effect_size)
                                                    & (results["jitter_prop"] == jitter_prop)
                                                    & (results["jitter_duration"] == jitter_duration)]
                        # Extract all the data:
                        avg_fsize = res_df["avgs_fsize"].to_numpy()
                        peaks_fsize = res_df["peaks_fsize"].to_numpy()
                        avg_tstat = res_df["avgs_tstat"].to_numpy()
                        peaks_tstat = res_df["peaks_tstat"].to_numpy()
                        latencies_std = res_df["latency_std"].to_numpy()
                        # Compute the effect size:
                        stats_df = stats_df.append(pd.DataFrame({
                            "sim_effect_size": effect_size,
                            "component": component,
                            "jitter_prop": jitter_prop,
                            "jitter_duration": jitter_duration,
                            "avg_fsize": np.mean(avg_fsize),
                            "peak_fsize": np.mean(peaks_fsize),
                            "avg_tstat": np.mean(avg_tstat),
                            "peak_tstat": np.mean(peaks_tstat),
                            "latencies_std": np.mean(latencies_std)
                        }, index=[0]))
                stats_df = stats_df.reset_index(drop=True)

            # ============================================================================================
            # Plot effect sizes with jitter:
            # First plot effect on peak amplitude:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            for sim_effect_size in stats_df["sim_effect_size"].unique():
                plot_df = stats_df.loc[(stats_df["component"] == component) &
                                       (stats_df["sim_effect_size"] == sim_effect_size)]
                ax = plot_3d(ax, plot_df["jitter_prop"].to_numpy(), plot_df["jitter_duration"].to_numpy(),
                             plot_df["peak_fsize"].to_numpy(),
                             "Simulated effect size = {}".format(sim_effect_size),
                             xlabel='Jitter proportion', y_label='Jitter duration', zlabel='Observed effect size',
                             alpha=.2)
            plt.title("Observed jitter effect on peak effect size")
            plt.legend()
            plt.tight_layout()
            plt.savefig(Path(fig_save_root, "{}_peak_effect_sizes_3d.png".format(component)))
            if show_plots:
                plt.show()
            else:
                plt.close()

            # Same for the mean amplitude:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            for sim_effect_size in stats_df["sim_effect_size"].unique():
                plot_df = stats_df.loc[(stats_df["component"] == component) &
                                       (stats_df["sim_effect_size"] == sim_effect_size)]
                ax = plot_3d(ax, plot_df["jitter_prop"].to_numpy(), plot_df["jitter_duration"].to_numpy(),
                             plot_df["avg_fsize"].to_numpy(),
                             "Simulated effect size = {}".format(sim_effect_size),
                             xlabel='Jitter proportion', y_label='Jitter duration', zlabel='Observed effect size',
                             alpha=.2)
            plt.title("Observed jitter effect on average effect size")
            plt.legend()
            plt.tight_layout()
            plt.savefig(Path(fig_save_root, "{}_average_effect_sizes_3d.png".format(component)))
            if show_plots:
                plt.show()
            else:
                plt.close()

            # Finally for the latency:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            for sim_effect_size in stats_df["sim_effect_size"].unique():
                plot_df = stats_df.loc[(stats_df["component"] == component) &
                                       (stats_df["sim_effect_size"] == sim_effect_size)]
                ax = plot_3d(ax, plot_df["jitter_prop"].to_numpy(), plot_df["jitter_duration"].to_numpy(),
                             plot_df["latencies_std"].to_numpy(),
                             "Simulated effect size = {}".format(sim_effect_size),
                             xlabel='Jitter proportion', y_label='Jitter duration', zlabel='Observed effect size',
                             alpha=.2)
            plt.title("Observed jitter effect on {} peak latency std".format(component))
            plt.legend()
            plt.tight_layout()
            plt.savefig(Path(fig_save_root, "{}_peak_latency_std_3d.png".format(component)))
            if show_plots:
                plt.show()
            else:
                plt.close()

            # Plot effect of jitter in 2D:
            for effect_size in results["effect_size"].unique():
                # Extract the data from this effect size:
                data_df = stats_df.loc[stats_df["sim_effect_size"] == effect_size]
                # Remove the rows with 0 jitter:
                data_df = data_df.loc[(data_df["jitter_prop"] != 0) & (data_df["jitter_duration"] != 0)]
                plot_heatmap(data_df.loc[stats_df["component"] == component], "jitter_prop", "jitter_duration",
                             "avg_fsize", xlabel="Jitter duration (ms)",
                             ylabel="Jittered trials proportion", zlabel="$\u03F4_{obs}$",
                             title="Average amplitude \u03F4 as a function of jitter"
                                   "\n comp={}, \u03F4={}".format(component, effect_size), midpoint=0.2)

                plt.savefig(Path(fig_save_root, "{}_jitter_average_fsize_{}.png".format(component, effect_size)))
                if show_plots:
                    plt.show()
                else:
                    plt.close()
                # Same for the peak amplitude:
                plot_heatmap(data_df.loc[stats_df["component"] == component], "jitter_prop", "jitter_duration",
                             "peak_fsize", xlabel="Jitter duration (ms)",
                             ylabel="Jittered trials proportion", zlabel="$\u03F4_{obs}$",
                             title="Average amplitude \u03F4 as a function of jitter"
                                   "\n comp={}, \u03F4={}".format(component, effect_size), midpoint=0.2)
                plt.savefig(Path(fig_save_root, "{}_jitter_peak_fsize_{}.png".format(component, effect_size)))
                if show_plots:
                    plt.show()
                else:
                    plt.close()

            # ============================================================================================
            # Plot t statistic with jitter:
            # First plot effect on peak amplitude:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            for sim_effect_size in stats_df["sim_effect_size"].unique():
                plot_df = stats_df.loc[(stats_df["component"] == component) &
                                       (stats_df["sim_effect_size"] == sim_effect_size)]
                ax = plot_3d(ax, plot_df["jitter_prop"].to_numpy(), plot_df["jitter_duration"].to_numpy(),
                             plot_df["peak_tstat"].to_numpy(),
                             "Simulated effect size = {}".format(sim_effect_size),
                             xlabel='Jitter proportion', y_label='Jitter duration', zlabel='T-statistic',
                             alpha=.2)
            plt.title("Observed jitter effect on peak effect size")
            plt.legend()
            plt.tight_layout()
            plt.savefig(Path(fig_save_root, "{}_peak_tstat_3d.png".format(component)))
            if show_plots:
                plt.show()
            else:
                plt.close()

            # Same for the mean amplitude:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            for sim_effect_size in stats_df["sim_effect_size"].unique():
                plot_df = stats_df.loc[(stats_df["component"] == component) &
                                       (stats_df["sim_effect_size"] == sim_effect_size)]
                ax = plot_3d(ax, plot_df["jitter_prop"].to_numpy(), plot_df["jitter_duration"].to_numpy(),
                             plot_df["avg_tstat"].to_numpy(),
                             "Simulated effect size = {}".format(sim_effect_size),
                             xlabel='Jitter proportion', y_label='Jitter duration', zlabel='T-statistic',
                             alpha=.2)
            plt.title("Observed jitter effect on average effect size")
            plt.legend()
            plt.tight_layout()
            plt.savefig(Path(fig_save_root, "{}_average_tstat_3d.png".format(component)))
            if show_plots:
                plt.show()
            else:
                plt.close()

            # Plot effect of jitter in 2D:
            for effect_size in results["effect_size"].unique():
                # Extract the data from this effect size:
                data_df = stats_df.loc[stats_df["sim_effect_size"] == effect_size]
                # Remove the rows with 0 jitter:
                data_df = data_df.loc[(data_df["jitter_prop"] != 0) & (data_df["jitter_duration"] != 0)]
                plot_heatmap(data_df.loc[stats_df["component"] == component], "jitter_prop", "jitter_duration",
                             "avg_tstat", xlabel="Jitter duration (ms)",
                             ylabel="Jittered trials proportion", zlabel="T statistic",
                             title="Average t-statistic as a function of jitter"
                                   "\n comp={}, \u03F4={}".format(component, effect_size), midpoint=1.96)
                plt.savefig(Path(fig_save_root, "{}_jitter_average_fsize_{}_tstat.png".format(component, effect_size)))
                if show_plots:
                    plt.show()
                else:
                    plt.close()
                # Same for the peak amplitude:
                plot_heatmap(data_df.loc[stats_df["component"] == component], "jitter_prop", "jitter_duration",
                             "peak_tstat", xlabel="Jitter duration (ms)",
                             ylabel="Jittered trials proportion", zlabel="T statistic",
                             title="Peak t-statistic as a function of jitter"
                                   "\n comp={}, \u03F4={}".format(component, effect_size), midpoint=1.96)
                plt.savefig(Path(fig_save_root, "{}_jitter_peak_fsize_{}_tstat.png".format(component, effect_size)))
                if show_plots:
                    plt.show()
                else:
                    plt.close()

        # ============================================================================================
        # Plot effect of shuffle:
        # Get the data for the jitter:
        shuffle_results = results.loc[(results["jitter_duration"] == 0) & (results["jitter_prop"] == 0)]
        # Remove duplicates:
        shuffle_results = shuffle_results.drop_duplicates(subset=["effect_size", "subject", "jitter_duration",
                                                                  "jitter_prop", "shuffle_proportion"])
        for component in results["component"].unique():
            stats_df = pd.DataFrame()
            for effect_size in results["effect_size"].unique():
                for shuffle_prop in results["shuffle_proportion"].unique():
                    res_df = shuffle_results.loc[(results["component"] == component)
                                                 & (results["effect_size"] == effect_size)
                                                 & (results["shuffle_proportion"] == shuffle_prop)]
                    # Extract all the data:
                    avg_fsize = res_df["avgs_fsize"].to_numpy()
                    peaks_fsize = res_df["peaks_fsize"].to_numpy()
                    avg_tstat = res_df["avgs_tstat"].to_numpy()
                    peaks_tstat = res_df["peaks_tstat"].to_numpy()
                    stats_df = stats_df.append(pd.DataFrame({
                        "sim_effect_size": effect_size,
                        "component": component,
                        "shuffle_proportion": shuffle_prop,
                        "avg_fsize": np.mean(avg_fsize),
                        "peak_fsize": np.mean(peaks_fsize),
                        "avg_tstat": np.mean(avg_tstat),
                        "peak_tstat": np.mean(peaks_tstat)
                    }, index=[0]))
                stats_df = stats_df.reset_index(drop=True)
            # Plot the heatmap:
            plot_heatmap(stats_df.loc[stats_df["component"] == component], "shuffle_proportion",
                         "sim_effect_size", "avg_fsize", xlabel="Proportion of shuffle trials",
                         ylabel="Simulated Effect sizes", zlabel="$\u03F4_{obs}$",
                         title="Average effect size as a function of label shuffles\n comp={}".format(component),
                         midpoint=0.2)
            plt.savefig(Path(fig_save_root, "{}_shuffle_average.png".format(component)))
            if show_plots:
                plt.show()
            else:
                plt.close()
            # Convert long to wide table to generate a heatmap:
            plot_heatmap(stats_df.loc[stats_df["component"] == component], "shuffle_proportion",
                         "sim_effect_size", "peak_fsize", xlabel="Proportion of shuffle trials",
                         ylabel="Simulated Effect sizes", zlabel="$\u03F4_{obs}$",
                         title="Peak effect size as a function of label shuffles\n comp={}".format(component),
                         midpoint=0.2)
            plt.savefig(Path(fig_save_root, "{}_shuffle_peak.png".format(component)))
            if show_plots:
                plt.show()
            else:
                plt.close()

            # Plot the t-statistic
            plot_heatmap(stats_df.loc[stats_df["component"] == component], "shuffle_proportion",
                         "sim_effect_size", "avg_tstat", xlabel="Proportion of shuffle trials",
                         ylabel="Simulated Effect sizes", zlabel="T-statistic",
                         title="Average T-statistic as a function of label shuffles\n comp={}".format(component),
                         midpoint=0.2)
            plt.savefig(Path(fig_save_root, "{}_shuffle_average_stat.png".format(component)))
            if show_plots:
                plt.show()
            else:
                plt.close()
            # Peaks:
            plot_heatmap(stats_df.loc[stats_df["component"] == component], "shuffle_proportion",
                         "sim_effect_size", "peak_tstat", xlabel="Proportion of shuffle trials",
                         ylabel="Simulated Effect sizes", zlabel="T-statistic",
                         title="Average T-statistic as a function of label shuffles\n comp={}".format(component),
                         midpoint=0.2)
            plt.savefig(Path(fig_save_root, "{}_shuffle_peak_tstat.png".format(component)))
            if show_plots:
                plt.show()
            else:
                plt.close()

        # ================================================================================================
        # Plot time series:
        for component in results["component"].unique():
            # Get all the data for this component:
            comp_results = results.loc[results["component"] == component]
            # Extract the data for plotting jitter without shuffle and the other way around:
            jitter_data = jitter_results.loc[jitter_results["component"] == component]
            shuffle_data = shuffle_results.loc[shuffle_results["component"] == component]

            for sim_effect_size in jitter_data["effect_size"].unique():
                # Extract these data:
                jitter_f_size_df = jitter_data.loc[jitter_data["effect_size"] == sim_effect_size]
                shuffle_f_size_df = shuffle_data.loc[shuffle_data["effect_size"] == sim_effect_size]

                # ============================================================================================
                # Printing the evoked response differences:

                # =========================================
                # Plot effect of jitter on evoked difference:
                for jitter_dur in jitter_f_size_df["jitter_duration"].unique():
                    jitter_dur_df = jitter_f_size_df.loc[
                        jitter_f_size_df["jitter_duration"] == jitter_dur]
                    fig, ax = plt.subplots()
                    for jitter_prop in jitter_dur_df["jitter_prop"].unique():
                        evks = jitter_dur_df.loc[
                            jitter_dur_df["jitter_prop"] == jitter_prop, "evoked_diff"].to_list()
                        times = [evk.times for evk in evks]
                        evks_list = [np.squeeze(evk.get_data()) for evk in evks]
                        # Depending on the jittere distribution, the subjects arrays might have different counts of data
                        # points. Adjusting for that:
                        times_intersect = np.sort(np.array(list(set.intersection(*map(set, times)))))
                        for ind_evk, evk in enumerate(evks_list):
                            # Find the overlap:
                            sub_inter, ind_1, ind_2 = np.intersect1d(times[ind_evk], times_intersect,
                                                                     return_indices=True)
                            evks_list[ind_evk] = evks_list[ind_evk][ind_2]
                        # Compute the mean and CIs:
                        avg, low_ci, up_ci = mean_confidence_interval(np.array(evks_list), confidence=0.95, axis=0)
                        ax.plot(times_intersect, avg, label="jitter={}".format(jitter_prop))
                        ax.fill_between(times_intersect, up_ci, low_ci, alpha=.2)
                    ax.set_xlim([times_intersect[0], times_intersect[-1]])
                    ax.set_ylabel("T statistic")
                    ax.set_xlabel("Time (sec)")
                    plt.legend()
                    plt.savefig(Path(fig_save_root, "{}_evoked_fsize_{}_jitter_dur_{}.png".format(component,
                                                                                                  sim_effect_size,
                                                                                                  jitter_dur)))
                    if show_plots:
                        plt.show()
                    else:
                        plt.close()

                # =========================================
                # Plot effect of shuffle on evoked difference:
                fig, ax = plt.subplots()
                for shuffle_prop in shuffle_f_size_df["shuffle_proportion"].unique():
                    evks = shuffle_f_size_df.loc[shuffle_f_size_df["shuffle_proportion"] == shuffle_prop,
                                                 "evoked_diff"].to_list()
                    times = evks[0].times
                    evks = np.array([np.squeeze(evk.get_data()) for evk in evks])
                    # Compute the mean and CIs:
                    avg, low_ci, up_ci = mean_confidence_interval(evks, confidence=0.95, axis=0)
                    ax.plot(times, avg, label="shuffle={}".format(shuffle_prop))
                    ax.fill_between(times, up_ci, low_ci, alpha=.2)
                ax.set_ylabel("T statistic")
                ax.set_xlabel("Time (sec)")
                ax.set_xlim([times_intersect[0], times_intersect[-1]])
                plt.legend()
                plt.savefig(Path(fig_save_root, "{}_evoked_fsize_{}_shuffle.png".format(component, sim_effect_size)))
                if show_plots:
                    plt.show()
                else:
                    plt.close()

                # ============================================================================================
                # Plotting the evoked responses per condition
                for jitter_duration in jitter_f_size_df["jitter_duration"].unique():
                    jitter_dur_df = jitter_f_size_df.loc[jitter_f_size_df["jitter_duration"] == jitter_duration]
                    # Looping through each jitter proportion:
                    for jitter_prop in jitter_dur_df["jitter_prop"].unique():
                        fig, ax = plt.subplots()
                        evks = jitter_dur_df.loc[jitter_dur_df["jitter_prop"] == jitter_prop,
                                                 "evoked"].to_list()
                        # Combine subjects data:
                        sub_evks = [[], []]
                        for evk in evks:
                            sub_evks[0].append(evk[0])
                            sub_evks[1].append(evk[1])
                        times = [evk[0].times for evk in evks]
                        # Depending on the jittere distribution, the subjects arrays might have different counts of data
                        # points. Adjusting for that:
                        times_intersect = np.sort(np.array(list(set.intersection(*map(set, times)))))
                        for ind, evk in enumerate(sub_evks):
                            # Find the overlap:
                            sub_inter, ind_1, ind_2 = np.intersect1d(times[ind], times_intersect, return_indices=True)
                            sub_evks[ind][0].data = sub_evks[ind][0].data[:, ind_2]
                            sub_evks[ind][1].data = sub_evks[ind][1].data[:, ind_2]
                        # Compute the mean and CIs:
                        avg, low_ci, up_ci = mean_confidence_interval(
                            np.array([np.squeeze(evk.data) for evk in sub_evks[0]]),
                            confidence=0.95, axis=0)
                        ax.plot(times_intersect, avg, label="cond_1")
                        ax.fill_between(times_intersect, up_ci, low_ci, alpha=.2)
                        avg, low_ci, up_ci = mean_confidence_interval(
                            np.array([np.squeeze(evk.data) for evk in sub_evks[1]]),
                            confidence=0.95, axis=0)
                        ax.plot(times_intersect, avg, label="cond_2")
                        ax.fill_between(times_intersect, up_ci, low_ci, alpha=.2)
                        ax.set_xlim([times_intersect[0], times_intersect[-1]])
                        ax.set_ylabel("Amplitude (mV)")
                        ax.set_xlabel("Time (sec)")
                        plt.legend()
                        plt.savefig(Path(evk_save_root, "{}_evoked_fsize_{}_jitter_dur_{}_jitter_prop_{}.png".format(
                            component,
                            sim_effect_size,
                            jitter_duration,
                            jitter_prop)))
                        if show_plots:
                            plt.show()
                        else:
                            plt.close()

        print("Finished!")


if __name__ == "__main__":
    erp_analysis()
