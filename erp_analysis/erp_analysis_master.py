import argparse
import json
from joblib import Parallel, delayed
from tqdm import tqdm
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg
from general_utilities.path_helper_function import find_files, path_generator, list_subjects
from general_utilities.data_helper_function import mean_confidence_interval
from erp_analysis.erp_analysis_helper_functions import subject_erp_wrapper, plot_heatmap
import numpy as np
from matplotlib.ticker import StrMethodFormatter
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

test_subs = ["1"]
n_repeats = 100
show_plots = False
debug = False
scatter_cmap = "autumn_r"
fig_size = [20, 15]
SMALL_SIZE = 26
MEDIUM_SIZE = 32
BIGGER_SIZE = 34
gaussian_sig = 4
cmap = 'RdYlBu_r'
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the fi


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
            param["shuffled_trials_proportion"],
            i
        ) for i in tqdm(range(param["n_repeats"])))
        results = pd.concat(results, ignore_index=True)
        # Save the results to file:
        results_to_save = results.loc[:, results.columns != "evoked"]
        results_to_save = results_to_save.loc[:, results_to_save.columns != "evoked_diff"]
        results_to_save.to_csv(Path(results_save_root, "components_results.csv"))
        # Convert the proportion to percent for plotting:
        results["jitter_prop"] = results["jitter_prop"] * 100
        results["shuffle_proportion"] = results["shuffle_proportion"] * 100
        results["jitter_prop"] = results["jitter_prop"].round()
        results["shuffle_proportion"] = results["shuffle_proportion"].round()
        # ============================================================================================
        # Plot effect of jitter:
        # Compute the effect sizes of the difference between the two conditions:
        results["cond_diff"] = results["cond_1-mean"] - results["cond_2-mean"]

        # Get the data for the jitter:
        jitter_results = results.loc[results["shuffle_proportion"] == 0]
        # Remove duplicates:
        jitter_results = jitter_results.drop_duplicates(subset=["effect_size", "subject", "iteration",
                                                                "jitter_duration", "jitter_prop",
                                                                "shuffle_proportion"])
        # Compute the correlation between the different jitter manipulations and the measured t-statistics:
        jitter_duration_stats = []
        for effect_size in jitter_results["effect_size"].unique():
            # First for the jitter duration, computing the correlation separately for each trial proportion:
            for jitter_prop in jitter_results["jitter_prop"].unique():
                corr_data = jitter_results.loc[
                    (jitter_results["effect_size"] == effect_size) & (jitter_results["jitter_prop"] == jitter_prop)]
                stats_peak = pg.corr(corr_data["jitter_duration"].to_numpy(), corr_data["peaks_tstat"].to_numpy(),
                                     alternative="less", method="pearson")
                stats_average = pg.corr(corr_data["jitter_duration"].to_numpy(), corr_data["avgs_tstat"].to_numpy(),
                                        alternative="less", method="pearson")
                # Organize the results into dataframe:
                jitter_duration_stats.append(pd.Series({
                    "effect_size": effect_size,
                    "jitter proportion": jitter_prop,
                    "peak r": stats_peak["r"].item(),
                    "peak p-value": stats_peak["p-val"].item(),
                    "peak CI95": stats_peak["CI95%"].item(),
                    "avg r": stats_average["r"].item(),
                    "avg p-value": stats_average["p-val"].item(),
                    "avg CI95": stats_average["CI95%"].item()
                }))
        jitter_duration_stats = pd.DataFrame(jitter_duration_stats)
        # Save the results to a dataframe:
        jitter_duration_stats.to_csv(Path(results_save_root, "jitter_duration_stats.csv"))

        # Repeating the same but for jitter proportion:
        jitter_proportion_stats = []
        for effect_size in jitter_results["effect_size"].unique():
            # First for the jitter duration, computing the correlation separately for each trial proportion:
            for jitter_dur in jitter_results["jitter_duration"].unique():
                corr_data = jitter_results.loc[
                    (jitter_results["effect_size"] == effect_size) & (jitter_results["jitter_duration"] == jitter_dur)]
                stats_peak = pg.corr(corr_data["jitter_prop"].to_numpy(), corr_data["peaks_tstat"].to_numpy(),
                                     alternative="less", method="pearson")
                stats_average = pg.corr(corr_data["jitter_prop"].to_numpy(), corr_data["avgs_tstat"].to_numpy(),
                                        alternative="less", method="pearson")
                # Organize the results into dataframe:
                jitter_proportion_stats.append(pd.Series({
                    "effect_size": effect_size,
                    "jitter duration": jitter_dur,
                    "peak r": stats_peak["r"].item(),
                    "peak p-value": stats_peak["p-val"].item(),
                    "peak CI95": stats_peak["CI95%"].item(),
                    "avg r": stats_average["r"].item(),
                    "avg p-value": stats_average["p-val"].item(),
                    "avg CI95": stats_average["CI95%"].item()
                }))
        jitter_proportion_stats = pd.DataFrame(jitter_proportion_stats)
        # Save the results to a dataframe:
        jitter_proportion_stats.to_csv(Path(results_save_root, "jitter_proportion_stats.csv"))

        # Remove columns we don't care about:
        jitter_summary = jitter_results.loc[:, jitter_results.columns != "task"]
        jitter_summary = jitter_summary.loc[:, jitter_summary.columns != "subject"]
        jitter_summary = jitter_summary.loc[:, jitter_summary.columns != "iteration"]
        jitter_summary = jitter_summary.loc[:, jitter_summary.columns != "evoked_diff"]
        jitter_summary = jitter_summary.loc[:, jitter_summary.columns != "evoked"]
        # Averaging across iterations within the different manipulated parameters:
        jitter_summary = jitter_summary.groupby(["component", "effect_size",
                                                 "jitter_duration", "jitter_prop",
                                                 "shuffle_proportion"]).mean().reset_index()
        jitter_summary.to_csv(Path(results_save_root, "jitter_summary.csv"))
        # Looping through each component again to plot the effect of jitter:
        for component in jitter_summary["component"].unique():
            # Get the results of this component only:
            comp_jitter_results = jitter_summary.loc[jitter_summary["component"] == component]
            # ============================================================================================
            # Plot effect sizes with jitter:
            # First plot effect on peak amplitude:
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(projection='3d')
            # Add the scatter:
            p = ax.scatter(comp_jitter_results["jitter_prop"].to_numpy(),
                           comp_jitter_results["jitter_duration"].to_numpy(),
                           comp_jitter_results["peaks_fsize"].to_numpy(),
                           c=comp_jitter_results["effect_size"].to_numpy(),
                           cmap=scatter_cmap, s=40)
            cbar = fig.colorbar(p, location='left', shrink=0.7, pad=0.04)
            cbar.ax.set_ylabel('\u03F4', rotation=270, labelpad=40)
            ax.set_xlabel('Jitter proportion (%)', labelpad=30)
            ax.set_ylabel('Jitter duration', labelpad=30)
            ax.set_zlabel('Observed effect size', labelpad=30)
            ax.tick_params(axis='z', which='major', pad=15)
            ax.zaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
            plt.title("Observed jitter effect on peak effect size")
            plt.tight_layout()
            plt.savefig(Path(fig_save_root, "{}_peak_effect_sizes_3d.png".format(component)))
            if show_plots:
                plt.show()
            else:
                plt.close()

            # Same for the mean amplitude:
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(projection='3d')
            # Add the scatter:
            p = ax.scatter(comp_jitter_results["jitter_prop"].to_numpy(),
                           comp_jitter_results["jitter_duration"].to_numpy(),
                           comp_jitter_results["avgs_fsize"].to_numpy(),
                           c=comp_jitter_results["effect_size"].to_numpy(),
                           cmap=scatter_cmap, s=40)
            cbar = fig.colorbar(p, location='left', shrink=0.7, pad=0.04)
            cbar.ax.set_ylabel('\u03F4', rotation=270, labelpad=40)
            ax.set_xlabel('Jitter proportion (%)', labelpad=30)
            ax.set_ylabel('Jitter duration', labelpad=30)
            ax.set_zlabel('Observed effect size', labelpad=40)
            ax.tick_params(axis='z', which='major', pad=15)
            ax.zaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
            plt.title("Observed jitter effect on average effect size")
            plt.tight_layout()
            plt.savefig(Path(fig_save_root, "{}_average_effect_sizes_3d.png".format(component)))
            if show_plots:
                plt.show()
            else:
                plt.close()

            # Finally for the latency:
            # Same for the mean amplitude:
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(projection='3d')
            # Add the scatter:
            p = ax.scatter(comp_jitter_results["jitter_prop"].to_numpy(),
                           comp_jitter_results["jitter_duration"].to_numpy(),
                           comp_jitter_results["latency_std"].to_numpy(),
                           c=comp_jitter_results["effect_size"].to_numpy(),
                           cmap=scatter_cmap, s=40)
            cbar = fig.colorbar(p, location='left', shrink=0.7, pad=0.04)
            cbar.ax.set_ylabel('\u03F4', rotation=270, labelpad=40)
            ax.set_xlabel('Jitter proportion (%)', labelpad=30)
            ax.set_ylabel('Jitter duration', labelpad=30)
            ax.set_zlabel('Observed effect size', labelpad=40)
            ax.tick_params(axis='z', which='major', pad=15)
            ax.zaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
            plt.title("Observed jitter effect on latencies standard deviation")
            plt.tight_layout()
            plt.savefig(Path(fig_save_root, "{}_latency_std_3d.png".format(component)))
            if show_plots:
                plt.show()
            else:
                plt.close()

            # Plot effect of jitter in 2D:
            for effect_size in comp_jitter_results["effect_size"].unique():
                # Extract the data from this effect size:
                data_df = comp_jitter_results.loc[comp_jitter_results["effect_size"] == effect_size]
                # Remove the rows with 0 jitter:
                data_df = data_df.loc[(data_df["jitter_prop"] != 0) & (data_df["jitter_duration"] != 0)]
                plot_heatmap(data_df.loc[data_df["component"] == component], "jitter_prop", "jitter_duration",
                             "avgs_fsize", xlabel="Jitter duration (ms)",
                             ylabel="Jittered trials proportion (%)", zlabel="$\u03F4_{obs}$",
                             title="Average amplitude \u03F4 as a function of jitter"
                                   "\n comp={}, \u03F4={}".format(component, effect_size), midpoint=0.2)

                plt.savefig(Path(fig_save_root, "{}_jitter_average_fsize_{}.png".format(component, effect_size)))
                if show_plots:
                    plt.show()
                else:
                    plt.close()
                # Same for the peak amplitude:
                plot_heatmap(data_df.loc[data_df["component"] == component], "jitter_prop", "jitter_duration",
                             "peaks_fsize", xlabel="Jitter duration (ms)",
                             ylabel="Jittered trials proportion (%)", zlabel="$\u03F4_{obs}$",
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
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(projection='3d')
            # Add the scatter:
            p = ax.scatter(comp_jitter_results["jitter_prop"].to_numpy(),
                           comp_jitter_results["jitter_duration"].to_numpy(),
                           comp_jitter_results["peaks_tstat"].to_numpy(),
                           c=comp_jitter_results["effect_size"].to_numpy(),
                           cmap=scatter_cmap, s=40)
            # Add a plane:
            ax.plot_trisurf(comp_jitter_results["jitter_prop"].to_numpy(),
                            comp_jitter_results["jitter_duration"].to_numpy(),
                            np.full(comp_jitter_results["jitter_duration"].to_numpy().shape, 1.96),
                            color="b", alpha=0.2)
            cbar = fig.colorbar(p, location='left', shrink=0.7, pad=0.04)
            cbar.ax.set_ylabel('\u03F4', rotation=270, labelpad=40)
            ax.set_xlabel('Jitter proportion (%)', labelpad=30)
            ax.set_ylabel('Jitter duration', labelpad=30)
            ax.set_zlabel('T-statistic', labelpad=40)
            ax.tick_params(axis='z', which='major', pad=15)
            ax.zaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
            plt.title("Observed jitter effect on peak t-stat")
            plt.tight_layout()
            plt.savefig(Path(fig_save_root, "{}_peak_effect_sizes_3d_tstat.png".format(component)))
            if show_plots:
                plt.show()
            else:
                plt.close()

            # Same for the mean amplitude:
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(projection='3d')
            # Add the scatter:
            p = ax.scatter(comp_jitter_results["jitter_prop"].to_numpy(),
                           comp_jitter_results["jitter_duration"].to_numpy(),
                           comp_jitter_results["avgs_tstat"].to_numpy(),
                           c=comp_jitter_results["effect_size"].to_numpy(),
                           cmap=scatter_cmap, s=40)
            # Add a plane:
            ax.plot_trisurf(comp_jitter_results["jitter_prop"].to_numpy(),
                            comp_jitter_results["jitter_duration"].to_numpy(),
                            np.full(comp_jitter_results["jitter_duration"].to_numpy().shape, 1.96),
                            color="b", alpha=0.2)
            cbar = fig.colorbar(p, location='left', shrink=0.7, pad=0.04)
            cbar.ax.set_ylabel('\u03F4', rotation=270, labelpad=40)
            ax.set_xlabel('Jitter proportion (%)', labelpad=30)
            ax.set_ylabel('Jitter duration', labelpad=30)
            ax.set_zlabel('T-statistic', labelpad=40)
            ax.tick_params(axis='z', which='major', pad=15)
            ax.zaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
            plt.title("Observed jitter effect on average t stat")
            plt.tight_layout()
            plt.savefig(Path(fig_save_root, "{}_average_effect_sizes_3d_tstat.png".format(component)))
            if show_plots:
                plt.show()
            else:
                plt.close()

            # Plot effect of jitter in 2D:
            for effect_size in results["effect_size"].unique():
                # Extract the data from this effect size:
                data_df = comp_jitter_results.loc[comp_jitter_results["effect_size"] == effect_size]
                # Remove the rows with 0 jitter:
                data_df = data_df.loc[(data_df["jitter_prop"] != 0) & (data_df["jitter_duration"] != 0)]
                plot_heatmap(data_df.loc[data_df["component"] == component], "jitter_prop", "jitter_duration",
                             "avgs_tstat", xlabel="Jitter duration (ms)",
                             ylabel="Jittered trials proportion (%)", zlabel="T statistic",
                             title="Average t-statistic as a function of jitter"
                                   "\n comp={}, \u03F4={}".format(component, effect_size), midpoint=1.96)
                plt.savefig(Path(fig_save_root, "{}_jitter_average_fsize_{}_tstat.png".format(component, effect_size)))
                if show_plots:
                    plt.show()
                else:
                    plt.close()
                # Same for the peak amplitude:
                plot_heatmap(data_df.loc[data_df["component"] == component], "jitter_prop", "jitter_duration",
                             "peaks_tstat", xlabel="Jitter duration (ms)",
                             ylabel="Jittered trials proportion (%)", zlabel="T statistic",
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
        shuffle_results = shuffle_results.drop_duplicates(subset=["effect_size", "subject", "iteration",
                                                                  "jitter_duration", "jitter_prop",
                                                                  "shuffle_proportion"])
        # Compute correlation for the label shuffle:
        shuffle_proportion_stats = []
        for effect_size in jitter_results["effect_size"].unique():
            corr_data = shuffle_results.loc[
                (shuffle_results["effect_size"] == effect_size)]
            stats_peak = pg.corr(corr_data["shuffle_proportion"].to_numpy(), corr_data["peaks_tstat"].to_numpy(),
                                 alternative="less", method="pearson")
            stats_average = pg.corr(corr_data["shuffle_proportion"].to_numpy(), corr_data["avgs_tstat"].to_numpy(),
                                    alternative="less", method="pearson")
            # Organize the results into dataframe:
            shuffle_proportion_stats.append(pd.Series({
                "effect_size": effect_size,
                "peak r": stats_peak["r"].item(),
                "peak p-value": stats_peak["p-val"].item(),
                "peak CI95": stats_peak["CI95%"].item(),
                "avg r": stats_average["r"].item(),
                "avg p-value": stats_average["p-val"].item(),
                "avg CI95": stats_average["CI95%"].item()
            }))
        shuffle_proportion_stats = pd.DataFrame(shuffle_proportion_stats)
        # Save the results to a dataframe:
        shuffle_proportion_stats.to_csv(Path(results_save_root, "shuffle_proportion_stats.csv"))

        # Remove columns we don't care about:
        shuffle_summary = shuffle_results.loc[:, shuffle_results.columns != "task"]
        shuffle_summary = shuffle_summary.loc[:, shuffle_summary.columns != "subject"]
        shuffle_summary = shuffle_summary.loc[:, shuffle_summary.columns != "iteration"]
        shuffle_summary = shuffle_summary.loc[:, shuffle_summary.columns != "evoked_diff"]
        shuffle_summary = shuffle_summary.loc[:, shuffle_summary.columns != "evoked"]
        # Averaging across iterations within the different manipulated parameters:
        shuffle_summary = shuffle_summary.groupby(["component", "effect_size",
                                                   "jitter_duration", "jitter_prop",
                                                   "shuffle_proportion"]).mean().reset_index()
        shuffle_summary.to_csv(Path(results_save_root, "shuffle_summary.csv"))
        for component in results["component"].unique():
            # Plot the heatmap:
            plot_heatmap(shuffle_summary.loc[shuffle_summary["component"] == component], "shuffle_proportion",
                         "effect_size", "avgs_fsize", xlabel="Simulated Effect sizes",
                         ylabel="Proportion of shuffle trials (%)", zlabel="$\u03F4_{obs}$",
                         title="Average effect size as a function of label shuffles\n comp={}".format(component),
                         midpoint=0.2)
            plt.savefig(Path(fig_save_root, "{}_shuffle_average.png".format(component)))
            if show_plots:
                plt.show()
            else:
                plt.close()
            # Convert long to wide table to generate a heatmap:
            plot_heatmap(shuffle_summary.loc[shuffle_summary["component"] == component], "shuffle_proportion",
                         "effect_size", "peaks_fsize", xlabel="Simulated Effect sizes",
                         ylabel="Proportion of shuffle trials (%)", zlabel="$\u03F4_{obs}$",
                         title="Peak effect size as a function of label shuffles\n comp={}".format(component),
                         midpoint=0.2)
            plt.savefig(Path(fig_save_root, "{}_shuffle_peak.png".format(component)))
            if show_plots:
                plt.show()
            else:
                plt.close()

            # Plot the t-statistic
            plot_heatmap(shuffle_summary.loc[shuffle_summary["component"] == component], "shuffle_proportion",
                         "effect_size", "avgs_tstat", xlabel="Simulated Effect sizes",
                         ylabel="Proportion of shuffle trials (%)", zlabel="T-statistic",
                         title="Average T-statistic as a function of label shuffles\n comp={}".format(component),
                         midpoint=1.96)
            plt.savefig(Path(fig_save_root, "{}_shuffle_average_stat.png".format(component)))
            if show_plots:
                plt.show()
            else:
                plt.close()
            # Peaks:
            plot_heatmap(shuffle_summary.loc[shuffle_summary["component"] == component], "shuffle_proportion",
                         "effect_size", "peaks_tstat", xlabel="Simulated Effect sizes",
                         ylabel="Proportion of shuffle trials (%)", zlabel="T-statistic",
                         title="Average T-statistic as a function of label shuffles\n comp={}".format(component),
                         midpoint=1.96)
            plt.savefig(Path(fig_save_root, "{}_shuffle_peak_tstat.png".format(component)))
            if show_plots:
                plt.show()
            else:
                plt.close()

        # ================================================================================================
        # Plot time series:
        for component in results["component"].unique():
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
