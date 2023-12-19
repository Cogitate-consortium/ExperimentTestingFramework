from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import StrMethodFormatter
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors
import os
from matplotlib import rcParams

plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
rcParams['axes.titlepad'] = 50
scatter_cmap = "autumn_r"
fig_size = [20, 15]
SMALL_SIZE = 40
MEDIUM_SIZE = 50
BIGGER_SIZE = 54
gaussian_sig = 4
azimuth = -50
elevation = 5
frame_width = 10
ticks_n = 2
title_padding = 50
xaxis_padding = 50
yaxis_padding = 50
zaxis_padding = 70
zticks_padding = 30
cbar_label_padding = 60
cmap = 'RdYlBu_r'
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the fi

show_results = False


def plot_jitters_sims(results_file, bids_root, signal, threshold=1.96):
    """
    This function plots the t statistics as a function of jitter duration and jitter proportion
    :param results_file:
    :param bids_root:
    :param signal:
    :param threshold:
    :return:
    """
    fig_save_root = Path(bids_root, "derivatives", signal)
    if not os.path.isdir(fig_save_root):
        os.makedirs(fig_save_root)
    # Load the file:
    results = pd.read_csv(results_file)
    results["jitter proportion"] = results["jitter proportion"] * 100
    results["jitter proportion"] = results["jitter proportion"].round()

    # Averaging across iterations within the different manipulated parameters:
    jitter_summary = results.groupby(["fsize", "jitter duration", "jitter proportion"]).mean().reset_index()
    jitter_summary.to_csv(Path(fig_save_root, "jitter_summary.csv"))
    # Subselecting only every second effect sizes for 3d plots:
    fsizes = jitter_summary["fsize"].unique()
    # Select only every second:
    sel_fsizes = fsizes[1::2]
    jitters_3d = jitter_summary.loc[jitter_summary["fsize"].isin(sel_fsizes)]
    # Plot the reaction time as a function of jitter duration and proportion in 3D:
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(projection='3d')
    # Add the scatter
    p = ax.scatter(jitters_3d["jitter proportion"].to_numpy(),
                   jitters_3d["jitter duration"].to_numpy(),
                   jitters_3d["observed f size"].to_numpy(),
                   c=jitters_3d["fsize"].to_numpy(),
                   cmap=scatter_cmap, s=0)
    # Add the surfaces:
    norm = Normalize(vmin=np.min(jitters_3d["fsize"].to_numpy()), vmax=np.max(jitters_3d["fsize"].to_numpy()))
    cmap = get_cmap(scatter_cmap)
    for fsize in jitters_3d["fsize"].unique():
        df = jitters_3d.loc[jitters_3d["fsize"] == fsize].reset_index(drop=True)
        # Add hyperplanes:
        ax.plot_trisurf(df["jitter proportion"].to_numpy(),
                        df["jitter duration"].to_numpy(),
                        df["observed f size"].to_numpy(), alpha=0.4,
                        color=cmap(norm(fsize)))
    cbar = fig.colorbar(p, location='left', shrink=0.7, pad=0.04)
    cbar.ax.set_ylabel('\u03F4', rotation=270, labelpad=cbar_label_padding)
    ax.set_xlabel('Jitter proportion (%)', labelpad=xaxis_padding)
    ax.set_ylabel('Jitter duration (ms)', labelpad=yaxis_padding)
    ax.set_zlabel('Observed effect size', labelpad=zaxis_padding)
    ax.tick_params(axis='z', which='major', pad=zticks_padding)
    ax.zaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    # Hide every second ticks:
    [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % ticks_n != 0]
    [l.set_visible(False) for (i, l) in enumerate(ax.yaxis.get_ticklabels()) if i % ticks_n != 0]
    [l.set_visible(False) for (i, l) in enumerate(ax.zaxis.get_ticklabels()) if i % ticks_n != 0]
    [l.set_visible(False) for (i, l) in enumerate(cbar.ax.yaxis.get_ticklabels()) if i % ticks_n != 0]
    plt.title("Observed jitter effect on effect size")
    plt.tight_layout()
    ax.view_init(elevation, azimuth)
    plt.savefig(Path(fig_save_root, "Effect_sizes_3d.png"))
    if show_results:
        plt.show()
    plt.close()

    # Same for the t-statistic:
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(projection='3d')
    # Add the scatter
    p = ax.scatter(jitters_3d["jitter proportion"].to_numpy(),
                   jitters_3d["jitter duration"].to_numpy(),
                   jitters_3d["t statistic"].to_numpy(),
                   c=jitters_3d["fsize"].to_numpy(),
                   cmap=scatter_cmap, s=0)
    # Add the surfaces:
    norm = Normalize(vmin=np.min(jitters_3d["fsize"].to_numpy()), vmax=np.max(jitters_3d["fsize"].to_numpy()))
    cmap = get_cmap(scatter_cmap)
    for ind, fsize in enumerate(jitters_3d["fsize"].unique()):
        df = jitters_3d.loc[jitters_3d["fsize"] == fsize].reset_index(drop=True)
        # Add hyperplanes:
        ax.plot_trisurf(df["jitter proportion"].to_numpy(),
                        df["jitter duration"].to_numpy(),
                        df["t statistic"].to_numpy(), alpha=0.4,
                        color=cmap(norm(fsize)))
        if ind == 0:
            # Plot the threshold plane to extend for the entire xlim and y lime:
            x_thresh = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num=20)
            y_thresh = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num=20)
            x2, y2 = np.meshgrid(x_thresh, y_thresh)
            thresh = np.zeros(x2.shape)
            thresh[:] = threshold
            ax.plot_trisurf(x2.flatten(),
                            y2.flatten(),
                            thresh.flatten(), alpha=0.8,
                            color=[0.5, 0.5, 0.5])
            ax.set_xlim(np.min(x_thresh), np.max(x_thresh))
            ax.set_ylim(np.min(y_thresh), np.max(y_thresh))
    cbar = fig.colorbar(p, location='left', shrink=0.7, pad=0.04)
    cbar.ax.set_ylabel('\u03F4', rotation=270, labelpad=cbar_label_padding)
    ax.set_xlabel('Jitter proportion (%)', labelpad=xaxis_padding)
    ax.set_ylabel('Jitter duration (ms)', labelpad=yaxis_padding)
    ax.set_zlabel('T statistic', labelpad=zaxis_padding)
    ax.tick_params(axis='z', which='major', pad=zticks_padding)
    ax.zaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % ticks_n != 0]
    [l.set_visible(False) for (i, l) in enumerate(ax.yaxis.get_ticklabels()) if i % ticks_n != 0]
    [l.set_visible(False) for (i, l) in enumerate(ax.zaxis.get_ticklabels()) if i % ticks_n != 0]
    [l.set_visible(False) for (i, l) in enumerate(cbar.ax.yaxis.get_ticklabels()) if i % ticks_n != 0]
    plt.title("Observed jitter effect on T statistic")
    plt.tight_layout()
    ax.view_init(elevation, azimuth)
    plt.savefig(Path(fig_save_root, "Tstat_3d.png"))
    if show_results:
        plt.show()
    plt.close()

    # Plot each effect size separately
    for fsize in results["fsize"].unique():
        # Plot the observed effect size:
        plot_heatmap(jitter_summary.loc[jitter_summary["fsize"] == fsize], "jitter proportion", "jitter duration",
                     "observed f size", xlabel="Jitter duration (ms)",
                     ylabel="Jittered trials proportion (%)", zlabel="$\u03F4_{obs}$",
                     title="Average amplitude \u03F4 as a function of jitter "
                           "\u03F4={}".format(fsize), midpoint=0.2, frame_color=cmap(norm(fsize)))
        plt.savefig(Path(fig_save_root, "rt_jitter_average_fsize_{}.png".format(fsize)))
        plt.close()
        # Plot the t statistic:
        plot_heatmap(jitter_summary.loc[jitter_summary["fsize"] == fsize], "jitter proportion", "jitter duration",
                     "t statistic", xlabel="Jitter duration (ms)",
                     ylabel="Jittered trials proportion (%)", zlabel="T stat",
                     title="T-stat as a function of jitter "
                           "\u03F4={}".format(fsize), midpoint=threshold,
                     frame_color=cmap(norm(fsize)))
        plt.savefig(Path(fig_save_root, "rt_jitter_tstat_{}.png".format(fsize)))
        plt.close()
        # Plot the thresholded maps:
        plot_heatmap(jitter_summary.loc[jitter_summary["fsize"] == fsize], "jitter proportion", "jitter duration",
                     "t statistic", xlabel="Jitter duration (ms)",
                     ylabel="Jittered trials proportion (%)", zlabel="T stat",
                     title="T-stat as a function of jitter "
                           "\u03F4={}".format(fsize), midpoint=threshold,
                     frame_color=cmap(norm(fsize)), threshold=threshold)
        plt.savefig(Path(fig_save_root, "rt_jitter_tstat_{}_thresh.png".format(fsize)))
        plt.close()
    return None


def plot_shuffle_sims(results_file, bids_root, signal, threshold=1.96):
    """

    :param results_file:
    :param bids_root:
    :param signal:
    :param threshold:
    :return:
    """
    fig_save_root = Path(bids_root, "derivatives", signal)
    if not os.path.isdir(fig_save_root):
        os.makedirs(fig_save_root)
    # Load the file:
    results = pd.read_csv(results_file)
    results["shuffle proportion"] = results["shuffle proportion"] * 100
    results["shuffle proportion"] = results["shuffle proportion"].round()
    shuffle_summary = results.groupby(["fsize", "shuffle proportion"]).mean().reset_index()

    # Plotting the results:
    plot_heatmap(shuffle_summary, "shuffle proportion",
                 "fsize", "observed f size", xlabel="Simulated Effect sizes",
                 ylabel="Proportion of shuffle trials (%)", zlabel="$\u03F4_{obs}$",
                 title="Observed effect size as a function of label shuffles",
                 midpoint=0.2)
    plt.savefig(Path(fig_save_root, "rt_shuffle_fsize.png"))
    plt.close()
    # Same for the t-statistic:
    plot_heatmap(shuffle_summary, "shuffle proportion",
                 "fsize", "t statistic", xlabel="Simulated Effect sizes",
                 ylabel="Proportion of shuffle trials (%)", zlabel="$\u03F4_{obs}$",
                 title="Observed effect size as a function of label shuffles",
                 midpoint=threshold)
    plt.savefig(Path(fig_save_root, "rt_shuffle_tstat.png"))
    plt.close()
    # Same for the t-statistic:
    plot_heatmap(shuffle_summary, "shuffle proportion",
                 "fsize", "t statistic", xlabel="Simulated Effect sizes",
                 ylabel="Proportion of shuffle trials (%)", zlabel="$\u03F4_{obs}$",
                 title="Observed effect size as a function of label shuffles",
                 midpoint=threshold, threshold=threshold)
    plt.savefig(Path(fig_save_root, "rt_shuffle_tstat_thresh.png"))
    plt.close()

    return None


def plot_heatmap(df, index, column, values, xlabel="", ylabel="", zlabel="", title="", cmap="RdYlBu_r", midpoint=1.96,
                 frame_color=None, threshold=None):
    """

    :param df:
    :param index:
    :param column:
    :param values:
    :param xlabel:
    :param ylabel:
    :param zlabel:
    :param title:
    :param cmap:
    :param midpoint:
    :param frame_color:
    :param threshold:
    :return:
    """
    # Convert long to wide table to generate a heatmap:
    avg_effect_size = df.pivot(index=index, columns=column, values=values)
    # Add color map
    norm = MidpointNormalize(vmin=np.min(avg_effect_size.to_numpy()), vmax=np.max(avg_effect_size.max()),
                             midpoint=midpoint)
    # Generate a heatmap:
    if frame_color is not None:
        fig, ax = plt.subplots(1, figsize=fig_size, edgecolor=frame_color, linewidth=frame_width)
    else:
        fig, ax = plt.subplots(1, figsize=fig_size)
    if threshold is None:
        ax = sns.heatmap(avg_effect_size, ax=ax, cmap=cmap, norm=norm,
                         cbar_kws={'label': zlabel})
    else:
        # Create a custom color bar:
        colors2 = plt.cm.YlOrBr(np.linspace(0., 1, 128))
        colors1 = plt.cm.Blues_r(np.linspace(0, 1, 128))
        colors = np.vstack((colors1, colors2))
        mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        ax = sns.heatmap(avg_effect_size, ax=ax, cmap=mymap, norm=norm)
    ax.set_xlabel(xlabel, labelpad=xaxis_padding)
    ax.set_ylabel(ylabel, labelpad=yaxis_padding)
    ax.set_title(title)
    cbar = ax.collections[0].colorbar
    cbar.set_label(zlabel, labelpad=cbar_label_padding)
    labels = np.arange(0.5, np.max(avg_effect_size.to_numpy()), 0.5)
    loc = labels
    if len(labels) > 3:
        cbar.set_ticks(loc)
        cbar.set_ticklabels(labels)
    [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % ticks_n != 0]
    [l.set_visible(False) for (i, l) in enumerate(ax.yaxis.get_ticklabels()) if i % ticks_n != 0]
    plt.tight_layout()
    return fig


class MidpointNormalize(Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))
