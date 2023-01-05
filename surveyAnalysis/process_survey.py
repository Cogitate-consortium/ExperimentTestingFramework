"""
Loading and plotting module for analysis of survey responses.
author: @RonyHirsch
"""
import re
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

FINAL_PAGE = 3  # the last page in the survey, every participant who finished the survey landed here
FINAL_PAGE_COLNAME = "Last page"
PAGE_2_FIRST = 3
PAGE_2_NOTALL = 17
PAGE_3_FIRST = 31
OTHER = "Other"
TEST_RESP = "please delete"
SOME_OLD = "Some ("
SOME_NEW = "Some (<50%)"
NEVER = "Never (0%)"
RESP_ID = "Response ID"

title_size = 20
x_axis_size = 18
y_axis_size = 18
x_tick_size = 16
y_tick_size = 16


def parse_responses(data_path):
    """
    Method used to parse the raw survey dataframe and perform basic sanity checks before further action is taken.
    :param data_path: path to the survey responses csv file
    :return: If no errors arise, returns the suervey responses dataframe
    """
    data_raw = pd.read_csv(data_path)  # raw data
    data_raw = data_raw.dropna(axis=0, how='all')
    subnum_raw = data_raw.shape[0]
    data_complete = data_raw[data_raw[FINAL_PAGE_COLNAME] == FINAL_PAGE]
    # make sure we get rid of test responses made by the experimenters
    df = data_complete.select_dtypes(object)
    mask = ~df.apply(lambda series: series.str.contains(TEST_RESP, regex=False)).any(axis=1)
    data_complete = data_complete[mask]
    subnum_complete = data_complete.shape[0]
    print(f"NOTE: {subnum_complete} subjects completed the survey (out of {subnum_raw} respondents)")

    # Sanity checks:
    # Question columns: are all columns starting column number 5
    qs_dict = {f"Q{i-4}": data_complete.columns.tolist()[i] for i in range(5, len(data_complete.columns.tolist()))}
    if not data_complete[pd.isna(data_complete[qs_dict["Q2"]])].empty:  # a question in page 1 everyone should fill
        print("ERROR: subjects are marked complete but missed Q2")
        return None
    data_subset_tests = data_complete[data_complete[qs_dict["Q2"]] != "Never (0%)"]  # subjects who DID test their exps
    p2_mandatory = [qs_dict[f"Q{i}"] for i in range(PAGE_2_FIRST, PAGE_2_NOTALL) if OTHER not in qs_dict[f"Q{i}"]]
    if data_subset_tests[p2_mandatory].isnull().values.any():  # subjects tested their exps but had no answers
        print("ERROR: subjects are marked complete and tested their experiments but missing answers from page 2")
        return None

    # Cosmetics
    data_complete.replace({SOME_OLD: SOME_NEW}, inplace=True)

    return data_complete


def save_plot(save_path, save_name, w=10, h=10, dpi=1000):
    """
    Plot saving method
    :param save_path: path to save plots in
    :param save_name: name of saved plot
    :param w: plot width
    :param h: plot height
    :param dpi: plot DPI
    """
    figure = plt.gcf()  # get the current figure
    figure.tight_layout()
    figure.set_size_inches(w, h)
    plt.savefig(os.path.join(save_path, f"PLOT_{save_name}.png"), dpi=dpi)
    return


def plot_part_a(dataset, save_path, qs_dict):
    sns.set_style("white")
    c = [plt.cm.RdBu_r(i / 6) for i in range(6)]
    # Generally, how many experiments were carried out and tested
    # QUESTION 1
    p = sns.histplot(data=dataset, x=qs_dict["Q1"], color=c[2])
    plt.xticks(fontsize=x_tick_size)
    plt.yticks(fontsize=y_tick_size)
    plt.xlabel("Number of different experiments carried out", fontsize=x_axis_size)
    plt.ylabel("Count", fontsize=y_axis_size)
    plt.title("The number of different experiments carried out by respondents", fontsize=title_size)
    save_plot(save_path, "a1", w=12, h=8)
    plt.clf()
    plt.close()

    # QUESTION 2
    p = sns.histplot(data=dataset, x=qs_dict["Q2"], color=c[2])
    plt.xticks(fontsize=x_tick_size)
    plt.yticks(fontsize=y_tick_size)
    plt.xlabel("Experiments for which a test was run", fontsize=x_axis_size)
    plt.ylabel("Count", fontsize=y_axis_size)
    plt.title("The number of different experiments carried out by respondents", fontsize=title_size)
    save_plot(save_path, "a2", w=12, h=8)
    plt.clf()
    plt.close()

    # COMBINATION
    colors = {'Never (0%)': c[5], 'Some (<50%)': c[4], 'Most (>=50%)': c[1], 'All (100%)': c[0]}
    # colors = ["#97AFB3", "#C39345", "#B66E3B", "#DAA79B"]  # all, most, never, some
    # sns.set_palette(sns.color_palette(colors))
    p = sns.histplot(data=dataset, x=qs_dict["Q1"], hue=qs_dict["Q2"], palette=colors, multiple="stack")
    plt.xticks(fontsize=x_tick_size)
    plt.yticks(fontsize=y_tick_size)
    plt.xlabel("Number of different experiments carried out", fontsize=x_axis_size)
    plt.ylabel("Count", fontsize=y_axis_size)
    p.legend_.set_title("Experiments for which a test was run")
    plt.title("The number of different experiments carried out by respondents", fontsize=title_size)
    save_plot(save_path, "a12", w=12, h=8)
    plt.clf()
    plt.close()
    return


def plot_b1(dataset, save_path, qs_dict):
    sns.set_style("white")
    c = [plt.cm.RdBu_r(i / 12) for i in range(12)]

    # QUESTION 1
    b1_cols = [qs_dict[f"Q{i}"] for i in range(3, 8)]
    tested_dataset = dataset[dataset[qs_dict["Q2"]] != NEVER].reset_index(drop=True, inplace=False)  # first, take ONLY people who tested their experiment
    resp_counts = []
    for q in b1_cols:
        resp_counts.append(tested_dataset[q].value_counts())
    tested_aspects = pd.concat(resp_counts, keys=["Experiment Duration",
                                                  "Event Timing",
                                                  "Event Content",
                                                  "Peripherals Integration",
                                                  "Randomization Scheme"], axis=1)
    tested_aspects = tested_aspects.fillna(0).T.reset_index()
    tested_aspects["PcntTested"] = 100 * tested_aspects["Yes"] / (tested_aspects["Yes"] + tested_aspects["No"])
    tested_aspects.sort_values(by="PcntTested", inplace=True)

    # colors = ["#616F76", "#C4C7B4", "#DDD7D1", "#C69376", "#9A5D48"]
    # sns.set_palette(sns.color_palette(colors))
    colors = {'Experiment Duration': c[0], 'Event Timing': c[4], 'Event Content': c[1],
              'Peripherals Integration': c[2], 'Randomization Scheme': c[3]}

    p = sns.barplot(data=tested_aspects, x="index", y="PcntTested", palette=colors)
    plt.yticks(range(0, 101, 20))
    plt.xlabel("Tested Aspect", fontsize=x_axis_size, labelpad=3)
    plt.xticks(fontsize=x_tick_size)
    plt.ylabel("% of Subjects", fontsize=y_axis_size)
    plt.yticks(fontsize=y_tick_size)
    plt.title("Tested Aspects in Published Experiments", fontsize=title_size)
    save_plot(save_path, "b1", w=16, h=9)
    plt.clf()
    plt.close()

    # QUESTION 1 : BY NUMBER OF TESTED ASPECTS
    mapbin = {"Yes": 1, "No": 0}
    for col in b1_cols:
        tested_dataset[col] = tested_dataset[col].map(mapbin)
    tested_dataset['b1_count'] = tested_dataset[b1_cols].sum(axis=1)

    aspect_counts = tested_dataset['b1_count'].value_counts().reset_index(drop=False, inplace=False)
    aspect_counts['b1_count_pcnt'] = (aspect_counts['b1_count'] / aspect_counts['b1_count'].sum()) * 100

    colors = {5: c[0], 4: c[1], 3: c[4], 2: c[8], 1: c[10], 0: c[11]}

    p = sns.barplot(data=aspect_counts, x="index", y="b1_count_pcnt", palette=colors)
    plt.xlabel("Number of Tested Aspects", fontsize=x_axis_size, labelpad=3)
    plt.xticks(fontsize=x_tick_size)
    plt.ylabel("% of Subjects", fontsize=y_axis_size)
    plt.yticks(fontsize=y_tick_size)
    plt.title("Number of Tested Aspects in Published Experiments", fontsize=title_size)
    save_plot(save_path, "b1_counts", w=16, h=9)
    plt.clf()
    plt.close()

    tested_aspects.to_csv(os.path.join(save_path, "b1.csv"))
    aspect_counts.to_csv(os.path.join(save_path, "b1_counts.csv"))
    return


def plot_b2(dataset, save_path, qs_dict):
    # QUESTION 2
    sns.set_style("white")
    c = [plt.cm.RdBu_r(i / 12) for i in range(12)]
    # colors = ["#972B60", "#337C8E", "C74726"]  # both, manual, scripted

    tested_dataset = dataset[dataset[qs_dict["Q2"]] != NEVER].reset_index(drop=True, inplace=False)  # first, take ONLY people who tested their experiment
    # Testing Method (manual, scripted, both)
    method_aspects = tested_dataset[[RESP_ID, qs_dict["Q9"], qs_dict["Q10"]]]  # still using the tested subset
    method_aspects.rename(columns={qs_dict["Q9"]: "Manual", qs_dict["Q10"]: "Scripted"}, inplace=True)
    method_aspects.loc[:, "Method"] = np.where((method_aspects.Manual == "Yes") & (method_aspects.Scripted == "Yes"), "Both (Manual & Scripted)",
                                               np.where((method_aspects.Manual == "Yes") & (method_aspects.Scripted != "Yes"), "Manual", "Scripted"))
    # plot
    method_aspects_cnt = method_aspects.groupby(["Method"])[RESP_ID].count()
    colors = {"Manual": c[7], "Scripted": c[2], "Both (Manual & Scripted)": c[4]}
    label_colors = [colors[label] for label in method_aspects_cnt.keys()]
    plt.pie(x=method_aspects_cnt, autopct="%.1f%%", explode=[0.05] * len(method_aspects_cnt.keys().tolist()),
            labels=method_aspects_cnt.keys(), colors=label_colors, pctdistance=0.5, textprops={'fontsize': x_tick_size})
    plt.title("Testing Method in Published Experiments", fontsize=title_size)
    save_plot(save_path, "b2")
    plt.clf()
    plt.close()
    method_aspects_cnt.to_csv(os.path.join(save_path, "b2.csv"))
    return


def plot_b3(dataset, save_path, qs_dict):
    # QUESTION 3
    sns.set_style("white")
    c = [plt.cm.RdBu_r(i / 12) for i in range(12)]

    tested_dataset = dataset[dataset[qs_dict["Q2"]] != NEVER].reset_index(drop=True, inplace=False)  # first, take ONLY people who tested their experiment
    # Following a Protocol
    protocol_cnt = tested_dataset.groupby([qs_dict["Q11"]])[RESP_ID].count()
    protocol_cnt.rename({"A: I have a protocol of tests all my experiments go through": "A: Unified Protocol",
                         "B: Every experiment had its own protocol of tests": "B: Per Individual Experiment",
                         "I do a mixture of A and B": "Mixture of A&B"}, inplace=True)
    colors = {"B: Per Individual Experiment": c[8], "A: Unified Protocol": c[1], "Mixture of A&B": c[3], "Other": c[5]}
    label_colors = [colors[label] for label in protocol_cnt.keys()]
    # colors = ["#337C8E", "#C69376", "#972B60", "#DDD7D1"]  # A, B, both, other
    # sns.set_palette(sns.color_palette(colors))
    plt.pie(x=protocol_cnt, autopct="%.1f%%", explode=[0.05] * len(protocol_cnt.keys().tolist()),
            labels=protocol_cnt.keys(), colors=label_colors, pctdistance=0.5, textprops={'fontsize': x_tick_size})
    plt.title("Testing Protocol in Published Experiments", fontsize=title_size)
    save_plot(save_path, "b3")
    plt.clf()
    plt.close()
    protocol_cnt.to_csv(os.path.join(save_path, "b3.csv"))
    return


def plot_b4(dataset, save_path, qs_dict):
    # QUESTION 4
    sns.set_style("white")
    c = [plt.cm.RdBu_r(i / 12) for i in range(12)]
    tested_dataset = dataset[dataset[qs_dict["Q2"]] != NEVER].reset_index(drop=True,
                                                                          inplace=False)  # first, take ONLY people who tested their experiment
    # Following a Protocol
    protocol_cnt = tested_dataset.groupby([qs_dict["Q13"]])[RESP_ID].count()
    protocol_cnt.rename({"By computing the overall duration from the log files": "Computed overall duration from logs",
                         "I did not test the overall duration": "Did not test"}, inplace=True)
    colors = {"Using a stopwatch": c[8], "Computed overall duration from logs": c[1], "Both": c[3], "Other": c[5],
              "Did not test": c[10]}
    label_colors = [colors[label] for label in protocol_cnt.keys()]
    plt.pie(x=protocol_cnt, autopct="%.1f%%", explode=[0.05] * len(protocol_cnt.keys().tolist()),
            labels=protocol_cnt.keys(), colors=label_colors, pctdistance=0.5, textprops={'fontsize': x_tick_size})
    plt.title("Experiment Duration Test", fontsize=title_size)
    save_plot(save_path, "b4")
    plt.close()
    protocol_cnt.to_csv(os.path.join(save_path, "b4.csv"))
    return


def plot_b5(dataset, save_path, qs_dict):
    # QUESTION 5
    sns.set_style("white")
    c = [plt.cm.RdBu_r(i / 12) for i in range(12)]
    tested_dataset = dataset[dataset[qs_dict["Q2"]] != NEVER].reset_index(drop=True,
                                                                          inplace=False)  # first, take ONLY people who tested their experiment
    # Following a Protocol
    protocol_cnt = tested_dataset.groupby([qs_dict["Q15"]])[RESP_ID].count()
    protocol_cnt.rename({
                            "By computing the duration of events from their logged timestamps (in the experiment output files)": "Computed from log timestamps",
                            "Using a measuring device (such as an oscilloscope) to compute the duration of a few events manually": "Manually on few sampled events",
                            "Using a recording device (such as a photodiode saving the signal to a file) to compute the duration of all of the events manually": "Manually on all events, with photodiode",
                            "Using a recording device (such as a photodiode saving the signal to a file) to compute the duration of all events using a custom script": "Scripted on all events, with photodiode",
                            "I did not test event timing": "Did not test"}, inplace=True)
    colors = {"Manually on few sampled events": c[9],
              "Manually on all events, with photodiode": c[8],
              "Scripted on all events, with photodiode": c[1],
              "Computed from log timestamps": c[3], "Other": c[5], "Did not test": c[10]}
    label_order = ["Scripted on all events, with photodiode", "Computed from log timestamps", "Other",
                   "Manually on all events, with photodiode", "Manually on few sampled events", "Did not test"]
    label_colors = [colors[label] for label in label_order]
    protocol_cnt = protocol_cnt.reindex(label_order)
    plt.pie(x=protocol_cnt, autopct="%.1f%%", explode=[0.06] * len(label_order),
            labels=label_order, colors=label_colors, pctdistance=0.5)
    patches, texts, auto = plt.pie(x=protocol_cnt, autopct="%.1f%%", explode=[0.06] * len(label_order),
                                   labels=label_order, colors=label_colors, pctdistance=0.5)
    # plt.legend(patches, label_order, loc="best")
    plt.title("Experiment Event Timing Test", fontsize=title_size)
    save_plot(save_path, "b5", w=12, h=8)
    plt.close()
    protocol_cnt.to_csv(os.path.join(save_path, "b5.csv"))
    return


def plot_b6(dataset, save_path, qs_dict):
    # QUESTION B6
    sns.set_style("white")
    c = [plt.cm.RdBu_r(i / 12) for i in range(12)]
    tested_dataset = dataset[dataset[qs_dict["Q2"]] != NEVER].reset_index(drop=True,
                                                                          inplace=False)  # first, take ONLY people who tested their experiment
    b6_cols = [qs_dict[f"Q{i}"] for i in range(17, 20)]

    tested_dataset.loc[:, 'b6'] = None
    tested_dataset.loc[
        ((tested_dataset[qs_dict[f"Q17"]] == "Yes") & (tested_dataset[qs_dict[f"Q18"]] == "Yes")), 'b6'] = "Both"
    tested_dataset.loc[((tested_dataset[qs_dict[f"Q17"]] == "Yes") & (
                tested_dataset[qs_dict[f"Q18"]] != "Yes")), 'b6'] = "Comparing on-screen\nwith logged events"
    tested_dataset.loc[((tested_dataset[qs_dict[f"Q17"]] != "Yes") & (
                tested_dataset[qs_dict[f"Q18"]] == "Yes")), 'b6'] = "Comparing logged events\nwith experimental scheme"
    tested_dataset.loc[((tested_dataset[qs_dict[f"Q17"]] != "Yes") & (
                tested_dataset[qs_dict[f"Q18"]] != "Yes")), 'b6'] = "Did not test"

    aspect_counts = tested_dataset['b6'].value_counts().reset_index(drop=False, inplace=False)
    aspect_counts['b6_count_pcnt'] = (aspect_counts['b6'] / aspect_counts['b6'].sum()) * 100

    colors = {"Comparing on-screen\nwith logged events": c[0],
              "Comparing logged events\nwith experimental scheme": c[2], "Both": c[4], "Did not test": c[9]}

    p = sns.barplot(data=aspect_counts, x="index", y="b6_count_pcnt", palette=colors,
                    order=["Did not test", "Both", "Comparing logged events\nwith experimental scheme",
                           "Comparing on-screen\nwith logged events"])
    plt.xlabel("On-Screen Content Test", fontsize=x_axis_size, labelpad=3)
    plt.xticks(fontsize=x_tick_size)
    plt.ylabel("% of Subjects", fontsize=y_axis_size)
    plt.yticks(fontsize=y_tick_size)
    plt.title("Testing On-Screen Content vc Recorded", fontsize=title_size)
    save_plot(save_path, "b6_counts", w=16, h=9)
    plt.clf()
    plt.close()

    tested_dataset.to_csv(os.path.join(save_path, "b6.csv"))
    aspect_counts.to_csv(os.path.join(save_path, "b6_counts.csv"))
    return


def plot_b7(dataset, save_path, qs_dict):
    # QUESTION B7
    sns.set_style("white")
    c = [plt.cm.RdBu_r(i / 12) for i in range(12)]

    # QUESTION 1
    b7_cols = [qs_dict[f"Q{i}"] for i in range(20, 23)]
    tested_dataset = dataset[dataset[qs_dict["Q2"]] != NEVER].reset_index(drop=True, inplace=False)  # first, take ONLY people who tested their experiment
    resp_counts = []
    for q in b7_cols:
        resp_counts.append(tested_dataset[q].value_counts())
    tested_aspects = pd.concat(resp_counts,
                               keys=["Manually checking the log file to see if the constrains are respected",
                                     "Using scripts automatically investigating the log files to see if the constrains are respected",
                                     "I did not test the pseudo-randomization of events"], axis=1)
    tested_aspects = tested_aspects.fillna(0).T.reset_index()
    tested_aspects["PcntTested"] = 100 * tested_aspects["Yes"] / (tested_aspects["Yes"] + tested_aspects["No"])
    tested_aspects.sort_values(by="PcntTested", inplace=True)
    tested_aspects.loc[:, "index"] = tested_aspects["index"].map(
        {"Manually checking the log file to see if the constrains are respected": "Manually",
         "Using scripts automatically investigating the log files to see if the constrains are respected": "Scripted",
         "I did not test the pseudo-randomization of events": "Did not test"})
    colors = {"Scripted": c[1], "Did not test": c[10], "Manually": c[5]}

    p = sns.barplot(data=tested_aspects, x="index", y="PcntTested", palette=colors,
                    order=["Did not test", "Manually", "Scripted"])
    plt.xlabel("Test Method", fontsize=x_axis_size, labelpad=3)
    plt.xticks(fontsize=x_tick_size)
    plt.ylabel("% of Subjects", fontsize=y_axis_size)
    plt.yticks(fontsize=y_tick_size)
    plt.title("Experiment Pseudo-Randomization Test", fontsize=title_size)
    save_plot(save_path, "b7", w=16, h=9)
    plt.clf()
    plt.close()

    # QUESTION 1 : BY NUMBER OF TESTED ASPECTS
    mapbin = {"Yes": 1, "No": 0}
    for col in b7_cols:
        tested_dataset[col] = tested_dataset[col].map(mapbin)
    tested_dataset['b7_count'] = tested_dataset[b7_cols].sum(axis=1)

    aspect_counts = tested_dataset['b7_count'].value_counts().reset_index(drop=False, inplace=False)
    aspect_counts['b7_count_pcnt'] = (aspect_counts['b7_count'] / aspect_counts['b7_count'].sum()) * 100

    colors = {2: c[3], 1: c[5], 0: c[9]}

    p = sns.barplot(data=aspect_counts, x="index", y="b7_count_pcnt", palette=colors)
    plt.xlabel("Number of Tested Methods Used", fontsize=x_axis_size, labelpad=3)
    plt.xticks(fontsize=x_tick_size)
    plt.ylabel("% of Subjects", fontsize=y_axis_size)
    plt.yticks(fontsize=y_tick_size)
    plt.title("Number of Testing Methods for Pseudo-Randomization", fontsize=title_size)
    save_plot(save_path, "b7_counts", w=16, h=9)
    plt.clf()
    plt.close()

    tested_aspects.to_csv(os.path.join(save_path, "b7.csv"))
    aspect_counts.to_csv(os.path.join(save_path, "b7_counts.csv"))
    return


def plot_b8(dataset, save_path, qs_dict):
    # QUESTION 8 and 10 (for no)
    sns.set_style("white")
    c = [plt.cm.RdBu_r(i / 12) for i in range(12)]
    tested_dataset = dataset[dataset[qs_dict["Q2"]] != NEVER].reset_index(drop=True, inplace=False)  # first, take ONLY people who tested their experiment

    # Did you report the results (yes, no, some): Q24
    # Why not? (Qs 26-28, 29 is the specification for "other")
    no_report = tested_dataset[[RESP_ID, qs_dict["Q24"], qs_dict["Q26"], qs_dict["Q27"]]]
    no_report.rename(columns={qs_dict["Q26"]: "Seems Irrelevant", qs_dict["Q27"]: "Didn't Know Where"}, inplace=True)
    no_report.loc[:, "Reason"] = np.where(
        (no_report["Seems Irrelevant"] == "Yes") & (no_report["Didn't Know Where"] == "Yes"), "Both",
        np.where((no_report["Seems Irrelevant"] == "Yes") & (no_report["Didn't Know Where"] != "Yes"),
                 "Seems Irrelevant", "Didn't Know Where"))
    no_report[qs_dict["Q24"]].replace({"I didn't report the results of any of the tests I performed": "No",
                                       "I reported the results of some of the tests I performed": "Some",
                                       "I reported in detail the results of all the tests I performed": "All"},
                                      inplace=True)
    # Add a collapsed column for plotting
    no_report.loc[:, "col_combination"] = no_report[[qs_dict["Q24"], "Reason"]].agg(':'.join, axis=1)
    no_report_cnt = no_report.groupby(["col_combination"])[RESP_ID].count()

    general_colors = {"All": c[1], "No": c[10], "Some": c[5]}
    general_label_order = ["All", "Some", "No"]
    general_label_colors = [general_colors[label] for label in general_label_order]

    colors = {"All:Didn't Know Where": c[6],
              "Some:Didn't Know Where": c[6], "Some:Seems Irrelevant": c[7], "Some:Both": c[8],
              "No:Didn't Know Where": c[6], "No:Seems Irrelevant": c[7], "No:Both": c[8]}

    specific_colors = {"Didn't Know Where": c[6], "Seems Irrelevant": c[7], "Both": c[8]}
    specific_label_order = ["All:Didn't Know Where",
                            "Some:Didn't Know Where", "Some:Seems Irrelevant", "Some:Both",
                            "No:Didn't Know Where", "No:Seems Irrelevant", "No:Both"]
    label_colors = [colors[label] for label in specific_label_order]
    no_report_cnt = no_report_cnt.reindex(specific_label_order)

    # nested pie plot
    # report_no, report_some = [["#DDD7D1", "#E6B89F", "#C69376", "#9A5D48"],
    #                          ["#DDD7D1", "#A5B8BB", "#7DA1A7", "#2F7384"]]
    fig, ax = plt.subplots()
    ax.axis('equal')

    # outer ring:
    no_report_general = no_report.groupby([qs_dict["Q24"]])[RESP_ID].count()
    # turn to %
    no_report_general = 100 * no_report_general / sum(no_report_general)
    no_report_general = no_report_general.reindex(general_label_order)
    outer, _, junk = ax.pie(x=no_report_general, radius=1.4, colors=general_label_colors,
                            labels=general_label_order, autopct="%.1f%%",
                            textprops={'fontsize': y_axis_size - 2}, pctdistance=0.89)
    plt.setp(outer, width=0.3, edgecolor='white')

    # inner ring:
    labs = [re.sub(r'^.*?:', '', x) for x in specific_label_order]
    no_report_cnt = 100 * no_report_cnt / sum(no_report_cnt)
    inner, _, junk = ax.pie(x=no_report_cnt, radius=1.08, colors=label_colors,
                            labels=labs, labeldistance=0.6, autopct="%.1f%%",
                            textprops={'fontsize': y_tick_size - 3}, pctdistance=0.83)
    plt.setp(inner, width=0.4, edgecolor='white')
    plt.margins(0, 0)
    plt.title("Reporting Experiment Testing in Published Experiments", fontsize=title_size)
    save_plot(save_path, "b8-10")
    plt.clf()
    plt.close()
    no_report_cnt.to_csv(os.path.join(save_path, "b8-10.csv"))
    return


def plot_b11(dataset, save_path, qs_dict):
    # QUESTION 11
    sns.set_style("white")
    c = [plt.cm.RdBu_r(i / 12) for i in range(12)]
    tested_dataset = dataset[dataset[qs_dict["Q2"]] != NEVER].reset_index(drop=True, inplace=False)  # first, take ONLY people who tested their experiment

    # Is the test battery you used replicable: Q29
    replicab = tested_dataset[[RESP_ID, qs_dict["Q29"]]]
    replicab_cnt = replicab.groupby([qs_dict["Q29"]])[RESP_ID].count()

    colors = {"Yes": c[1], "No": c[10], "Other": c[6]}
    label_order = ["Yes", "Other", "No"]
    label_colors = [colors[label] for label in label_order]
    replicab_cnt = replicab_cnt.reindex(label_order)

    plt.pie(x=replicab_cnt, autopct="%.1f%%", colors=label_colors,
            labels=label_order, pctdistance=0.5, textprops={'fontsize': y_axis_size})
    plt.title("Are the Tests Replicable?", fontsize=title_size)
    save_plot(save_path, "b11")
    plt.clf()
    plt.close()
    replicab_cnt.to_csv(os.path.join(save_path, "b11.csv"))
    return


def plot_part_b(dataset, save_path, qs_dict):
    plot_b1(dataset, save_path, qs_dict)
    plot_b2(dataset, save_path, qs_dict)
    plot_b3(dataset, save_path, qs_dict)
    plot_b4(dataset, save_path, qs_dict)
    plot_b5(dataset, save_path, qs_dict)
    plot_b6(dataset, save_path, qs_dict)
    plot_b7(dataset, save_path, qs_dict)
    plot_b8(dataset, save_path, qs_dict)
    plot_b11(dataset, save_path, qs_dict)
    return


def plot_c1(dataset, save_path, qs_dict):
    # QUESTION 1
    sns.set_style("white")
    c = [plt.cm.RdBu_r(i / 12) for i in range(12)]
    # HERE WE WILL TAKE -ALL-SUBJECTS, NOT JUST THOSE WHO FILLED PART B
    tested_dataset = dataset

    # IWhat were the subjects: Q31
    subj = tested_dataset[[RESP_ID, qs_dict["Q31"]]]
    subj_cnt = subj.groupby([qs_dict["Q31"]])[RESP_ID].count()
    colors = {"Human": c[2], "Non-human": c[3], "Both": c[4], "Aliens": c[10]}
    label_order = ["Human", "Non-human", "Both", "Aliens"]
    label_colors = [colors[label] for label in label_order]
    subj_cnt = subj_cnt.reindex(label_order)

    plt.pie(x=subj_cnt, autopct="%.1f%%", colors=label_colors,
            labels=label_order, pctdistance=0.5, textprops={'fontsize': y_axis_size})
    plt.title("What were the subjects?", fontsize=title_size)
    save_plot(save_path, "c1")
    plt.clf()
    plt.close()
    subj_cnt.to_csv(os.path.join(save_path, "c1.csv"))

    return


def plot_c2(dataset, save_path, qs_dict):
    # QUESTION 2
    sns.set_style("white")
    c = [plt.cm.RdBu_r(i / 12) for i in range(12)]
    # HERE WE WILL TAKE -ALL-SUBJECTS, NOT JUST THOSE WHO FILLED PART B
    tested_dataset = dataset

    # What did you record?
    relevant_cols = [RESP_ID] + [qs_dict[f"Q{x}"] for x in range(32, 39)]
    tested_dataset.rename(columns={qs_dict["Q32"]: "Behavior", qs_dict["Q33"]: "Gaze", qs_dict["Q34"]: "Skin Biomarkers",
                             qs_dict["Q35"]: "EEG", qs_dict["Q36"]: "fMRI", qs_dict["Q37"]: "iEEG", qs_dict["Q38"]: "Other"}, inplace=True)

    tested_dataset.loc[:, "Neural Data"] = np.where((tested_dataset["EEG"] == "Yes") |
                                              (tested_dataset["fMRI"] == "Yes") |
                                              (tested_dataset["iEEG"] == "Yes"), "Yes",
                                              np.where((tested_dataset["Other"].str.contains("MEG")) |  # in the "Other" neural data
                                                       (tested_dataset["Other"].str.contains("electrophysiology")) |
                                                       (tested_dataset["Other"].str.contains("ERP")) |
                                                       (tested_dataset["Other"].str.contains("PET")) |
                                                       (tested_dataset["Other"].str.contains("single unit")) |
                                                       (tested_dataset["Other"].str.contains("Single-cell")) |
                                                       (tested_dataset["Other"].str.contains("MRI") |
                                                        (tested_dataset["Other"].str.contains("ECOG"))), "Yes", "No"))

    tested_dataset.loc[:, "Behavioral / Physchological Data"] = np.where((tested_dataset["Behavior"] == "Yes") |
                                                                       (tested_dataset["Gaze"] == "Yes") |
                                                                       (tested_dataset["Skin Biomarkers"] == "Yes"), "Yes",
                                                                         np.where((tested_dataset["Other"].str.contains("EMG")) |  # in the "Other"
                                                                        (tested_dataset["Other"].str.contains("sensors")) |
                                                                         (tested_dataset["Other"].str.contains("respiration")) |
                                                                         (tested_dataset["Other"].str.contains("MRI")), "Yes", "No"))

    tested_dataset.loc[:, "Recorded"] = np.where((tested_dataset["Neural Data"] == "Yes") &
                                           (tested_dataset["Behavioral / Physchological Data"] == "Yes"), "Both",
                                           np.where(tested_dataset["Neural Data"] == "Yes", "Neural Data", "Behavioral / Physchological Data"))
    recorded_cnt = tested_dataset.groupby(["Recorded"])[RESP_ID].count()

    colors = {"Behavioral / Physchological Data": c[6], "Neural Data": c[2], "Both": c[4]}
    label_order = ["Neural Data", "Behavioral / Physchological Data", "Both"]
    label_colors = [colors[label] for label in label_order]
    recorded_cnt = recorded_cnt.reindex(label_order)

    plt.pie(x=recorded_cnt, autopct="%.1f%%", colors=label_colors,
            labels=label_order, pctdistance=0.5, textprops={'fontsize': y_axis_size})
    plt.title("What was Recorded?", fontsize=title_size)
    save_plot(save_path, "c2")
    plt.clf()
    plt.close()
    recorded_cnt.to_csv(os.path.join(save_path, "c2.csv"))
    return


def plot_c3(dataset, save_path, qs_dict):
    # QUESTION 3
    sns.set_style("white")
    c = [plt.cm.RdBu_r(i / 12) for i in range(12)]

    # Have you ever noticed an issue after data collection has begun?
    # PERFORMED TESTS
    tested_dataset = dataset[dataset[qs_dict["Q2"]] != NEVER].reset_index(drop=True,  inplace=False)  # first, take ONLY people who tested their experiment
    issues_tested = tested_dataset[[RESP_ID, qs_dict[f"Q39"]]]
    issues_tested_cnt = issues_tested.groupby([qs_dict[f"Q39"]])[RESP_ID].count()
    issues_tested_cnt = 100 * issues_tested_cnt / sum(issues_tested_cnt)  # TO %S

    colors = {"No": c[1], "Yes": c[10]}
    label_order = ["Yes", "No"]
    label_colors = [colors[label] for label in label_order]
    issues_tested_cnt = issues_tested_cnt.reindex(label_order)

    plt.pie(x=issues_tested_cnt, autopct="%.1f%%", labels=label_order, pctdistance=0.5, colors=label_colors)
    plt.title("Noticed an issue that could have been prevented [Tested]", fontsize=title_size)
    save_plot(save_path, "c3_tested")
    plt.clf()
    plt.close()
    issues_tested_cnt.to_csv(os.path.join(save_path, "c3_tested.csv"))

    # and now to those who did not perform tests
    nontested_dataset = dataset[dataset[qs_dict["Q2"]] == NEVER].reset_index(drop=True,  inplace=False)
    issues_not_tested = nontested_dataset[[RESP_ID, qs_dict[f"Q39"]]]  # NO TESTS
    issues_not_tested_cnt = issues_not_tested.groupby([qs_dict[f"Q39"]])[RESP_ID].count()
    issues_not_tested_cnt = 100 * issues_not_tested_cnt / sum(issues_not_tested_cnt)  # TO %S
    # a 100% YES!
    plt.pie(x=issues_not_tested_cnt, autopct="%.1f%%", labels=["Yes"], pctdistance=0.5, colors=[c[10]])
    plt.title("Noticed an issue that could have been prevented [Not Tested]", fontsize=title_size)
    save_plot(save_path, "c3_not")
    plt.clf()
    plt.close()
    issues_not_tested_cnt.to_csv(os.path.join(save_path, "c3_not.csv"))
    return


def plot_c4(dataset, save_path, qs_dict):
    # QUESTION 4
    sns.set_style("white")
    c = [plt.cm.RdBu_r(i / 12) for i in range(12)]

    # Dicipline in neuroscience
    dicipline_cnt = dataset.groupby([qs_dict[f"Q40"]])[RESP_ID].count()  # ALL DATA

    colors = {"Biology": c[3], "Psychology": c[4], "Psychiatry": c[5], "Other": c[6]}
    label_order = ["Biology", "Psychology", "Psychiatry", "Other"]
    label_colors = [colors[label] for label in label_order]
    dicipline_cnt = dicipline_cnt.reindex(label_order)
    dicipline_cnt = 100 * dicipline_cnt / sum(dicipline_cnt)  # TO %S

    plt.pie(x=dicipline_cnt, autopct="%.1f%%", colors=label_colors, labels=label_order, pctdistance=0.5, textprops={'fontsize': y_axis_size})
    plt.title("Dicipline", fontsize=title_size)
    save_plot(save_path, "c4")
    plt.clf()
    plt.close()
    dicipline_cnt.to_csv(os.path.join(save_path, "c4.csv"))
    return


def plot_c5(dataset, save_path, qs_dict):
    sns.set_style("white")
    c = [plt.cm.RdBu_r(i / 12) for i in range(12)]
    # What is your position?
    position_cnt = dataset.groupby([qs_dict[f"Q42"]])[RESP_ID].count()  # ALL DATA

    colors = {"Undergraduate/ bachelor/ master's student": c[4], "Graduate student/ PhD student": c[3],
                   "Post doc/senior researcher": c[2], "PI/Group leader": c[1], "Research assistant": c[5], "Other": c[6]}
    label_order = ["Undergraduate/ bachelor/ master's student", "Graduate student/ PhD student",
                   "Post doc/senior researcher", "PI/Group leader", "Research assistant", "Other"]
    label_colors = [colors[label] for label in label_order]
    position_cnt = position_cnt.reindex(label_order)
    position_cnt = 100 * position_cnt / sum(position_cnt)  # TO %S

    plt.pie(x=position_cnt, autopct="%.1f%%", colors=label_colors, labels=label_order, pctdistance=0.5, textprops={'fontsize': y_axis_size-7})
    plt.title("Position", fontsize=title_size)
    save_plot(save_path, "c5")
    plt.clf()
    plt.close()
    position_cnt.to_csv(os.path.join(save_path, "c5.csv"))
    return


def plot_part_c(dataset, save_path, qs_dict):
    plot_c1(dataset, save_path, qs_dict)
    plot_c2(dataset, save_path, qs_dict)
    plot_c3(dataset, save_path, qs_dict)
    plot_c4(dataset, save_path, qs_dict)
    plot_c5(dataset, save_path, qs_dict)
    return


def plot_data(dataset, save_path):
    """
    Manages all plotting of data
    :param dataset: survey responses dataframe
    :param save_path: path to save plots in
    """
    qs_dict = {f"Q{i-4}": dataset.columns.tolist()[i] for i in range(5, len(dataset.columns.tolist()))}

    plot_part_a(dataset, save_path, qs_dict)

    plot_part_b(dataset, save_path, qs_dict)

    plot_part_c(dataset, save_path, qs_dict)

    return


def manage_processing(data_path, save_path):
    """
    Manage any processing of the survey responses.
    :param data_path: path to the survey csv file
    :param save_path: path to save all plots in
    """
    dataset = parse_responses(data_path)  # parse the data

    dataset.to_csv(os.path.join(save_path, "processed_dataset.csv"))
    plot_data(dataset, save_path)


if __name__ == "__main__":
    manage_processing(data_path=r"\results-survey.csv",
                    save_path=r"")
