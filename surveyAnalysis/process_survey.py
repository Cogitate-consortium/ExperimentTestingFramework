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


def plot_data(dataset, save_path):
    """
    Manages all plotting of data
    :param dataset: survey responses dataframe
    :param save_path: path to save plots in
    """
    qs_dict = {f"Q{i-4}": dataset.columns.tolist()[i] for i in range(5, len(dataset.columns.tolist()))}
    # Generally, how many experiments were carried out and tested
    colors = ["#97AFB3", "#C39345", "#B66E3B", "#DAA79B"]  # all, most, never, some
    sns.set_palette(sns.color_palette(colors))
    sns.set_style("whitegrid")
    p = sns.histplot(data=dataset, x=qs_dict["Q1"], hue=qs_dict["Q2"], multiple="stack")
    plt.xlabel("Number of different experiments carried out")
    p.legend_.set_title("Experiments for which a test was run")
    plt.title("The number of different experiments carried out by respondents")
    save_plot(save_path, "1")
    plt.clf()
    plt.close()

    # Tested aspects
    tested_dataset = dataset[dataset[qs_dict["Q2"]] != NEVER]  # first, take ONLY people who tested their experiment

    resp_counts = []
    for q in [qs_dict[f"Q{i}"] for i in range(3, 8)]:
        resp_counts.append(tested_dataset[q].value_counts())
    tested_aspects = pd.concat(resp_counts, keys=["Experiment Duration",
                                              "Event Timing",
                                              "Event Content",
                                              "Peripherals Integration",
                                              "Randomization Scheme"], axis=1)
    tested_aspects = tested_aspects.fillna(0).T.reset_index()
    tested_aspects["PcntTested"] = 100 * tested_aspects["Yes"] / (tested_aspects["Yes"] + tested_aspects["No"])
    tested_aspects.sort_values(by="PcntTested", inplace=True)
    colors = ["#616F76", "#C4C7B4", "#DDD7D1", "#C69376", "#9A5D48"]
    sns.set_palette(sns.color_palette(colors))
    sns.set_style("whitegrid")
    p = sns.barplot(data=tested_aspects, x="index", y="PcntTested")
    plt.yticks(range(0, 101, 20))
    plt.xlabel("Tested Aspect")
    plt.ylabel("% of Subjects")
    plt.title("Tested Aspects in Published Experiments")
    save_plot(save_path, "2")
    plt.clf()
    plt.close()

    # Testing Method
    method_aspects = tested_dataset[[RESP_ID, qs_dict["Q9"], qs_dict["Q10"]]]  # still using the tested subset
    method_aspects.rename(columns={qs_dict["Q9"]: "Manual", qs_dict["Q10"]: "Scripted"}, inplace=True)
    method_aspects.loc[:, "Method"] = np.where((method_aspects.Manual == "Yes") & (method_aspects.Scripted == "Yes"), "Both (Manual & Scripted)",
                                      np.where((method_aspects.Manual == "Yes") & (method_aspects.Scripted != "Yes"), "Manual", "Scripted"))
    # plot
    method_aspects_cnt = method_aspects.groupby(["Method"])[RESP_ID].count()
    colors = ["#972B60", "#337C8E", "C74726"]  # both, manual, scripted
    sns.set_palette(sns.color_palette(colors))
    sns.set_style("whitegrid")
    plt.pie(x=method_aspects_cnt, autopct="%.1f%%", explode=[0.05]*len(method_aspects_cnt.keys().tolist()), labels=method_aspects_cnt.keys(), pctdistance=0.5)
    plt.title("Testing Method in Published Experiments")
    save_plot(save_path, "3")
    plt.clf()
    plt.close()

    # Following a Protocol
    protocol_cnt = tested_dataset.groupby([qs_dict["Q11"]])[RESP_ID].count()
    protocol_cnt.rename({"A: I have a protocol of tests all my experiments go through": "A: Unified Protocol",
                         "B: Every experiment had its own protocol of tests": "B: Per Individual Experiment",
                         "I do a mixture of A and B": "Mixture of A&B"}, inplace=True)
    colors = ["#337C8E", "#C69376", "#972B60", "#DDD7D1"]  # A, B, both, other
    sns.set_palette(sns.color_palette(colors))
    sns.set_style("whitegrid")
    plt.pie(x=protocol_cnt, autopct="%.1f%%", explode=[0.05] * len(protocol_cnt.keys().tolist()), labels=protocol_cnt.keys(), pctdistance=0.5)
    plt.title("Testing Protocol in Published Experiments")
    save_plot(save_path, "4")
    plt.clf()
    plt.close()

    # Did you report the results (yes, no, some): Q24
    # Why not? (Qs 26-28, 29 is the specification for "other")
    no_report = tested_dataset[[RESP_ID, qs_dict["Q24"], qs_dict["Q26"], qs_dict["Q27"]]]
    no_report.rename(columns={qs_dict["Q26"]: "Seems Irrelevant", qs_dict["Q27"]: "Didn't Know Where"}, inplace=True)
    no_report.loc[:, "Reason"] = np.where((no_report["Seems Irrelevant"] == "Yes") & (no_report["Didn't Know Where"] == "Yes"),"Both",
                                               np.where((no_report["Seems Irrelevant"] == "Yes") & (no_report["Didn't Know Where"] != "Yes"),
                                                        "Seems Irrelevant", "Didn't Know Where"))
    no_report[qs_dict["Q24"]].replace({"I didn't report the results of any of the tests I performed": "No",
                       "I reported the results of some of the tests I performed": "Some"}, inplace=True)
    # Add a collapsed column for plotting
    no_report.loc[:, "col_combination"] = no_report[[qs_dict["Q24"], "Reason"]].agg(':'.join, axis=1)
    no_report_cnt = no_report.groupby(["col_combination"])[RESP_ID].count()
    # nested pie plot
    report_no, report_some = [["#DDD7D1", "#E6B89F", "#C69376", "#9A5D48"], ["#DDD7D1", "#A5B8BB", "#7DA1A7", "#2F7384"]]
    fig, ax = plt.subplots()
    ax.axis('equal')
    # outer ring:
    no_report_general = no_report.groupby([qs_dict["Q24"]])[RESP_ID].count()
    outer, _ = ax.pie(x=no_report_general, radius=1.3, colors=[report_no[3], report_some[3]],
                      labels=no_report_general.keys(), textprops={'fontsize': 12})
    plt.setp(outer, width=0.3, edgecolor='white')
    # inner ring:
    labs = [re.sub(r'^.*?:', '', x) for x in no_report_cnt.keys()]
    inner, _ = ax.pie(x=no_report_cnt, radius=1.3-0.3, colors=report_no[:3] + report_some[:3],
                      labels=labs, labeldistance=0.65, textprops={'fontsize': 8.5})
    plt.setp(inner, width=0.4, edgecolor='white')
    plt.margins(0, 0)
    plt.title("Reporting Experiment Testing in Published Experiments")
    save_plot(save_path, "5")
    plt.clf()
    plt.close()

    # Is the test battery you used replicable: Q29
    replicab = tested_dataset[[RESP_ID, qs_dict["Q29"]]]
    replicab_cnt = replicab.groupby([qs_dict["Q29"]])[RESP_ID].count()
    colors = ["#972B60", "#DDD7D1", "#337C8E"]  # No, Other, Yes
    sns.set_palette(sns.color_palette(colors))
    sns.set_style("whitegrid")
    plt.pie(x=replicab_cnt, autopct="%.1f%%", explode=[0.05] * len(replicab_cnt.keys().tolist()),
            labels=replicab_cnt.keys(), pctdistance=0.5)
    plt.title("Are the Tests Replicable?")
    save_plot(save_path, "6")
    plt.clf()
    plt.close()

    # What did you record?
    relevant_cols = [RESP_ID] + [qs_dict[f"Q{x}"] for x in range(32, 39)]
    recorded = tested_dataset[relevant_cols]  # PERFORMED TESTS
    recorded.rename(columns={qs_dict["Q32"]: "Behavior", qs_dict["Q33"]: "Gaze", qs_dict["Q34"]: "Skin Biomarkers",
                             qs_dict["Q35"]: "EEG", qs_dict["Q36"]: "fMRI", qs_dict["Q37"]: "iEEG", qs_dict["Q38"]: "Other"}, inplace=True)

    recorded.loc[:, "Neural Data"] = np.where((recorded["EEG"] == "Yes") |
                                              (recorded["fMRI"] == "Yes") |
                                              (recorded["iEEG"] == "Yes"), "Yes",
                                              np.where((recorded["Other"].str.contains("MEG")) |  # in the "Other" neural data
                                                       (recorded["Other"].str.contains("electrophysiology")), "Yes", "No"))

    recorded.loc[:, "Behavioral and Other Biological Data"] = np.where((recorded["Behavior"] == "Yes") |
                                                                       (recorded["Gaze"] == "Yes") |
                                                                       (recorded["Skin Biomarkers"] == "Yes"), "Yes", "No")
    recorded.loc[:, "Recorded"] = np.where((recorded["Neural Data"] == "Yes") &
                                           (recorded["Behavioral and Other Biological Data"] == "Yes"), "Both Neural and Behavioral/Physio",
                                           np.where(recorded["Neural Data"] == "Yes", "Neural Data", "Behavior/Physio"))
    recorded_cnt = recorded.groupby(["Recorded"])[RESP_ID].count()
    colors = ["#616F76", "#C4C7B4", "#DAA79B"]  # Behavioral/Other, Both, Neural
    sns.set_palette(sns.color_palette(colors))
    sns.set_style("whitegrid")
    plt.pie(x=recorded_cnt, autopct="%.1f%%", explode=[0.05] * len(recorded_cnt.keys().tolist()),
            labels=recorded_cnt.keys(), pctdistance=0.5)
    plt.title("What was Recorded?")
    save_plot(save_path, "7")
    plt.clf()
    plt.close()

    # Have you ever noticed an issue after data collection has begun?
    issues_tested = tested_dataset[[RESP_ID, qs_dict[f"Q39"]]]  # PERFORMED TESTS
    issues_tested_cnt = issues_tested.groupby([qs_dict[f"Q39"]])[RESP_ID].count()
    colors = ["#337C8E", "#972B60"]  # No, Yes
    sns.set_palette(sns.color_palette(colors))
    sns.set_style("whitegrid")
    plt.pie(x=issues_tested_cnt, autopct="%.1f%%", explode=[0.05] * len(issues_tested_cnt.keys().tolist()),
            labels=issues_tested_cnt.keys(), pctdistance=0.5)
    plt.title("Noticed an issue that could have been prevented with tests? [Tested]")
    save_plot(save_path, "8")
    plt.clf()
    plt.close()

    nontested_dataset = dataset[dataset[qs_dict["Q2"]] == NEVER]
    issues_not_tested = nontested_dataset[[RESP_ID, qs_dict[f"Q39"]]]  # NO TESTS
    issues_not_tested_cnt = issues_not_tested.groupby([qs_dict[f"Q39"]])[RESP_ID].count()
    plt.pie(x=issues_not_tested_cnt, autopct="%.1f%%", explode=[0.05] * len(issues_not_tested_cnt.keys().tolist()),
            labels=issues_not_tested_cnt.keys(), pctdistance=0.5)
    plt.title("Noticed an issue that could have been prevented with tests? [No Tests]")
    save_plot(save_path, "9")
    plt.clf()
    plt.close()

    # Dicipline in neuroscience
    dicipline_cnt = dataset.groupby([qs_dict[f"Q40"]])[RESP_ID].count()  # ALL DATA
    colors = ["#C39345", "#616F76", "#3B40B8", "#337C8E"]  # Biology, Other, Psychiatry, Psychology
    sns.set_palette(sns.color_palette(colors))
    sns.set_style("whitegrid")
    plt.pie(x=dicipline_cnt, autopct="%.1f%%", explode=[0.05] * len(dicipline_cnt.keys().tolist()),
            labels=dicipline_cnt.keys(), pctdistance=0.5)
    plt.title("Dicipline")
    save_plot(save_path, "10")
    plt.clf()
    plt.close()

    # What is your position?
    position_cnt = dataset.groupby([qs_dict[f"Q42"]])[RESP_ID].count()  # ALL DATA
    colors = ["#6A7CA7", "#DDD7D1", "#160F4C", "#616F76", "#C4C7B4", "#97AFB3"]  # Graduate, Other, PI, Postdoc, RA, Undergrad
    sns.set_palette(sns.color_palette(colors))
    sns.set_style("whitegrid")
    plt.pie(x=position_cnt, autopct="%.1f%%", explode=[0.05] * len(position_cnt.keys().tolist()),
            labels=position_cnt.keys(), pctdistance=0.5)
    plt.title("Position")
    save_plot(save_path, "11")
    plt.clf()
    plt.close()

    return


def manage_processing(data_path, save_path):
    """
    Manage any processing of the survey responses.
    :param data_path: path to the survey csv file
    :param save_path: path to save all plots in
    """
    dataset = parse_responses(data_path)  # parse the data
    if dataset:
        plot_data(dataset, save_path)


if __name__ == "__main__":
    manage_processing(data_path=r"...\results-survey475519.csv",
                    save_path=r"...\testing_framework")
