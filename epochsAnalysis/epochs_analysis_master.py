import argparse


def single_participant_evoked():
    # Get the parameters:
    parser = argparse.ArgumentParser(
        description="Implements analysis of EDFs for experiment1")
    parser.add_argument('--AnalysisParametersFile', type=str, default=None,
                        help="Config file for analysis parameters (file name + path)")
    parser.add_argument('--subjectID', type=str, default=None,
                        help="Directory and name of the subject info json")
    args = parser.parse_args()

    # Create the parameters file
    analysis_parameters =


if __name__ == "__main__":
    single_participant_evoked()
