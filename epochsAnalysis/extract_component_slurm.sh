#!/bin/bash
#SBATCH --partition=octopus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=alex.lepauvre@ae.mpg.de
#SBATCH --time 30:00:00
#SBATCH --output=slurm-%A_%a.out
#SBATCH --job-name=epochs_analysis
analysis_parameters_file=""
participant_id=""
while [ $# -gt 0 ]; do
  case "$1" in
    --analysis_parameters_file=*)
      analysis_parameters_file="${1#*=}"
      ;;
    --participant_id=*)
      participant_id="${1#*=}"
      ;;
    *)
      printf "***************************\n"
      printf "* Error: Invalid argument: ${1}*\n"
      printf "***************************\n"
      exit 1
  esac
  shift
  echo participant_id
done

module purge; module load Anaconda3/2020.11; source /hpc/shared/EasyBuild/apps/Anaconda3/2020.11/bin/activate; conda activate /hpc/users/$USER/.conda/envs/mne_ecog02

export PYTHONPATH=$PYTHONPATH:/hpc/users/alexander.lepauvre/sw/github/ECoG

python extract_component_master.py --AnalysisParametersFile "${analysis_parameters_file}" --subjectID "${participant_id}"
