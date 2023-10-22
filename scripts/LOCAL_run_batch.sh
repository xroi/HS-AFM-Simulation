#!/bin/csh -f
#SBATCH --mem=2g
#SBATCH --time=7-0
#SBATCH --mail-user=roi.eliasian@mail.huji.ac.il
#SBATCH --mail-type=ALL
#SBATCH --array=0-10

if ($#argv != 2) then
    echo "Syntax: $0 <input_folder_path> <output_folder_path> <config_path>"
    exit 0
endif

set OUTPUT_PATH=$1/
set INPUT_PATH=$2/
set CONFIG_PATH=$3

# Note: Input path should be a folder, that has folder named 0-array_max
# (generated with LOCAL_npctransport_sequential_batch.sh)

mkdir -p $OUTPUT_PATH
/cs/labs/ravehb/roi.eliasian/NpcTransportExperiment/HS-AFM-Simulation/scripts/LOCAL_run.sh $INPUT_PATH/$SLURM_ARRAY_TASK_ID $OUTPUT_PATH/$SLURM_ARRAY_TASK_ID $CONFIG_PATH

