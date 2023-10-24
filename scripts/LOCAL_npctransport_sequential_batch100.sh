#!/bin/csh -f
#SBATCH --mem=2g
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --array=0-100

if ($#argv != 2) then
    echo "Syntax: $0 <output_folder_path> <config_path>"
    exit 0
endif

set OUTPUT_PATH=$1/
set CONFIG_PATH=$2


mkdir -p $OUTPUT_PATH
/cs/labs/ravehb/roi.eliasian/NpcTransportExperiment/HS-AFM-Simulation/scripts/LOCAL_npctransport_sequential.sh 1 1000 1000 $OUTPUT_PATH/$SLURM_ARRAY_TASK_ID $CONFIG_PATH