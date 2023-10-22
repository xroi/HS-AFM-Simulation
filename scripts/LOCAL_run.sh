#!/bin/csh -f
#SBATCH --mem=2g
#SBATCH --time=7-0
#SBATCH --mail-user=roi.eliasian@mail.huji.ac.il
#SBATCH --mail-type=ALL

if ($#argv != 2) then
    echo "Syntax: $0 <input_folder_path> <output_prefix> <args_path>"
    exit 0
endif

# Make sure you don't supply the --input-path and ----output-path-prefix arguments in the config file.

source /cs/labs/ravehb/roi.eliasian/NpcTransportExperiment/HS-AFM-Simulation/venv_new/bin/activate.csh

set INPUT_PATH=$1/
set OUTPUT_PATH=$2
set ARGS_PATH=$3
mkdir -p $OUTPUT_PATH
echo output path is $OUTPUT_PATH

python3 /cs/labs/ravehb/roi.eliasian/NpcTransportExperiment/HS-AFM-Simulation/src/main.py --input-path $INPUT_PATH --output-path-prefix $OUTPUT_PATH @$1

