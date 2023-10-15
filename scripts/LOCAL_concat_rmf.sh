#!/bin/csh -f
# Setup Environment
set IMP_FOLDER=/cs/labs/ravehb/ravehb/imp/fast_conda/
set IMP=$IMP_FOLDER/setup_environment.sh
source /cs/labs/ravehb/ravehb/External/venv_imp2023_v2/bin/activate.csh

if ($#argv != 5) then
    echo "Syntax: $0 <folder_input_path> <rmf_output_path> <start_time> <end_time> <interval>"
    exit 0
endif

$IMP python3 concat_rmf.py --input-path $1 --output-path $2 --start-time-ns $3 --end-time-ns $4 --interval-ns $5
