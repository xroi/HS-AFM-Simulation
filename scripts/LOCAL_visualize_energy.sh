#!/bin/csh -f

if ($#argv != 1) then
    echo "Syntax: $0 <pb_file_path>"
    exit 0
endif

# Setup Environment
set IMP_FOLDER=/cs/labs/ravehb/ravehb/imp/fast_conda/
set IMP=$IMP_FOLDER/setup_environment.sh
source /cs/labs/ravehb/ravehb/External/venv_imp2023_v2/bin/activate.csh
set SCRIPT=`readlink -f "$0"`
set DIRNAME=`dirname "$SCRIPT"`

$IMP python3 $DIRNAME/other_stats.py --input-path $1 --output-path .