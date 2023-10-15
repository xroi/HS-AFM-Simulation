#!/bin/csh -f
set IMP_FOLDER=/cs/labs/ravehb/ravehb/imp/fast_conda/
set IMP=$IMP_FOLDER/setup_environment.sh
set SCR=/cs/labs/ravehb/ravehb/imp/repository/modules/npctransport/utility/add_cylinder.py
source /cs/labs/ravehb/ravehb/External/venv_imp2023_v2/bin/activate.csh

if ($#argv != 3) then
    echo "Syntax: $0 <rmf_input_path> <pb_input_path> <rmf_output_path>"
    exit 0
endif

$IMP python3 $SCR --input_rmf $1 --ref_output $2 --output_rmf $3