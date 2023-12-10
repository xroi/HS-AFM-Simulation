#!/bin/csh -f
set IMP_FOLDER=/cs/labs/ravehb/ravehb/imp/fast_conda/
set IMP=$IMP_FOLDER/setup_environment.sh
set SCR=/cs/labs/ravehb/roi.eliasian/NpcTransportExperiment/HS-AFM-Simulation/scripts/NPC_rmf_to_pdb_v3.py
source /cs/labs/ravehb/ravehb/External/venv_imp2023_v2/bin/activate.csh

if ($#argv != 3) then
    echo "Syntax: $0 <rmf_input_path> <pb_input_path> <pdb_output_path>"
    exit 0
endif

$IMP python3 $SCR --input_rmf $1 --ref_output $2 --output_pdb $3