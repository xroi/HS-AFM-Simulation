#!/bin/csh -f
# Setup Environment
set IMP_FOLDER=/cs/labs/ravehb/ravehb/imp/fast_conda/
set IMP=$IMP_FOLDER/setup_environment.sh
source /cs/labs/ravehb/ravehb/External/venv_imp2023_v2/bin/activate.csh

#$IMP python3 ../scripts/concat_rmf.py $argv:q
$IMP python3 ../scripts/concat_rmf.py --input-path ../../08-10-2023-NTR/ --output-path new.rmf --start-time-ns 1000 --end-time-ns 10000 --interval-ns 1000
