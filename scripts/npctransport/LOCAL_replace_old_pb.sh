#!/bin/bash

set IMP_FOLDER=/cs/labs/ravehb/ravehb/imp/fast_conda/
set IMP=$IMP_FOLDER/setup_environment.sh
source /cs/labs/ravehb/ravehb/External/venv_imp2023_v2/bin/activate.csh

if [ $# != 1 ]; then
  echo "Syntax: $0 <path>"
  exit 1
fi

$IMP python3 /cs/labs/ravehb/roi.eliasian/NpcTransportExperiment/HS-AFM-Simulation/scripts/npctransport/edit_old_pb.py ${1} TEMP_replace_old_pb.pb
cp TEMP_replace_old_pb.pb ${1}
rm TEMP_replace_old_pb.pb
