#!/bin/bash

if [ $# != 2 ]; then
  echo "Syntax: $0 <folder_path> <amount>"
  exit 1
fi



for (( i=1 ; i<=${2} ; i++ ));
do
  BIGGEST_PB = $(/cs/labs/ravehb/roi.eliasian/NpcTransportExperiment/HS-AFM-Simulation/scripts/LOCAL_get_biggest_pb.sh $OUTPUT_PATH/${ID})
  /cs/labs/ravehb/roi.eliasian/NpcTransportExperiment/HS-AFM-Simulation/scripts/npctransport/LOCAL_replace_old_pb.sh $BIGGEST_PB
done