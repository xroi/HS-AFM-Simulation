#!/bin/bash

if [ $# != 1 ]; then
  echo "Syntax: $0 <dir_path>"
  exit 1
fi

#tree -fQFi --sort=size ${1}/ | grep \.pb\"$ | tail +2 | cut -d '.' -f 1

BIGGEST_PB_N=$(/cs/labs/ravehb/roi.eliasian/NpcTransportExperiment/HS-AFM-Simulation/scripts/LOCAL_get_biggest_pb_n.sh ${1})
BIGGEST_PB="${BIGGEST_PB_N}.pb"
echo ${BIGGEST_PB}