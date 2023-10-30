#!/bin/bash
#SBATCH --mem=2g
#SBATCH --time=7-0
#SBATCH --array=1-25
#SBATCH --killable
#SBATCH --requeue
#SBATCH -c2

SECOND_JOB_OFFSET=25

if [ $# != 3 ]; then
  echo "Syntax: $0 <output_folder_path> <config_path> <step>"
  exit 1
fi

OUTPUT_PATH=${1}/
CONFIG_PATH=${2}
STEP=${3}
mkdir -p $OUTPUT_PATH

declare -a IDs=()
IDs+=(${SLURM_ARRAY_TASK_ID})
IDs+=($((SLURM_ARRAY_TASK_ID + SECOND_JOB_OFFSET)))
echo IDs: ${IDs[@]}

echo "Running jobs"
declare -a PIDs=()
for ID in ${IDs[@]}; do
  /cs/labs/ravehb/roi.eliasian/NpcTransportExperiment/HS-AFM-Simulation/scripts/npctransport/LOCAL_npctransport_sequential.sh 1 $STEP $STEP $OUTPUT_PATH/${ID} $CONFIG_PATH &
  PID=$!
  PIDs+=($PID)
  echo $PID submitted, workid $ID
done

echo "Waiting for all jobs to finish"
for PID in "${PIDs[@]}"; do
  wait "$PID"
  echo "$PID finished"
done