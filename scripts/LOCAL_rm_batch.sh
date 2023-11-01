#!/bin/bash
#SBATCH --mem=2g
#SBATCH --time=2-0

if [ $# != 2 ]; then
  echo "Syntax: $0 <dir_path> <amount>"
  exit 1
fi

for (( i=1 ; i<=${2} ; i++ ));
do
   rm -rf ${1}/4{i}
done