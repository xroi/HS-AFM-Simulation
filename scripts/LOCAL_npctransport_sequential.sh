#!/bin/csh -f
#SBATCH --mem=2g
#SBATCH --time=7-0
#SBATCH --mail-type=ALL

if ($#argv != 5) then
    echo "Syntax: $0 <from-start(1/0)> <start> <step> <output_folder_path> <config_path>"
    exit 0
endif

# Setup Environment
set IMP_FOLDER=/cs/labs/ravehb/ravehb/imp/fast_conda/
set IMP=$IMP_FOLDER/setup_environment.sh
source /cs/labs/ravehb/ravehb/External/venv_imp2023_v2/bin/activate.csh

set OUTPUT_PATH=$4/
set CONFIG_PATH=$5
set seed=`od -An -N4 -td4 /dev/random`

mkdir -p $OUTPUT_PATH
echo output path is $OUTPUT_PATH

if (`echo "$1==1" | bc`) then
    echo Initilising new simulation...
    set i=$3
    $IMP $IMP_FOLDER/bin/fg_simulation --configuration $CONFIG_PATH --output $OUTPUT_PATH$i.pb --short_init_factor 0.5 --short_sim_factor 1.00 --conformations $OUTPUT_PATH$i.movie.rmf --final_conformations $OUTPUT_PATH$i.pb.final.rmf --random_seed $seed
endif

#ignore this idk scripting format
set i=(`echo "$2 + $3" | bc`)
set j=$2

while (1)
    $IMP $IMP_FOLDER/bin/fg_simulation --output $OUTPUT_PATH$i.pb --conformations $OUTPUT_PATH$i.movie.rmf --final_conformations $OUTPUT_PATH$i.pb.final.rmf --restart $OUTPUT_PATH$j.pb
    echo cur: $i using:$j
    @ i+=$3
    @ j+=$3
end
