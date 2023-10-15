#!/bin/csh -f
#SBATCH --mem=2g
#SBATCH --time=7-0
#SBATCH --mail-user=roi.eliasian@mail.huji.ac.il
#SBATCH --mail-type=ALL

source ../venv_new/bin/activate.csh
python3 ../src/main.py @args.txt
