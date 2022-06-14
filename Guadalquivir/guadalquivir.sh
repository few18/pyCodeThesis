#!/bin/bash
# specifies the shell you're using

# Starting the line with "#PBS" means it's a queue system control
# directive.

# -N flag lets you name the job.
#PBS -N Maizuru

# -m flag makes system send you an when job begins (b), ends (e) or
# is aborted (a). Can use a, b and e or none of them or
# any combination of them.
#PBS -m a

# -q flag specifies which queue to use.
#PBS -q parallel8
#PBS -l ncpus=8
#PBS -l mem=4gb

#PBS -t 0-104

# Change the home directory to approriate sub-directory
cd ${HOME}/CodeCluster/guadalquivir
# run the script
python -i guadalquivir.py $(($PBS_ARRAYID))
