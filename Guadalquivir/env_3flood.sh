#!/bin/bash
# specifies the shell you're using

# Starting the line with "#PBS" means it's a queue system control
# directive.

# -N flag lets you name the job.
#PBS -N env_3flood

# -m flag makes system send you an when job begins (b), ends (e) or
# is aborted (a). Can use a, b and e or none of them or
# any combination of them.
#PBS -m a

# -q flag specifies which queue to use.
#PBS -q medium

#PBS -t 0-189

# Change the home directory to approriate sub-directory
cd ${HOME}/guada_final/env_3flood
# run the script
python -i env_3flood.py $(($PBS_ARRAYID))
