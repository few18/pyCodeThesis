#!/bin/bash
# specifies the shell you're using

# Starting the line with "#PBS" means it's a queue system control
# directive.

# -N flag lets you name the job.
#PBS -N SardineAnchovyTest

# -m flag makes system send you an when job begins (b), ends (e) or
# is aborted (a). Can use a, b and e or none of them or
# any combination of them.
#PBS -m abe

# -q flag specifies which queue to use.
#PBS -q standard
# #PBS -l ncpus=8
# #PBS -l mem=4gb

# Change the home directory to approriate sub-directory
cd ${HOME}/DataCluster
# run the script
python SardineAnchovy.py
