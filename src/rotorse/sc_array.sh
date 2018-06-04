#!/bin/bash
#SBATCH --time=0:20:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1 --mem=32G
#SBATCH -J "test_ranges" # job name

# need to run twice (comment out array that isn't running)

#SBATCH --array=0-999 # job array size

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

echo ${SLURM_ARRAY_TASK_ID}

python test_FAST_DEMs.py ${SLURM_ARRAY_TASK_ID}
