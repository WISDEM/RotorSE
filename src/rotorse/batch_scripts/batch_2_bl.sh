#!/bin/bash

#SBATCH --time=2-00:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=4G   # memory per CPU core
#SBATCH -J "blade length study"   # job name
#SBATCH --mail-user=bryceingersoll@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-20     # job array of size 21

echo ${SLURM_ARRAY_TASK_ID}

python BL_2_study.py ${SLURM_ARRAY_TASK_ID}
