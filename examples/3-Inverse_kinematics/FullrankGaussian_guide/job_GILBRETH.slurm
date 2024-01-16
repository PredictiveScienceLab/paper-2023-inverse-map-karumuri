#!/bin/sh -l
#SBATCH -A lilly-comp-g
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=72:00:00
#SBATCH --job-name IK
#SBATCH --mail-user=skarumur@purdue.edu
#SBATCH --mail-type=ALL

# These are comment lines
# Load the Anaconda distribution of Python, which comes
# pre-bundled with many of the popular scientific computing tools like
# numpy, scipy, pandas, scikit-learn, etc.
module --force purge
module load anaconda/5.1.0-py36
source activate py36

# Place python script 
python -u metrics_data.py > metrics_data_outputs.out

