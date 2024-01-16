#!/bin/sh -l
#SBATCH -A lilly-comp-g
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=240:00:00
#SBATCH --job-name INN
#SBATCH --mail-user=skarumur@purdue.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=64G

# These are comment lines
# Load the Anaconda distribution of Python, which comes
# pre-bundled with many of the popular scientific computing tools like
# numpy, scipy, pandas, scikit-learn, etc.
module --force purge
module load anaconda/2020.11-py38
source activate myenv_py38


### Place your python script run command here
### Example :

python -u AVI_CNF.py $1 $2 $3 $4 $5 $6 > outputs/${7}

