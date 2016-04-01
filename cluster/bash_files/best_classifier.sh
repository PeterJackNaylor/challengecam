#!/bin/bash

#$ -cwd # execute the job from the current directory
#$ -S /bin/bash #set bash environment
#$ -N Training_RF # name of the job as it will appear in qstat -f
#$ -o /cbio/donnees/pnaylor/PBS/OUT
#$ -e /cbio/donnees/pnaylor/PBS/ERR
#$ -l h_vmem=4G
#$ -pe orte 2  

## others optional options
## #$ -V  Pass all current environment variables to the job.
## #$ -q bath # Tell the system which queue to use

##$ -t 1-4 # les valeures successives que va prendre $SGE_TASK_ID
##$ -tc 160 # nbre de job qui peuvent fonctionner en parallèle ensemble


PYTHON_FILE=/share/data40T/pnaylor/Cam16/scripts/challengecam/cluster/bash_files/best_classifier.py
SOURCE=/share/data40T_v2/challengecam_results/train/
n_samples=5000
version=default
n_tree=1000
m_try=50
bootstrap=10000
saving=0  ### it is going to be saved
OUTPUT=/share/data40T_v2/challengecam_results/training/
JOBS=1
C=1.0
model=forest


python $PYTHON_FILE --source $SOURCE --n_samples $n_samples --version $version --n_tree $n_tree --m_try $m_try --bootstrap $bootstrap --save $saving --output $OUTPUT --n_jobs $JOBS -c $C --model $model
