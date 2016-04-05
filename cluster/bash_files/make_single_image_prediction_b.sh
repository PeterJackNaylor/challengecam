#!/bin/bash

#$ -cwd # execute the job from the current directory
#$ -S /bin/bash #set bash environment
#$ -N SlideClass # name of the job as it will appear in qstat -f
#$ -o /cbio/donnees/twalter/PBS/OUT
#$ -e /cbio/donnees/twalter/PBS/ERR
#$ -l h_vmem=4G

## others optional options
## #$ -V  Pass all current environment variables to the job.
## #$ -q bath # Tell the system which queue to use



### #$ -t 1-160 # les valeures successives que va prendre $SGE_TASK_ID
### #$ -tc 160 # nbre de job qui peuvent fonctionner en parallèle ensemble



SOURCE_FOLDER=/cbio/donnees/twalter/src/challengecam/cluster

source /cbio/donnees/twalter/src/challengecam/cluster/bash_files/twalter_code_profile

python $SOURCE_FOLDER/image_prediction.py --classifier_name=/share/data40T_v2/challengecam_results/training/best_classifier.pickle --feature_folder=/share/data40T_v2/challengecam_results/Pred_data_set/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set --subsample_factor=16 --slide_number=097



