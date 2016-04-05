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
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set_svm/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set_svm/whole_slide --slide_number=001
