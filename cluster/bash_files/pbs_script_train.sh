#!/bin/bash

#$ -S /bin/bash #set bash environment
#$ -N Normal # name of the job as it will appear in qstat -f
#$ -o /cbio/donnees/twalter/PBS/OUT
#$ -e /cbio/donnees/twalter/PBS/ERR
#$ -l h_vmem=4G

##   $ -t 1-160 # les valeures successives que va prendre $SGE_TASK_ID
##   $ -tc 160 # nbre de job qui peuvent fonctionner en parallèle ensemble

source /share/apps/user_apps/challengecam/cluster/bash_files/peter_profile

python /cbio/donnees/twalter/src/challengecam/cluster/Data_set.py --input_folder /share/data40T/pnaylor/Cam16 --output_folder /share/data40T_v2/challengecam_results/train --type Normal --id $SGE_TASK_ID --nb_samples 10000 --nb_images 16
