#!/bin/bash

#$ -cwd # execute the job from the current directory
#$ -S /bin/bash #set bash environment
#$ -N Training_RF # name of the job as it will appear in qstat -f
#$ -o /cbio/donnees/pnaylor/PBS/OUT
#$ -e /cbio/donnees/pnaylor/PBS/ERR
#$ -l nodes=1:ppn=4,h_vmem=4G


## others optional options
## #$ -V  Pass all current environment variables to the job.
## #$ -q bath # Tell the system which queue to use

##$ -t 1-160 # les valeures successives que va prendre $SGE_TASK_ID
##$ -tc 160 # nbre de job qui peuvent fonctionner en parallèle ensemble

CAM16=/share/data40T_v2/challengecam_results/train/
KFOLD=/share/data40T_v2/challengecam_results/training/kfold.txt
VERSION=default
OUTPUT=/share/data40T_v2/challengecam_results/training/
spe_tag=__
source $HOME/.bash_profile

FILE=settings_for_machine_learning.txt # fichier csv (delimiter=' ') où la premiere colonne est la valeur de $PBS_ARRAYID, la seconde est le nom du programme, et les autres les différents paramètres à faire passer au code python
FIELD1=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f2) # la partie gauche est pour chopper la ligne numéro $PBS_ARRAYID
FIELD2=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f3) # la partie droite est pour chopper la valeur qui est dans la colonne voulue 
FIELD3=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f4) # sachant que le séparateur est l'espace
FIELD4=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f5)
FIELD5=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f6)
FIELD6=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f7)


python $CAM16/scripts/challengecam/cluster/machine_learning.py --source $CAM16 --kfold_file $KFOLD --fold $FIELD1 --n_samples $FIELD2 --version $VERSION --n_tree $FIELD4 --m_try $FIELD5 --bootstrap $FIELD6 --save 0 --output $OUTPUT --n_jobs 4