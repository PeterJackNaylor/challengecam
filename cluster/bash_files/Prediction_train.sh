#!/bin/bash

#$ -cwd # execute the job from the current directory
#$ -S /bin/bash #set bash environment
#$ -N Prediction # name of the job as it will appear in qstat -f
#$ -o /cbio/donnees/twalter/PBS/OUT
#$ -e /cbio/donnees/twalter/PBS/ERR
#$ -l h_vmem=4G

## others optional options
## #$ -V  Pass all current environment variables to the job.
## #$ -q bath # Tell the system which queue to use



### #$ -t 1-160 # les valeures successives que va prendre $SGE_TASK_ID
### #$ -tc 160 # nbre de job qui peuvent fonctionner en parallèle ensemble

CAM16=/share/data40T/pnaylor/Cam16
OUTPUT_FOLDER=/share/data40T_v2/challengecam_results/results_on_train
spe_tag=__
#source $HOME/.bash_profile

source /cbio/donnees/twalter/src/challengecam/cluster/bash_files/twalter_code_profile

FILE=settings_for_train_base4.txt # fichier csv (delimiter=' ') où la premiere colonne est la valeur de $PBS_ARRAYID, la seconde est le nom du programme, et les autres les différents paramètres à faire passer au code python
FIELD1=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f2) # la partie gauche est pour chopper la ligne numéro $PBS_ARRAYID
FIELD2=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f3) # la partie droite est pour chopper la valeur qui est dans la colonne voulue 
FIELD3=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f4) # sachant que le séparateur est l'espace
FIELD4=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f5)
FIELD5=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f6)
FIELD6=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f7)
FIELD7=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f8)


echo python /share/apps/user_apps/challengecam/cluster/Pred_Data_set.py -s $CAM16 --type $FIELD1 --number $FIELD2 -x $FIELD3 -y $FIELD4 --width $FIELD5 --height $FIELD6 --resolution $FIELD7 --output $OUTPUT_FOLDER

python /cbio/donnees/twalter/src/challengecam/cluster/Pred_Data_set.py -s $CAM16 --type $FIELD1 --number $FIELD2 -x $FIELD3 -y $FIELD4 --width $FIELD5 --height $FIELD6 --resolution $FIELD7 --output $OUTPUT_FOLDER

