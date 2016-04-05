#!/bin/bash

#$ -cwd # execute the job from the current directory
#$ -S /bin/bash #set bash environment
#$ -N Normal # name of the job as it will appear in qstat -f
#$ -o logs
#$ -l h_vmem=4G
#$ -M peter.naylor@mines-paristech.fr # set email adress to notify once a job changes states as specified by -m
#$ -m ae # a- send mail when job is aborted by batch system ; b- send mail when begins execution; e- send mail when job ends; n- do not send mail

## others optional options
## #$ -V  Pass all current environment variables to the job.
## #$ -q bath # Tell the system which queue to use



#$ -t 1-160 # les valeures successives que va prendre $SGE_TASK_ID
#$ -tc 160 # nbre de job qui peuvent fonctionner en parallèle ensemble

CAM16=/share/data40T/pnaylor/Cam16
spe_tag=__
source $HOME/.bash_profile

FILE=settings_for_pred_base.txt # fichier csv (delimiter=' ') où la premiere colonne est la valeur de $PBS_ARRAYID, la seconde est le nom du programme, et les autres les différents paramètres à faire passer au code python
FIELD1=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f2) # la partie gauche est pour chopper la ligne numéro $PBS_ARRAYID
FIELD2=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f3) # la partie droite est pour chopper la valeur qui est dans la colonne voulue 
FIELD3=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f4) # sachant que le séparateur est l'espace
FIELD4=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f5)
FIELD5=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f6)
FIELD6=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f7)
FIELD7=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f8)


python $CAM16/scripts/challengecam/cluster/Pred_Data_set.py -s $CAM16 -t $FIELD1 -n $FIELD2 -x $FIELD3 -y $FIELD4 -w $FIELD5 --height $FIELD6 -r $FIELD7 -o $CAM16/Pred_data_set
