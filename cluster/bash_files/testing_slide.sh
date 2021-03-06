#!/bin/bash

#$ -cwd # execute the job from the current directory
#$ -S /bin/bash #set bash environment
#$ -N testing_files # name of the job as it will appear in qstat -f
#$ -o logs
#$ -M peter.naylor@mines-paristech.fr # set email adress to notify once a job changes states as specified by -m
#$ -m ae # a- send mail when job is aborted by batch system ; b- send mail when begins execution; e- send mail when job ends; n- do not send mail

## others optional options
## #$ -V  Pass all current environment variables to the job.
## #$ -q bath # Tell the system which queue to use




source $HOME/.bash_profile

#FILE=param_list.txt # fichier csv (delimiter=' ') où la premiere colonne est la valeur de $PBS_ARRAYID, la seconde est le nom du programme, et les autres les différents paramètres à faire passer au code python
#FIELD1=$(grep "^$SGE_TASK_ID " $FILE | cut -d' ' -f2) # la partie gauche est pour chopper la ligne numéro $PBS_ARRAYID
#FIELD2=$(grep "^$SGE_TASK_ID " $FILE | cut -d' ' -f3) # la partie droite est pour chopper la valeur qui est dans la colonne voulue 
#FIELD3=$(grep "^$SGE_TASK_ID " $FILE | cut -d' ' -f4) # sachant que le séparateur est l'espace
#FIELD4=$(grep "^$SGE_TASK_ID " $FILE | cut -d' ' -f5)
#FIELD5=$(grep "^$SGE_TASK_ID " $FILE | cut -d' ' -f6)
#FIELD6=$(grep "^$SGE_TASK_ID " $FILE | cut -d' ' -f7)

python /share/data40T/pnaylor/Cam16/scripts/challengecam/cluster/testing_files.py -s /share/data40T/pnaylor/Cam16/ -o /share/data40T/pnaylor/Cam16/
