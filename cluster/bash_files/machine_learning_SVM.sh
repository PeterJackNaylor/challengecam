#!/bin/bash

#$ -cwd # execute the job from the current directory
#$ -S /bin/bash #set bash environment
#$ -N Training_RF # name of the job as it will appear in qstat -f
#$ -o /cbio/donnees/pnaylor/PBS/OUT
#$ -e /cbio/donnees/pnaylor/PBS/ERR
#$ -l h_vmem=4G
##$ -pe orte 2  

## others optional options
## #$ -V  Pass all current environment variables to the job.
## #$ -q bath # Tell the system which queue to use

##$ -t 1-4 # les valeures successives que va prendre $SGE_TASK_ID
##$ -tc 160 # nbre de job qui peuvent fonctionner en parallèle ensemble

CAM16=/share/data40T_v2/challengecam_results/train/
KFOLD=/share/data40T_v2/challengecam_results/training/kfold.txt
OUTPUT=/share/data40T_v2/challengecam_results/SVM/
spe_tag=__
PYTHON_FILE=/share/data40T/pnaylor/Cam16/scripts/challengecam/cluster/machine_learning_SVM.py

KGROUPS=30
KSAMPLES=1
KERNEL=rbf

source $HOME/.bash_profile

FILE=/share/data40T_v2/challengecam_results/settings_for_machine_learning_SVM.txt # fichier csv (delimiter=' ') où la premiere colonne est la valeur de $PBS_ARRAYID, la seconde est le nom du programme, et les autres les différents paramètres à faire passer au code python
FIELD1=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f2) # la partie gauche est pour chopper la ligne numéro $PBS_ARRAYID
FIELD2=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f3) # la partie droite est pour chopper la valeur qui est dans la colonne voulue 
FIELD3=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f4) # sachant que le séparateur est l'espace
FIELD4=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f5)
FIELD5=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f6)

python $PYTHON_FILE --source $CAM16 --kfold_file $KFOLD --fold $FIELD1 --n_samples $FIELD2 --version $FIELD3 --norm1 0 --penalty $FIELD4 --save 1  --output $OUTPUT  --kmean_k $KGROUPS --kmean_n $KSAMPLES --kernel $KERNEL --gamma $FIELD5