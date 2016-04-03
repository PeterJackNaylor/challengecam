#!/bin/bash

#$ -cwd # execute the job from the current directory
#$ -S /bin/bash #set bash environment
#$ -N csv_making # name of the job as it will appear in qstat -f
#$ -o /share/data40T/pnaylor/PBS/OUT_CSV
#$ -e /share/data40T/pnaylor/PBS/ERR_CSV
#$ -l h_vmem=4G
##$ -pe orte 2  

## others optional options
## #$ -V  Pass all current environment variables to the job.
## #$ -q bath # Tell the system which queue to use

#$ -t 1-130 # les valeures successives que va prendre $SGE_TASK_ID
#$ -tc 130 # nbre de job qui peuvent fonctionner en parallèle ensemble

RES=2
DISK_SIZE=5
SUBSAMPLING=16
SIGMA=5

spe_tag=__
PYTHON_FILE=/share/data40T/pnaylor/Cam16/scripts/challengecam/cluster/fonctions_for_csv.py
source /share/data40T/pnaylor/.bash_profile

FILE=/share/data40T_v2/challengecam_results/inputs_to_csv.txt

FIELD0=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f2) ## file probability map
FIELD1=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f3) ## slide name
FIELD2=$(grep "$spe_tag$SGE_TASK_ID$spe_tag " $FILE | cut -d' ' -f4) ## csv file name


python $PYTHON_FILE -f $FIELD0 -r $RES --disk_size $DISK_SIZE --plot 1 --sigma $SIGMA --subsampling $SUBSAMPLING --slide_name $FIELD1 --output $FIELD2
