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

python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=1
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=2
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=3
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=4
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=5
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=6
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=7
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=8
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=9
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=10
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=11
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=12
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=13
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=14
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=15
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=16
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=17
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=18
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=19
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=20
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=21
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=22
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=23
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=24
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=25
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=26
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=27
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=28
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=29
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=30
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=31
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=32
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=33
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=34
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=35
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=36
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=37
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=38
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=39
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=40
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=41
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=42
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=43
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=44
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=45
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=46
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=47
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=48
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=49
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=50
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=51
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=52
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=53
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=54
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=55
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=56
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=57
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=58
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=59
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=60
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=61
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=62
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=63
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=64
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=65
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=66
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=67
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=68
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=69
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=70
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=71
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=72
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=73
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=74
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=75
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=76
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=77
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=78
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=79
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=80
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=81
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=82
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=83
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=84
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=85
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=86
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=87
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=88
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=89
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=90
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=91
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=92
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=93
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=94
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=95
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=96
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=97
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=98
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=99
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=100
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=101
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=102
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=103
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=104
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=105
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=106
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=107
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=108
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=109
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=110
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=111
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=112
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=113
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=114
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=115
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=116
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=117
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=118
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=119
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=120
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=121
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=122
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=123
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=124
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=125
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=126
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=127
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=128
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=129
python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Test --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slide_number=130
