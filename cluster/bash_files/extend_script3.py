import os


input_folder = 'train_results2_svm'

str1 = 'SOURCE_FOLDER=/cbio/donnees/twalter/src/challengecam/cluster'
str2 = 'source /cbio/donnees/twalter/src/challengecam/cluster/bash_files/twalter_code_profile'

str3 = 'python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Tumor --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/%s/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/%s/whole_slide' % (input_folder, input_folder) 

fp = open('./extended_glue_train3.sh', 'w')
slides = os.listdir('/share/data40T_v2/challengecam_results/results_on_train')
fp.write(str1 + '\n')
fp.write(str2 + '\n')

tumor_slides = filter(lambda x: x[:len('Tumor')] == 'Tumor', slides)
for ts in tumor_slides:
    print ts
    print str3
    print 
    fp.write(str3 + ' --slide_name %s\n' % ts)

str3 = 'python $SOURCE_FOLDER/slide_generator.py --orig_folder=/share/data40T/pnaylor/Cam16/Normal --prob_map_folder=/share/data40T_v2/challengecam_results/probmap/%s/crops/ --output_folder=/share/data40T_v2/challengecam_results/probmap/%s/whole_slide' % (input_folder, input_folder) 
normal_slides = filter(lambda x: x[:len('Normal')] == 'Normal', slides)
for ns in normal_slides:
    print ns
    print str3
    print 
    fp.write(str3 + ' --slide_name %s\n' % ns)


fp.close()
 
