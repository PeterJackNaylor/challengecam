import os

#filename = 'make_single_image_prediction_template.sh'
#filename = 'glue_image_template.sh'
#filename = 'make_single_image_train_template.sh'
#filename = 'subsampling_template.sh'
filename = 'glue_image_train2_template.sh'
#filename = 'glue_image_template_svm.sh'
#filename = 'subsampling_template_train.sh'

fp = open(filename, 'r')
temp = filter(lambda x: x!='\n' and len(x) > 0, fp.readlines())
fp.close()

print temp

last_line = temp[-1]
print last_line

fp = open('./extended_%s' % filename, 'w')
for pre in temp[:-1]:
	fp.write(pre + '\n')

for i in range(1,13):
	fp.write(last_line[:-4] + str(i) + '\n')

fp.close()
 
