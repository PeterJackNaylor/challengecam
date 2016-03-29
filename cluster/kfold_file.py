





import sys
import os
import random
import numpy as np





if __name__ ==  "__main__":

	input_1 = int(sys.argv[1])
	#### Kfold 

	input_2 = sys.argv[2]
	#### folder address

	input_3 = sys.argv[3]
	#### version name

	folder_Normal = os.path.join(input_2,'Normal')
	folder_Tumor  = os.path.join(input_2,'Tumor')
	file_name = os.path.join(input_2, input_3, 'kfold.txt')

	nber_Normal = 160
	nber_Tumor  =  110

	x_normal = np.array(range(nber_Normal))
	x_tumor = np.array(range(nber_Tumor))

	nber_Tumor  = nber_Tumor / input_1
	nber_Normal = nber_Normal/ input_1

	random.shuffle(x_normal)
	random.shuffle(x_tumor)

	file = open(file_name, "w")

	for i in range(input_1):
		file.write("Fold "+str(i)+"\n")
		file.write("train\n")
		file.write("Normal\n")
		
		if i!=input_1-1:
			val_x_normal = x_normal[i*nber_Normal:(i+1)*nber_Normal]
		else:
			val_x_normal = x_normal[i*nber_Normal::]
		
		file.write(str(list(set(val_x_normal)))+"\n") 
		file.write("Tumor\n")
		
		if i!=input_1-1:
			val_x_tumor = x_tumor[i*nber_Tumor:(i+1)*nber_Tumor]
		else:
			val_x_tumor = x_tumor[i*nber_Tumor::]

		file.write(str(list(set(val_x_tumor)))+"\n")
		file.write("test:\n")
		file.write("Normal\n")
		file.write(str(list(set(x_normal) - set(val_x_normal)))+"\n")
		file.write("Tumor\n")
		file.write(str(list(set(x_tumor) - set(val_x_tumor)))+"\n")
