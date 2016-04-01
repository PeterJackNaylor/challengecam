############ This files creates a setting file for 
# machine_learning.sh and machine_learning.py
# it has to be a text file seperated by ' ' and with 6 fields
# field0 is the line
# field1 is the fold number
# field2 is the number of samples taking from one of the images
# field3 is the version name
# field4 is the number of trees
# field5 is the m_try for random forest



import os
from optparse import OptionParser

number_of_folds = 10
n_samples = [2000, 5000]
version = "default"
C = [ '0.001', '0.01', '0.1', '1', '10', '100', '1000', '0.002', '0.02', '0.2', '2', '20', '200', '2000', '0.005', '0.05', '0.5', '5', '50', '500', '5000']
gamma = ['0.001', '0.01', '0.1', '1', '10', '100', '1000']

if __name__ ==  "__main__":
	parser = OptionParser()
	parser.add_option("-o","--output",default=".",dest="output",
					  help="output folder",metavar="folder")

	(options, args) = parser.parse_args()


	f = open(os.path.join(options.output,"settings_for_machine_learning_SVM.txt"), "a")
	line = 1
	for n_sampl in n_samples:
		for small_c in C:
			for small_g in gamma:
				for i in range(number_of_folds):
					f.write("__"+str(line) + "__ "+str(i)+" "+str(n_sampl)+" "+version+" "
						+ str(small_c) + " " +str(small_g) +"\n")
					line += 1
	f.close()
