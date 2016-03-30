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
p_s = [10, 15, 20, 30, 50]
n_samples = 2000
version = "default"
number_of_trees = [500, 1000]
n_bootstrap = [ 500, 1000, 2000, 5000 ]



if __name__ ==  "__main__":
	parser = OptionParser()
	parser.add_option("-o","--output",default=".",dest="output",
					  help="output folder",metavar="folder")

	(options, args) = parser.parse_args()


	f = open(os.path.join(options.output,"settings_for_machine_learning.txt"), "a")
	line = 0
	for n_boot in n_bootstrap:
		for p in p_s:
			for t in number_of_trees:
				for i in range(number_of_folds):
					f.write("__"+str(line) + "__ "+str(i)+" "+str(n_samples)+" "+version+" "
						+ str(t) + " " +str(p) + " " + str(n_boot) +"\n")
					line += 1
	f.close()
