############ This files creates a setting file for 
# machine_learning.sh and machine_learning.py
# it has to be a text file seperated by ' ' and with 6 fields
# field0 is the line
# field1 is the fold number
# field2 is the number of samples taking from one of the images
# field3 is the version name
# field4 is the number of trees
# field5 is the m_try for random forest




from optparse import OptionParser

number_of_folds = 10
p_s = [10, 15, 20, 50, 300]
n_samples = 1000
version = "version_0"
number_of_trees = [500,800,1000]

if __name__ ==  "__main__":

	f = open("settings_for_machine_learning.txt", "a")
	line = 0
	for p in p_s:
		for t in number_of_trees:
			for i in range(number_of_folds):
				f.write(str(line) + "** "+str(i)+" "+str(n_samples)+" "+version+" "
					+ str(t) + " " +str(p) +"\n")
				line += 1