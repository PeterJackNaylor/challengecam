




import pdb
import os
import numpy as np
from optparse import OptionParser
import cPickle as pkl



if __name__ ==  "__main__":

	parser = OptionParser()
	parser.add_option("-t", "--text", dest="text",
	                  help="the test input file", metavar="FILE")
	parser.add_option("-s", "--saving", dest="saving",
	                  help="Where the outputs of the jobs are", metavar="FILE")
	parser.add_option("-o", "--output", dest="output",
	                  help="Output folder", metavar="FILE")
	(options, args) = parser.parse_args()


	folder_save = options.saving
	folder_out  = options.output
	
	output = open(os.path.join(folder_out,"output_of_checking_score_folder.py"),'a')

	f = open(options.text, 'rb')
	content = f.readlines()
	for para in content:
		paras = para.split(' ')
		line = paras[0]
		FIELD1 = paras[1]
		FIELD2 = paras[2]
		FIELD3 = paras[3]
		FIELD4 = paras[4]
		FIELD5 = paras[5]
		FIELD6 = paras[6].split('\n')[0]

		file_name = "score_fold_"+FIELD1+"_tree_"+FIELD4+"_mtry_"+FIELD5+"_boot_"+FIELD6+"_nsample_"+FIELD2+".pickle"

		try:
			path = os.path.join(folder_save,file_name)
			D  = pkl.load(open(path,'rb'))
		except:
			print "line "+line+" not here..."
			output.write("line "+line+" not here... \n")
	output.close()
	f.close()