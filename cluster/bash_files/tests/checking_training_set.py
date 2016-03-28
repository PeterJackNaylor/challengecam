
### this scripts will tell you which files are missing
### and will also tell the number of variables and the 
### number of lines in total.

import pdb
import os
import numpy as np
from optparse import OptionParser
import cPickle as pkl


def f(file, output, dic_ref ,seuil = 10000):
	try:
		if ".pickle" in file:

			im_pickle = open(file,  'r')
			dic = pkl.load(im_pickle)

			val = np.sort(dic.values())
			key = np.sort(dic.keys())

			val_ref = np.sort(dic_ref.values())
			key_ref = np.sort(dic_ref.keys())


			if not((val == val_ref).all() and (key == key_ref).all()):
				#pdb.set_trace()
				output.write("Problems with "+file+"\n")
		
		else:

			y = np.load(file)
			TUP = y.shape
			if TUP[0] < seuil:
				output.write("File "+file+" has less then "+str(seuil)+"lines \n" )

	except:
		#pdb.set_trace()
		output.write("Problems with "+file+"\n")






if __name__ ==  "__main__":

	parser = OptionParser()
	parser.add_option("-s", "--source", dest="folder_source",
	                  help="Where to find Tumor files", metavar="FILE")
	parser.add_option("-o", "--output", dest="output",
	                  help="Output folder", metavar="FILE")
	parser.add_option("-e","--seuil",dest="seuil",default="10000",
			  help="threshold for number of lines to print",metavar="INT")
	(options, args) = parser.parse_args()

	n_lines = 0
	n_ones = 0
	variables = {}

	of = open("output_checking_training_set.txt", "a")
	for pref in ["Tumor","Normal"]:

		if "Tumor"==pref:
			n_range = 110
		else:
			n_range = 160

		for i in range(1,n_range+1):
			
			digit = "%03d" % (i,)
			slide = pref + "_" + digit
			ad = os.path.join(options.folder_source, slide)

			suff = [".npy", ".pickle", "_y_.npy"]
			
			for s in suff:

				file = os.path.join(ad, slide + s)
				if variables == {}:
					if s == ".pickle":
						variables = pkl.load(open(file,'r'))
				f(file, of, variables, int(options.seuil))
				try:
					if "_y_" in file:
						y = np.load(file)
						s = np.sum(y)
						s_max = np.max(y)
						if s_max != 0:
							s = s / s_max
							n_ones += s
						n_lines += y.shape[0]
				except:
					#pdb.set_trace()
					#print file + "\n"
					pppppp = 1	
	of.write("Their is %d lines \n" % n_lines)
	of.write("Their is %d ones \n" % n_ones)
