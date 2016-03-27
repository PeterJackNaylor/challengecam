
### this scripts will tell you which files are missing
### and will also tell the number of variables and the 
### number of lines in total.


import os
import numpy as np
from optparse import OptionParser
import cPickles as pkl


def f(file, output, dic_ref ):
	try:
		if ".pickle" in file:

			im_pickle = open(file,  'r')
			dic = pkl.load(im_pickle)

			val = np.sort(dic.values())
			key = np.sort(dic.keys())

			val_ref = np.sort(dic_ref.values())
			key_ref = np.sort(dic_ref.keys())


			if not(val == val_ref and key == key_ref):
				output.write("Problems with "+file+"\n")
		
		else:

			y = np.load(file)
			TUP = y.shape

	except:

		output.write("Problems with "+file+"\n")






if __name__ ==  "__main__":

	parser = OptionParser()
	parser.add_option("-s", "--source", dest="folder_source",
	                  help="Where to find Tumor files", metavar="FILE")
	parser.add_option("-o", "--output", dest="output",
	                  help="Output folder", metavar="FILE")
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

		for i in range(n_range):
			
			digit = "%03d" % (i,)
			slide = pref + "_" + digit
			ad = os.path.join(options.folder_source, slide)

			suff = [".npy", ".pickle", "_y_.npy"]
			
			for s in suff:

				file = os.path.join(ad, slide + s)
				f(file, of, variables)
				try:
					if "_y_" in file:
						y = np.load(file)
						s = np.sum(y)
						s_max = np.max(y)
						if s_max != 0:
							s = s / s_max
							n_ones + = s
						n_lines += y.shape[0]
	of.write("Their is %d lines \n") % n_lines
	of.write("Their is %d ones \n") % n_ones