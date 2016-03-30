from optparse import OptionParser
import os

import pandas as pd


if __name__ ==  "__main__":

	#from cluster_parameters import *

	### inputs and folder reads
	parser = OptionParser()
	
	parser.add_option("-s", "--source", dest="folder_source",
	                  help="Where to find score files", metavar="FILE")
	
	parser.add_option("-o","--output",default=".",dest="output",
					  help="output folder",metavar="folder")

	(options, args) = parser.parse_args()


	for fn in 