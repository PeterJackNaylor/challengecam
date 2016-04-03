## this file creates the settings for the csv output

import os
from optparse import OptionParser


if __name__ ==  "__main__":
	parser = OptionParser()
	parser.add_option("-o","--output",default=".",dest="output",
					  help="output folder",metavar="folder")
	parser.add_option("--ProbMapFolder",default=".",dest="ProbMap",
					  help="ProbMap folder",metavar="folder")
	parser.add_option("--slideFolder",default=".",dest="SlideFolder",
					  help="Slide folder",metavar="folder")
	parser.add_option("--CSVFolder",default=".",dest="CSVFolder",
					  help="Slide folder",metavar="folder")
	(options, args) = parser.parse_args()


	f = open(os.path.join(options.output,"inputs_to_csv.txt"), "a")
	line = 1
	n_range = 130
	for i in range(1,n_range+1):
		id_test = '%03i' % i
		ProbMap = os.path.join(options.ProbMap, "whole_probmap_Test_"+id_test+".png")
		SlideName = os.path.join(options.SlideFolder, "Test_"+id_test+".tif")
		CsvName = os.path.join(options.CSVFolder, "Test_"+id_test+".csv")
		
		f.write("__"+str(line) + "__ "+ProbMap+" "+SlideName+" "+CsvName+"\n")
		line += 1
	f.close()
