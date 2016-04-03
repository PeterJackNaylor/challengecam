import os
import pandas as pd
import pdb
from optparse import OptionParser
import matplotlib.pylab as plt


if __name__ ==  "__main__":

	parser = OptionParser()
	parser.add_option("-s", "--source", dest="folder_source",
	                  help="Where to find result folder", metavar="FILE")
	parser.add_option("--threshold", dest="threshold",
					   help="threshold for the prediction cut", metavar="float")
	parser.add_option("--plot", dest="plot",default="1",
					   help="Ploting disabled by default", metavar="float")

	(options, args) = parser.parse_args()
	print "source file: |"+options.folder_source
	print "Threshold  : |"+options.threshold
	print "Plotting   : |"+options.plot

	names = ["Test_%03i" %i  for i in range(1,130+1)]

	names_for_csv = []
	confidences_for_csv = []

	for name in names:
		file_name = os.path.join(options.folder_source, "Evaluation2",name+".csv")
		confidence = pd.DataFrame.from_csv(file_name, header=0, index_col=None).max()['Confidence']
		names_for_csv.append(name+".tif")
		confidences_for_csv.append(confidence)

	Evaluation1 = pd.DataFrame({"File_name":names_for_csv, "Probability":confidences_for_csv})
	Evaluation1.to_csv(os.path.join(options.folder_source, "Evaluation1","Evaluation1"+".csv"))

	if options.plot == '0':
		fig, ax0 = plt.subplots(ncols=1, figsize=(8, 8))

		ax0.hist(Evaluation1.Probability, 20, normed=1, histtype='stepfilled', facecolor='r', alpha=0.75)
		ax0.set_title('Probability distribution')
		plt.show()