from optparse import OptionParser
import os
import glob
import pandas as pd
import cPickle as pkl
import pdb
if __name__ ==  "__main__":

	#from cluster_parameters import *

	### inputs and folder reads
	parser = OptionParser()
	
	parser.add_option("-s", "--source", dest="folder_source",
	                  help="Where to find score files", metavar="FILE")
	
	parser.add_option("-o","--output",default=".",dest="output",
					  help="output folder",metavar="folder")

	(options, args) = parser.parse_args()

	source_folder = os.path.join(options.folder_source, "score_*.pickle")
#	output_folder 
	data = pd.DataFrame(columns=('Fold','C','gamma','n_samples', 'TP', 'TN', 'FP', 'FN'))

	files = glob.glob(source_folder)
	i=0
	for fn in files:
		try:
			D = pkl.load(open(fn,'rb'))
			para = fn.split('.')[0].split('/')[-1].split('_')[1::]
			Fold = para[1]
			C = para[3]
			n_samples = para[5]
			gamma = para[7]
			values = [Fold, C, gamma, n_samples, D['TP'], D['TN'], D['FP'], D['FN']]
			data.loc[i] = values		
			i += 1
		except:
			print fn.split('/')[-1] + ' is corrupted'

	data['Precision'] = data['TP'] / (data['TP'] + data['FP'])
	data['Recall']    = data['TP'] / (data['TP'] + data['FN'])
	data['F1']        = 2 * data['Precision'] * data['Recall'] / (data['Precision'] + data['Recall'])
	data['Accuracy']  = (data['TP'] + data['TN']) / (data['TP'] + data['FP'] + data['TN'] + data['FN'])

	groups_mean = data.dropna().groupby(['C','gamma', 'n_samples']).mean()
	groups_std  = data.dropna().groupby(['C','gamma', 'n_samples']).std()

	best_F1=groups_mean['F1'].argmax()
	print "best set of parameters: ", best_F1
	print "with score: "
	print groups_mean.ix[best_F1]
	print "With variations over parameters:"
	print groups_std.ix[best_F1]


	pdb.set_trace()

