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

	source_folder = os.path.join(options.folder_source, "classifier_*.pickle")

	data = pd.DataFrame(columns=('Fold','N_tree','m_try','n_bootstrap','TP','TN','FP','FN'))

	files = glob.glob(source_folder)
	i=0
	for fn in files:
		D = pkl.load(open(fn,'rb'))
		para = fn.split('.')[0].split('/')[-1].split('_')[1::]
		Fold = para[1]
		N_tree = para[3]
		m_try = para[5]
		n_bootstrap = para[7]
		values = [Fold, N_tree, m_try, n_bootstrap, D['TP'], D['TN'], D['FP'], D['FN']]
		data.loc[i] = values		
		i += 1
	pdb.set_trace()

