# -*- coding: cp1252 -*-
"""
Description: 
Challenge CAMELYON16.
Script for pixel classification. 

Authors:  Vaïa Machairas, Etienne Decencière, Peter Naylor, Thomas Walter.

Creation date: 2016-02-24
"""

import sys
import time
from sklearn import svm

import pdb
import os

from getpass import getuser
import numpy as np

import folder_functions as ff
import segmentation_by_classification as sc
from evaluation import my_metrics
import segm_db_access as sdba
import cPickle as pickle
from find_ROI import subsample,from_list_string_to_list_Tumor
from sklearn.preprocessing import StandardScaler
from optparse import OptionParser

### metrics to use
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,confusion_matrix
from forest_Peter import PeterRandomForestClassifier

##---------------------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------------------

if __name__ ==  "__main__":

	#from cluster_parameters import *

	### inputs and folder reads
	parser = OptionParser()
	parser.add_option("-s", "--source", dest="folder_source",
	                  help="Where to find Tumor files", metavar="FILE")
	parser.add_option("-n","--n_samples", dest="n_samples",default="1000",
					  help="Number of samples taking from one image",metavar="int")
	parser.add_option("-v","--version",dest="version",default="default",
					  help="sub sample Version",metavar="string")
	parser.add_option("--norm1",dest="norm1",default="0",
					  help="Normalization scheme slide by slide",metavar="int")
	parser.add_option("-t","--n_tree",dest="n_tree",
					  help="Number of trees for the random Forest",metavar="int")
	parser.add_option("-p","--m_try",dest="m_try",
					  help="Number of selected features at each tree",metavar="int")
	parser.add_option("-b","--bootstrap",dest="n_bootstrap",
					  help="Number of selected instances at each tree",metavar="int")
	parser.add_option("-c","--penalty",dest="c",
					  help="value of the penalty in the SVM",metavar="int")
	parser.add_option("--save", dest="save",default="1",
					  help="booleen to save, 0: True 1: False", metavar="bool")
	parser.add_option("-o","--output",default=".",dest="output",
					  help="output folder",metavar="folder")
	parser.add_option("--n_jobs",default="1",dest="n_jobs",
					  help="number of jobs to pass to randomforest",metavar="int")
	parser.add_option("--model",dest="model",
					  help="Model to be used",metavar="str")
	parser.add_option("--kernel",dest="kernel",
					  help="kernel to be used",metavar="kernel")
	parser.add_option("--kmean_k",default="0",dest="kmean_k",
					  help="number of clusters of the k mean",metavar="int > 0 ")
	parser.add_option("--kmean_n",dest="kmean_n",
					  help="downsampling number with the k mean algorithm",metavar="int")
	parser.add_option("--gamma",dest="gamma",
					  help="value of the hyper parameter for the gaussian kernel",metavar="int")
	(options, args) = parser.parse_args()

	print "n_samples:   |"+options.n_samples
	print "subsampling: |"+options.version
	print "source file: |"+options.folder_source
	print "C:           |"+options.c
	print "saving:      |"+options.save
	print "output folde:|"+options.output 
	print "n_tree:      |"+options.n_tree
	print "m_try:       |"+options.m_try
	print "bootstrap:   |"+options.n_bootstrap
	print "n_jobs:      |"+options.n_jobs
	print "norm1:       |"+options.norm1
	print "model:       |"+options.model
	print "kernel:      |"+options.kernel
	print "gamma        |"+options.gamma
	print "kmeans k    :|"+options.kmean_k
	print "kmean downsa:|"+options.kmean_n
	version_para = { 'n_sub': int(options.n_samples) }

	para_kmean = { 'n_sub':int(options.kmean_n), 'k':int(options.kmean_k)}

	data_location   = options.folder_source
	saving_location = options.output
	
	start_time = time.time()

	training_names = ['Tumor_'+'%03i'%i for i in range(1,110+1)]
	training_names+= ['Normal_'+'%03i'%i for i in range(1,160+1)] 

	sample_name = training_names[0]
	image_sauv_name_pickle = os.path.join(data_location ,sample_name, sample_name  + ".pickle")
	image_sauv_name_npy    = os.path.join(data_location ,sample_name, sample_name  + ".npy")
	image_sauv_name_y_npy  = os.path.join(data_location ,sample_name, sample_name  + "_y_.npy")

	X_temp = np.load( image_sauv_name_npy )
	Y_temp = np.load( image_sauv_name_y_npy ).ravel()
	index = subsample(Y_temp, options.version, version_para)

	X_temp = X_temp[index,:]
	Y_temp = Y_temp[index]
	if int(options.kmean_k) != 0:
		para_kmean['X'] = X_temp
		index_kmean = subsample(Y_temp, 'kmeans' , para_kmean)
		X_temp = X_temp[index_kmean,:]
		Y_temp = Y_temp[index_kmean]

	if int(options.kmean_k) !=0:
		step = int(options.kmean_k) * int(options.kmean_n)
	else:
		step = int(options.n_samples)
	n_train = len(training_names) * step
	p_train = X_temp.shape[1]


	X_train = np.zeros(shape=(n_train, p_train))
	n_temp = X_temp.shape[0]
	X_train[0:n_temp,:] = X_temp

	Y_train = np.zeros(n_train)
	Y_train[0:n_temp] = Y_temp

	i = 0
	for sample_name in training_names[1::]:
		try:
			i += 1
			image_sauv_name_pickle = os.path.join(data_location ,sample_name, sample_name  + ".pickle")
			image_sauv_name_npy    = os.path.join(data_location ,sample_name, sample_name  + ".npy")
			image_sauv_name_y_npy  = os.path.join(data_location ,sample_name, sample_name  + "_y_.npy")

			X_temp = np.load( image_sauv_name_npy )
			Y_temp = np.load( image_sauv_name_y_npy ).ravel()
			index = subsample(Y_temp, options.version, version_para)

			X_temp = X_temp[index,:]
			Y_temp = Y_temp[index]
			if int(options.norm1) == 0:
				X_temp = StandardScaler().fit_transform(X_temp)
			if int(options.kmean_k) != 0:
				para_kmean['X'] = X_temp
				index_kmean = subsample(Y_temp, 'kmeans' , para_kmean)
				X_temp = X_temp[index_kmean,:]
				Y_temp = Y_temp[index_kmean]
			n_temp = X_temp.shape[0]
			X_train[ i * step : i * step + n_temp,: ] = X_temp[:,:]
			Y_train[ i * step : i * step + n_temp] = Y_temp[:]
		except:
			print sample_name+" was not possible"

	index_to_keep_X = np.where(X_train.any(axis=1))[0]

	X_train = X_train[index_to_keep_X,:]

	Y_train = Y_train[index_to_keep_X]

	Y_train[Y_train>0]=1

	diff_time = time.time() - start_time

	print 'Setting up X_train:'
	print '\t%02i:%02i:%02i' % (diff_time/3600, (diff_time%3600)/60, diff_time%60)
	print 'With dim X_train = %d, %d' %X_train.shape
	print 'With n_ones = %d' %len(np.where(Y_train != 0)[0])
	start_time = time.time()
	if options.model == 'svm':
		clf = svm.SVC(C=float(options.c),kernel=options.kernel,degree=3, gamma=float(options.gamma),coef0=0.0, shrinking=True, probability=True,tol=0.001, cache_size=200, class_weight='balanced',verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
		clf.fit(X_train,Y_train)
	elif options.model == "forest":
		clf = PeterRandomForestClassifier(n_estimators = int(options.n_tree), max_features = int(options.m_try),
										max_depth = None, class_weight="balanced_subsample",
										n_bootstrap = int(options.n_bootstrap) ,
										n_jobs= int(options.n_jobs))
		clf.fit(X_train,Y_train)
	if int(options.save) == 0:

		file_name = "best_classifier_SVM"+".pickle"
		pickle_file = open( os.path.join(saving_location, file_name) , "wb")
		pickle.dump(clf, pickle_file)
	diff_time = time.time() - start_time
	print 'Training:'
	print '\t%02i:%02i:%02i' % (diff_time/3600, (diff_time%3600)/60, diff_time%60)
