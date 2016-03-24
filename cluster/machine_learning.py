# -*- coding: cp1252 -*-
"""
Description: 
Challenge CAMELYON16.
Script for pixel classification. 

Authors:  Vaïa Machairas, Etienne Decencière, Peter Naylor, Thomas Walter.

Creation date: 2016-02-24
"""

import sys

sys.path.append('../RandomForest_Peter/')

import pdb
import os

from getpass import getuser
import smilPython as sp
import numpy as np

import folder_functions as ff
import segmentation_by_classification as sc
from evaluation import my_metrics
import segm_db_access as sdba
import cPickle as pickle
from find_ROI import subsample,from_list_string_to_list_Tumor

from forest_Peter import *

from optparse import OptionParser

### metrics to use
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score

##---------------------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------------------

############### Before doing anything you have 
############### to get a fold file, so launch that script
###############

if __name__ ==  "__main__":

	from cluster_parameters import *

	### inputs and folder reads
	parser = OptionParser()
	parser.add_option("-s", "--source", dest="folder_source",
	                  help="Where to find Tumor files", metavar="FILE")
	parser.add_option("-k", "--fold", dest="k_folds",
	                  help="Number of the fold in the cross validation", metavar="int")
	parser.add_option("-n","--n_samples", dest="n_samples",default=1000,
					  help="Number of samples taking from one image",metavar="int")
	parser.add_option("-ve","--version",dest="version",
					  help="Version", dest="version name")
	parser.add_option("-t","--n_tree",dest="n_tree",
					  help="Number of trees for the random Forest",metavar="int")
	parser.add_option("-p","--m_try",dest="m_try",
					  help="Number of selected features at each tree",metavar="int")
	parser.add_option("-b","--bootstrap",dest="n_bootstrap",
					  help="Number of selected instances at each tree",metavar="int")

	(options, args) = parser.parse_args()


	version_para = { 'n_sub': options.n_samples }

	
	saving_location = os.path.join(options.folder_source, options.version)
	
	kfold_file = os.path.join(saving_location,'kfold.txt')
	f = open(kfold_file,'r')
	all_para = f.read().split('\n')
	
	i = options.k_folds ## fold number

	Normal_slides_train = from_list_string_to_list_Tumor(all_para[ i*11 + 3 ],all_para[ i*11 + 2 ])
	Tumor_slides_train  = from_list_string_to_list_Tumor(all_para[ i*11 + 5 ],all_para[ i*11 + 4 ])
	Normal_slides_test  = from_list_string_to_list_Tumor(all_para[ i*11 + 8 ],all_para[ i*11 + 7 ])
	Tumor_slides_test   = from_list_string_to_list_Tumor(all_para[ i*11 + 10],all_para[ i*11 + 9 ])

	training_names = Normal_slides_train + Tumor_slides_train

	sample_name = training_names[0]
	image_sauv_name_pickle = os.path.join(saving_location ,sample_name, sample_name  + ".pickle")
	image_sauv_name_npy    = os.path.join(saving_location ,sample_name, sample_name  + ".npy")
	image_sauv_name_y_npy  = os.path.join(saving_location ,sample_name, sample_name  + "_y_.npy")

	X_temp = np.load( image_sauv_name_npy )
	Y_temp = np.load( image_sauv_name_y_npy ).ravel()
	index = subsample(Y_temp, options.version, version_para)

	X_temp = X_temp[index,:]
	Y_temp = Y_temp[index]

	n_train = len(training_names) * options.n_samples
	p_train = X_temp.shape[1]


	X_train = np.zeros(shape=(n_train, p_train))
	X_train[0:options.n_samples,:] = X_temp

	Y_train = np.zeros(n_train)
	Y_train[0:options.n_samples] = Y_temp

	i = 1

	for sample_name in training_names[1::]:

		image_sauv_name_pickle = os.path.join(saving_location ,sample_name, sample_name  + ".pickle")
		image_sauv_name_npy    = os.path.join(saving_location ,sample_name, sample_name  + ".npy")
		image_sauv_name_y_npy  = os.path.join(saving_location ,sample_name, sample_name  + "_y_.npy")

		X_temp = np.load( image_sauv_name_npy )
		Y_temp = np.load( image_sauv_name_y_npy ).ravel()
		index = subsample(Y_temp, options.version, version_para)

		X_temp = X_temp[index,:]
		Y_temp = Y_temp[index]

		X_train[ i * options.n_samples : (i+1) * options.n_samples ] = X_temp
		Y_train[ i * options.n_samples : (i+1) * options.n_samples ] = Y_temp


	myforest = PeterRandomForestClassifier(n_estimators = options.n_tree, max_features = options.mtry,
										   max_depth = None, n_bootstrap = options.n_bootstrap ) ## penser a changer bootstrap

	myforest.fit(X_train,Y_train)

	for sample_name in Normal_slides_train+Tumor_slides_test:

		image_sauv_name_npy    = os.path.join(saving_location ,sample_name, sample_name  + ".npy")
		image_sauv_name_y_npy  = os.path.join(saving_location ,sample_name, sample_name  + "_y_.npy")

		X_pred = np.load( image_sauv_name_npy )
		Y_pred = np.load( image_sauv_name_y_npy ).ravel()

		Y_hat = myforest.predict(X_pred)
		Y_hat_prob = myforest.predict_proba(X_pred)

		auc = roc_auc_score(Y_pred, Y_hat_prob)
		acc = accuracy_score(Y_pred, Y_hat)
		f1_ = f1_score(Y_pred, Y_hat)
		rec = recall_score(Y_pred, Y_hat)
		pre = precision_score(Y_pred, Y_hat)

		image_sauv_name_score = os.path.join(saving_location ,sample_name, "score_fold_"+str(options.k_folds)+ ".pickle")
		
		im_pickle = open(image_sauv_name_score,  'w')

		D = {'auc':auc, 'accuracy':acc, 'f1':f1_, 'recall':rec, 'precision':pre}
		pickle.dump(image_sauv_name_score,D)
