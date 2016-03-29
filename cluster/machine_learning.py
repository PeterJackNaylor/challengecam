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

### This has to be launch from the folder cluster..
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

from forest_Peter import PeterRandomForestClassifier

from optparse import OptionParser

### metrics to use
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,confusion_matrix

##---------------------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------------------

############### Before doing anything you have 
############### to get a fold file, so launch that script
###############

def Score(Y_pred,Y_hat):
	TP = 0
	FP = 0
	TN = 0
	FN = 0
	for i in range(len(Y_hat)): 
		val_Y_pred_i = Y_pred[i]
		val_Y_hat_i  = Y_hat[i]

		if val_Y_pred_i==val_Y_hat_i and val_Y_pred_i==1:
			TP += 1 
		elif val_Y_pred_i==1 and val_Y_pred_i!=val_Y_hat_i:
			FP += 1 
		elif val_Y_pred_i==val_Y_hat_i and val_Y_hat_i==0:
			TN += 1
		elif val_Y_pred_i==0 and val_Y_pred_i!=val_Y_hat_i:
			FN += 1
	return(TP, FP, TN, FN)


if __name__ ==  "__main__":

	from cluster_parameters import *

	### inputs and folder reads
	parser = OptionParser()
	parser.add_option("-s", "--source", dest="folder_source",
	                  help="Where to find Tumor files", metavar="FILE")
	parser.add_option("-f","--kfold_file",dest="kfold_file",
					  help="where to find the kfold file",metavar="FILE")
	parser.add_option("-k", "--fold", dest="k_folds",
	                  help="Number of the fold in the cross validation", metavar="int")
	parser.add_option("-n","--n_samples", dest="n_samples",default=1000,
					  help="Number of samples taking from one image",metavar="int")
	parser.add_option("-v","--version",dest="version",default="default",
					  help="sub sample Version",metavar="string")
	parser.add_option("-t","--n_tree",dest="n_tree",
					  help="Number of trees for the random Forest",metavar="int")
	parser.add_option("-p","--m_try",dest="m_try",
					  help="Number of selected features at each tree",metavar="int")
	parser.add_option("-b","--bootstrap",dest="n_bootstrap",
					  help="Number of selected instances at each tree",metavar="int")
	parser.add_option("--save", dest="save",default="1",
					  help="booleen to save, 0: True 1: False", metavar="bool")
	parser.add_option("-o","--output",default=".",
					  help="output folder",metavar="folder")

	(options, args) = parser.parse_args()


	version_para = { 'n_sub': int(options.n_samples) }

	data_location   = options.folder_source
	saving_location = options.output
	
	kfold_file = options.kfold_file
	f = open(kfold_file,'r')
	all_para = f.read().split('\n')
	
	i = int(options.k_folds) ## fold number

	start_time = time.time()

	Normal_slides_test   = from_list_string_to_list_Tumor(all_para[ i*11 + 3 ],all_para[ i*11 + 2 ])
	Tumor_slides_test    = from_list_string_to_list_Tumor(all_para[ i*11 + 5 ],all_para[ i*11 + 4 ])
	Normal_slides_train  = from_list_string_to_list_Tumor(all_para[ i*11 + 8 ],all_para[ i*11 + 7 ])
	Tumor_slides_train   = from_list_string_to_list_Tumor(all_para[ i*11 + 10],all_para[ i*11 + 9 ])

	training_names = Normal_slides_train + Tumor_slides_train
	sample_name = training_names[0]
	image_sauv_name_pickle = os.path.join(data_location ,sample_name, sample_name  + ".pickle")
	image_sauv_name_npy    = os.path.join(data_location ,sample_name, sample_name  + ".npy")
	image_sauv_name_y_npy  = os.path.join(data_location ,sample_name, sample_name  + "_y_.npy")

	X_temp = np.load( image_sauv_name_npy )
	Y_temp = np.load( image_sauv_name_y_npy ).ravel()
	index = subsample(Y_temp, options.version, version_para)

	X_temp = X_temp[index,:]
	Y_temp = Y_temp[index]

	n_train = len(training_names) * int(options.n_samples)
	p_train = X_temp.shape[1]


	X_train = np.zeros(shape=(n_train, p_train))
	X_train[0:int(options.n_samples),:] = X_temp

	Y_train = np.zeros(n_train)
	Y_train[0:int(options.n_samples)] = Y_temp

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

			X_train[ i * int(options.n_samples) : (i+1) * int(options.n_samples),: ] = X_temp[:,:]
			Y_train[ i * int(options.n_samples) : (i+1) * int(options.n_samples) ] = Y_temp[:]
		except:
			print sample_name+" was not possible \n"

	index_to_keep_X = np.where(X_train.any(axis=1))[0]

	X_train = X_train[index_to_keep_X,:]

	Y_train = Y_train[index_to_keep_X]
	
	diff_time = time.time() - start_time
	print 'Setting up X_train:'
	print '\t%02i:%02i:%02i' % (diff_time/3600, (diff_time%3600)/60, diff_time%60)

	start_time = time.time()
	
	myforest = PeterRandomForestClassifier(n_estimators = int(options.n_tree), max_features = int(options.m_try),
										   max_depth = None, n_bootstrap = int(options.n_bootstrap) ) ## penser a changer bootstrap
	myforest.fit(X_train,Y_train)
	if int(options.save) == 0:
		file_name = "classifier_fold_"+options.k_folds+"_tree_"+options.n_tree+"_mtry_"+options.m_try+"_boot_"+options.n_bootstrap+".pickle"
		pickle_file = open( os.path.join(saving_location, file_name) , "wb")

	diff_time = time.time() - start_time
	print 'Training:'
	print '\t%02i:%02i:%02i' % (diff_time/3600, (diff_time%3600)/60, diff_time%60)
	
	start_time = time.time()

	D = {'TP':0,'FP':0,'TN':0,'FN':0}

	for sample_name in Normal_slides_test+Tumor_slides_test:
		try:
			image_sauv_name_npy    = os.path.join(data_location ,sample_name, sample_name  + ".npy")
			image_sauv_name_y_npy  = os.path.join(data_location ,sample_name, sample_name  + "_y_.npy")

			X_pred = np.load( image_sauv_name_npy )
			Y_pred = np.load( image_sauv_name_y_npy ).ravel()

			Y_hat = myforest.predict(X_pred)
			Y_hat_prob = myforest.predict_proba(X_pred)
			TP, FP, TN, FN = Score(Y_pred,Y_hat)
			D['TP'] += TP
			D['FP'] += FP
			D['TN'] += TN
			D['FN'] += FN
		except:
			print sample_name+" was not possible \n"
	file_name = "score_fold_"+options.k_folds+"_tree_"+options.n_tree+"_mtry_"+options.m_try+"_boot_"+options.n_bootstrap+".pickle"
	image_sauv_name_score = os.path.join(saving_location , file_name)


	im_pickle = open(image_sauv_name_score,  'w')

	pickle.dump(image_sauv_name_score, D)
	
	diff_time = time.time() - start_time
	print 'Prediction time:'
	print '\t%02i:%02i:%02i' % (diff_time/3600, (diff_time%3600)/60, diff_time%60)
