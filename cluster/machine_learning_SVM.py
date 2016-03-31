# -*- coding: cp1252 -*-
"""
Description: 
Challenge CAMELYON16.
Script for pixel classification. 

Authors:  Vaïa Machairas, Etienne Decencière, Peter Naylor, Thomas Walter.

Creation date: 2016-02-24

%run /share/data40T/pnaylor/Cam16/scripts/challengecam/cluster/machine_learning_SVM.py --source /share/data40T_v2/challengecam_results/train/ --kfold_file /share/data40T_v2/challengecam_results/training/kfold.txt --fold $FIELD1 --n_samples $FIELD2 --version $FIELD3 --norm1 0 --penalty 10 --save 1 --output /share/data40T_v2/challengecam_results/training/ --kmean_k 50 --kmean_n 100

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
		else:
			print val_Y_pred_i, val_Y_hat_i
	return(TP, FP, TN, FN)


if __name__ ==  "__main__":

	#from cluster_parameters import *

	### inputs and folder reads
	parser = OptionParser()
	parser.add_option("-s", "--source", dest="folder_source",
	                  help="Where to find Tumor files", metavar="FILE")

	parser.add_option("-f","--kfold_file",dest="kfold_file",
					  help="where to find the kfold file",metavar="FILE")
	parser.add_option("-k", "--fold", dest="k_folds",
	                  help="Number of the fold in the cross validation", metavar="int")
	parser.add_option("-n","--n_samples", dest="n_samples",default="1000",
					  help="Number of samples taking from one image",metavar="int")
	parser.add_option("-v","--version",dest="version",default="default",
					  help="sub sample Version",metavar="string")
	parser.add_option("--norm1",dest="norm1",default="0",
					  help="Normalization scheme slide by slide",metavar="int")
	parser.add_option("-c","--penalty",dest="c",
					  help="value of the penalty in the SVM",metavar="int")
	parser.add_option("--save", dest="save",default="1",
					  help="booleen to save, 0: True 1: False", metavar="bool")
	parser.add_option("-o","--output",default=".",dest="output",
					  help="output folder",metavar="folder")
	parser.add_option("--kmean_k",default="0",dest="kmean_k",
					  help="number of clusters of the k mean",metavar="int > 0 ")
	parser.add_option("--kmean_n",dest="kmean_n",
					  help="downsampling number with the k mean algorithm",metavar="int")
	

	(options, args) = parser.parse_args()
	print "source file: |"+options.folder_source
	print "kfold file:  |"+options.kfold_file
	print "fold number: |"+options.k_folds
	print "n_samples:   |"+options.n_samples
	print "version:     |"+options.version
	print "norm1:       |"+options.norm1
	print "C:           |"+options.c
	print "saving:      |"+options.save
	print "output folde:|"+options.output
	print "kmeans k    :|"+options.kmean_k
	print "kmean downsa:|"+options.kmean_n

	version_para = { 'n_sub': int(options.n_samples) }

	if int(options.kmean_k) != 0:
		para_kmean = { 'n_sub':int(options.kmean_n), 'k':int(options.kmean_k)} 

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
	if int(options.norm1) == 0:
		X_temp = StandardScaler().fit_transform(X_temp)
	X_temp = X_temp[index,:]
	Y_temp = Y_temp[index]
	if int(options.kmean_k) != 0:
		para_kmean['X'] = X_temp
		index_kmean = subsample(Y_temp, 'kmeans' , para_kmean)
		X_temp = X_temp[index_kmean,:]
		Y_temp = Y_temp[index_kmean]
	
	step = int(options.kmean_k) * int(options.kmean_n)
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
			pdb.set_trace()
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
	
	clf = svm.SVC(C=float(options.c), kernel='linear',
				  degree=3, gamma='auto',
				  coef0=0.0, shrinking=True, probability=True,
				  tol=0.001, cache_size=200, class_weight='balanced',
				  verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)

	clf.fit(X_train,Y_train)
	if int(options.save) == 0:
		file_name = "classifierSVM_fold_"+options.k_folds+"_C_"+options.c+"_nsample_"+options.n_samples+".pickle"
		pickle_file = open( os.path.join(saving_location, file_name) , "wb")
		pickle.dump(clf, pickle_file)
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
			if int(options.norm1) == 0:
				X_pred = StandardScaler().fit_transform(X_pred)

			Y_pred[Y_pred>0] = 1

			Y_hat = clf.predict(X_pred)
			#Y_hat_prob = clf.predict_proba(X_pred)
			TP, FP, TN, FN = Score(Y_pred,Y_hat)
			D['TP'] += TP
			D['FP'] += FP
			D['TN'] += TN
			D['FN'] += FN
		except:
			print sample_name+" was not possible"
	file_name = "scoreSVM_fold_"+options.k_folds+"_tree_"+options.n_tree+"_mtry_"+options.m_try+"_boot_"+options.n_bootstrap+"_nsample_"+options.n_samples+".pickle"
	image_sauv_name_score = os.path.join(saving_location , file_name)


	im_pickle = open(image_sauv_name_score,  'wb')

	pickle.dump(D, im_pickle)
	
	diff_time = time.time() - start_time
	print 'Prediction time:'
	print '\t%02i:%02i:%02i' % (diff_time/3600, (diff_time%3600)/60, diff_time%60)
