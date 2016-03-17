# -*- coding: cp1252 -*-
"""
Description: 
Challenge CAMELYON16.
Script for pixel classification. 

Authors:  Vaïa Machairas, Etienne Decencière, Peter Naylor, Thomas Walter.

Creation date: 2016-02-24
"""

import sys


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
from sklearn.ensemble import PeterRandomForestClassifier

##---------------------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------------------

############### Before doing anything you have 
############### to get a fold file, so launch that script
###############

if __name__ ==  "__main__":

	from cluster_parameters import *

	### inputs and folder reads

	input_1 = sys.argv[1]
	# input directories


	input_2 = int(sys.argv[2])
	# fold number

	input_3 = int(sys.argv[3])
	# Nb_samples

	input_4 = sys.argv[4]
	## version name


	if input_3 == 0:
		input_3 = 1000

	version_para = { 'n_sub': input_3 }

	
	saving_location = os.path.join(input_1, input_4)
	
	kfold_file = os.path.join(saving_location,'kfold.txt')
	f = open(kfold_file,'r')
	all_para = f.read().split('\n')
	
	i = input_2 ## fold number

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
	index = subsample(Y_temp, input_4, version_para)

	X_temp = X_temp[index,:]
	Y_temp = Y_temp[index]

	n_train = len(training_names) * input_3
	p_train = X_temp.shape[1]


	X_train = np.zeros(shape=(n_train, p_train))
	X_train[0:input_3,:] = X_temp

	Y_train = np.zeros(n_train)
	Y_train[0:input_3] = Y_temp

	i = 1

	for sample_name in training_names[1::]:

		image_sauv_name_pickle = os.path.join(saving_location ,sample_name, sample_name  + ".pickle")
		image_sauv_name_npy    = os.path.join(saving_location ,sample_name, sample_name  + ".npy")
		image_sauv_name_y_npy  = os.path.join(saving_location ,sample_name, sample_name  + "_y_.npy")

		X_temp = np.load( image_sauv_name_npy )
		Y_temp = np.load( image_sauv_name_y_npy ).ravel()from_list_string_to_list_Tumor
		index = subsample(Y_temp, input_4, version_para)

		X_temp = X_temp[index,:]
		Y_temp = Y_temp[index]

		X_train[ i * input_3 : (i+1) * input_3 ] = X_temp
		Y_train[ i * input_3 : (i+1) * input_3 ] = Y_temp


	myforest = RandomForestClassifier(n_estimators = n_tree, max_features = mtry, max_depth = None ) ## penser a changer bootstrap

	for sample_name in Normal_slides_train:

		image_sauv_name_npy    = os.path.join(saving_location ,sample_name, sample_name  + ".npy")
		image_sauv_name_y_npy  = os.path.join(saving_location ,sample_name, sample_name  + "_y_.npy")

		image_sauv_name_score = os.path.join(saving_location ,sample_name, "score_fold_"+str(input_2)+ ".pickle")
		im_pickle = open(image_sauv_name_pickle,  'w')
		D = {}


	pdb.set_trace()
	#os.path.join(version_path,original_image_name.split(".")[0])
	#X = classif.get_X_per_image_with_save_3_original(sp.Image(1000,1000), original_image_name
	#												 ,version_path,os.path.join(version_path,original_image_name.split(".")[0]))




	nature = input_2.split('_')[0]


  
	im_pickle = open(image_sauv_name_pickle,  'w')

	pickle.dump(dico,  im_pickle)
	im_pickle.close()
	print "save new matrix " + input_2
	np.save(image_sauv_name_npy,  X)
	np.save(os.path.join(saving_location ,  input_2 + "_y_.npy"), Y)