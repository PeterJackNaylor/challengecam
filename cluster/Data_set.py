# -*- coding: cp1252 -*-
"""
Description: 
Challenge CAMELYON16.
Script for pixel classification. 

Authors:  Vaïa Machairas, Etienne Decencière, Peter Naylor, Thomas Walter.

Creation date: 2016-02-24
"""

import sys
import timeit

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

##---------------------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------------------



if __name__ ==  "__main__":

	from cluster_parameters import *

	input_1 = sys.argv[1]
	# input directories


	input_2 = sys.argv[2]
	# Tumor or Normal

	input_2_bis = sys.argv[3]

	input_2 = input_2 + "_" +(3-len(input_2_bis))*"0"+input_2_bis
	### number of zeros before digit, example 007 
	input_3 = int(sys.argv[4])
	# Nb_samples for one slide

	input_4 = sys.argv[5]
	## version name


	if input_3 == 0:
		input_3 = None
	
	saving_location = os.path.join(input_1, input_4, input_2)
	if os.path.isdir( os.path.join(input_1, input_4) ):
		if not os.path.isdir(saving_location):
			os.mkdir(saving_location)
	else:
		print "No version named folder created"

	db_server = sdba.SegmChallengeCamelyon16(input_1, slide_to_do= input_2)

	classif = sc.PixelClassification(db_server, input_2, pixel_features_list, nb_samples=input_3)
	print "starting " + input_2
	
	start = timeit.default_timer()

	X, Y, dico= classif.get_X_Y_for_train("train")
	X = classif.deal_with_missing_values_2(X)
 	
	stop = timeit.default_timer()

	print "time for "+input_2+" "+str(stop-start)

 	print "ending " + input_2
	
	nature = input_2.split('_')[0]

	image_sauv_name_pickle = os.path.join(saving_location ,  input_2 + ".pickle")
	image_sauv_name_npy    = os.path.join(saving_location ,  input_2 + ".npy")
  
	im_pickle = open(image_sauv_name_pickle,  'w')

	pickle.dump(dico,  im_pickle)
	im_pickle.close()
	print "save new matrix " + input_2
	np.save(image_sauv_name_npy,  X)
	np.save(os.path.join(saving_location ,  input_2 + "_y_.npy"), Y)