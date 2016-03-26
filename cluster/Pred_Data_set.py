# -*- coding: cp1252 -*-
"""
Description: 
Challenge CAMELYON16.
Script for pixel classification. 

Authors:  Vaïa Machairas, Etienne Decencière, Peter Naylor, Thomas Walter.

Creation date: 2016-02-24
"""
from optparse import OptionParser

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
	
	parser = OptionParser()

	parser.add_option("-s", "--source", dest="folder_source",
	                  help="Where to find Tumor files", metavar="FILE")
	parser.add_option("-t", "--type", dest="type",
	                  help="Normal, Tumor or Test", metavar="str")
	parser.add_option("-n", "--number", dest="n",
	                  help="Number of the type of slide", metavar="int")
	parser.add_option("-x", "--x_axis", dest="x",
	                  help="x", metavar="int")
	parser.add_option("-y", "--y_axis", dest="y",
	                  help="y", metavar="int")
	parser.add_option("-w", "--width", dest="w",
	                  help="width of the square", metavar="int")
	parser.add_option("-a", "--height", dest="h",
	                  help="height of the square", metavar="int")
	parser.add_option("-r", "--resolution", dest="res",
	                  help="resolution", metavar="int")
	parser.add_option("-o", "--output", dest="out",
	                  help="Output folder", metavar="folder")

	(options, args) = parser.parse_args()
	para_ = [int(options.x),int(options.y),int(options.w),int(options.h),int(options.res)]
	dico_input = {'para':para_}

	slide_to_do = options.type + "_" + (3-len(str(options.n)))*"0" + options.n + '.tif'

	db_server = sdba.SegmChallengeCamelyon16(options.folder_source, 
											 slide_to_do = slide_to_do,
											 type_method = "pred",
											 dico_ROI = dico_input)

	classif = sc.PixelClassification(db_server, options.out, pixel_features_list)#, nb_samples=input_3)
	print "starting " + slide_to_do
	
	start = timeit.default_timer()
	alll =[]
	for el in db_server.iter_training("train"):
		alll.append(el)
	if len(alll)>2:
		print "wrong inputs"
	original_image = alll[0][0]
	original_image_name = alll[0][2]
	folder_sauv_path = options.out

	image_sauv_path = folder_sauv_path+"/"+original_image_name.split('_')[0] + "_" + original_image_name.split('_')[1]

	X = classif.get_X_per_image_with_save_3_original(original_image,  original_image_name,
															 folder_sauv_path,  image_sauv_path)
	X = classif.deal_with_missing_values_2(X)
 	
	stop = timeit.default_timer()

	print "time for "+slide_to_do+" "+str(stop-start)

 	print "ending " + slide_to_do