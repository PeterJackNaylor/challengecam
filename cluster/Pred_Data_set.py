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
	parser.add_option("--subsample_folder", dest="subsample_folder",
	                  help="Subsample folder", metavar="folder")

	(options, args) = parser.parse_args()
	para_ = [int(options.x),int(options.y),int(options.w),int(options.h),int(options.res)]
	dico_input = {'para':para_}

	#pdb.set_trace()

	#slide_to_do = options.type + "_" + (3-len(str(options.n)))*"0" + options.n + '.tif'
	try:	
		slide_to_do = '%s_%03i.tif' % (options.type, int(options.n))
	except:
		print 'incoherent inputs: '
		print 'type (-t): ', options.type
		print 'number (-n): ', options.number
		raise ValueError("aborted due to incoherent inputs to Pred_.py")

	db_server = sdba.SegmChallengeCamelyon16(options.folder_source, 
						 slide_to_do = slide_to_do,
						 type_method = "pred",
						 dico_ROI = dico_input)

	classif = sc.PixelClassification(db_server, options.out, pixel_features_list)#, nb_samples=input_3)
	print "starting " + slide_to_do
	
	start = timeit.default_timer()
	if options.type in ['Normal', 'Tumor']:
		iter_type = 'train'
	else:
		iter_type = 'prediction'
	
	alll =[]
        print iter_type
	for el in db_server.iter_final_prediction(iter_type):
		alll.append(el)
	if len(alll)>2:
		print "wrong inputs"
	#pdb.set_trace()
	original_image = alll[0][0]
	original_image_name = alll[0][1]
	folder_sauv_path = options.out

	#image_sauv_path = folder_sauv_path+"/"+original_image_name.split('_')[0] + "_" + original_image_name.split('_')[1]
	image_sauv_path = os.path.join(folder_sauv_path, '_'.join(original_image_name.split('_')[:2]))
	print image_sauv_path
	X = classif.get_X_per_image_with_save_3(original_image,  original_image_name,
							 folder_sauv_path,  image_sauv_path, save=True)
	#X = classif.deal_with_missing_values_2(X)
 	# we can deal with missing value later. 

	#if not options.subsample_folder is None:
         
	stop = timeit.default_timer()

	print "time for "+slide_to_do+" "+str(stop-start)

 	print "ending " + slide_to_do
