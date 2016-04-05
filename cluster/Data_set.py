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

from optparse import OptionParser
from cluster_parameters import *

##---------------------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------------------



if __name__ ==  "__main__":
	description =\
'''
%prog - main loop for segmentation challenge .
'''

	#from cluster_parameters import *

    parser = OptionParser(usage="usage: %prog [options]",
			      description=description)
	parser.add_option("-i", "--input_folder", dest="input_folder",
			  help="input folder")
	parser.add_option("-o", "--output_folder", dest="output_folder",
			  help="output folder")
  	parser.add_option("-t", "--type", dest="img_type",
			  help="type of the image; can be either tumor or normal.")
	parser.add_option("--id", dest="img_id", 
                          help="image id.")    
    parser.add_option("--nb_samples", dest="nb_samples",
                          help="number of pixels drawn from the slide.")
	parser.add_option("--nb_images", dest="nb_images",
                          help="number of images drawn from the slide.")


	(options, args) = parser.parse_args()

	if (options.input_folder is None) or (options.output_folder is None) or (options.img_id is None):
		parser.error("incorrect number of arguments!")
	
    if not options.nb_samples is None:
		nb_samples=int(options.nb_samples)
	else:
		nb_samples = 100000

	if not options.nb_images is None:
		nb_images = int(options.nb_images)
	else:
		nb_images = 16

        slide_id = '%s_%03i' % (options.img_type, int(options.img_id))
        
	saving_location = os.path.join(options.output_folder, slide_id)	  
	if not os.path.isdir(saving_location):
	    print 'making ', saving_location
	    os.makedirs(saving_location)
	   
	db_server = sdba.SegmChallengeCamelyon16(options.input_folder, slide_to_do=slide_id)

	classif = sc.PixelClassification(db_server, slide_id, pixel_features_list, nb_samples=nb_samples)
	print "starting " + slide_id
	
	start = timeit.default_timer()

	X, Y, dico= classif.get_X_Y_for_train("train", N_squares=nb_images)
	X = classif.deal_with_missing_values_2(X)
 	
	stop = timeit.default_timer()

	print "time for " + slide_id + " " + str(stop-start)

 	print "ending " + slide_id
	
	image_sauv_name_pickle = os.path.join(saving_location ,  slide_id + ".pickle")
	image_sauv_name_npy    = os.path.join(saving_location ,  slide_id + ".npy")
  
	im_pickle = open(image_sauv_name_pickle,  'w')

	pickle.dump(dico,  im_pickle)
	im_pickle.close()
	
    print "save new matrix " + slide_id
	np.save(image_sauv_name_npy,  X)
	np.save(os.path.join(saving_location ,  slide_id + "_y_.npy"), Y)
