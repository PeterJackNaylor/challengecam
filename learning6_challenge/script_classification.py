# -*- coding: cp1252 -*-
"""
Description: 
Challenge CAMELYON16.
Script for pixel classification. 

Authors:  Vaïa Machairas, Etienne Decencière, Peter Naylor, Thomas Walter.

Creation date: 2016-02-24
"""
import pdb
import os

from getpass import getuser
import smilPython as sp
import numpy as np

import folder_functions as ff
import segmentation_by_classification as sc
from evaluation import my_metrics
##---------------------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------------------
if getuser() == "vaiamachairas":
    from vaia_parameters2 import *

if getuser() == "decencie":
    from etienne_parameters import *

if getuser()=="naylor":
    from peter_parameters import *

if getuser()=="pubuntu":
    from peter_parameters import *

print getuser()
##---------------------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------------------
## Pour traitement de la database:
if base == "my_verif":
    db_server = sdba.SegmDataBaseStandard(base_path)
if base == "CAMELYON_1":
    db_server = sdba.SegmChallengeCamelyon16(base_path)
##---------------------------------------------------------------------------------------------------------------------------------------
## Création des dossiers:
nevermind = ff.create_all_subsets_folders(base_path)
##--------------------------------------------------------------------------------------------------------------------------------------- 
output_dir = os.path.join(base_path, "resultats")
##--------------------------------------------------------------------------------------------------------------------------------------- 
## For evaluation of the model:
if CROSS_VALIDATION:
    classif_cv = sc.PixelClassification(db_server, output_dir, pixel_features_list, nb_samples=NB_SAMPLES)
    classif_cv.set_classifier(myforest)
    performance = classif_cv.cross_validation(nber_of_folds)
    print "Performance of learning method: ",  performance
##--------------------------------------------------------------------------------------------------------------------------------------- 
## Learning on the train database:
if LEARNING:
    print "Pixel classification"
    print pixel_features_list
    classif = sc.PixelClassification(db_server, output_dir, pixel_features_list, nb_samples=NB_SAMPLES)
    classif.set_classifier(myforest)
    classif.fit()
##--------------------------------------------------------------------------------------------------------------------------------------- 
    ## Features importance:
    list_feat = classif.get_features_names_list()
    importances = myforest.feature_importances_
    ##std = np.std([tree.feature_importances_ for tree in myforest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    ### Print the feature ranking:
    print("Feature ranking:")
    for f in range(len(indices)):
        print("%d. feature %s (%f)" % (f + 1, list_feat[indices[f]], importances[indices[f]]))   
##---------------------------------------------------------------------------------------------------------------------------------------
## Prediction:
if PREDICTION:
    for im_train,  name in db_server.train_final_prediction():
        print name
        im_uc_lab = classif.transform_image_uc(im_train)
        Y = classif.predict(im_train,  name)  
        im_train_out = classif.visu_prediction_on_uc_image(im_uc_lab, Y)
        sp.compare(im_train_out, ">", 0, 255, 0, im_train_out)
        sp.write(im_train_out, os.path.join(db_server._res_dir['train'],name))
        
    for im_test, name in db_server.test_final_prediction():
        print name
        im_uc_lab = classif.transform_image_uc(im_test)
        Y = classif.predict(im_test,  name)  
        im_test_out = classif.visu_prediction_on_uc_image(im_uc_lab, Y)
        sp.compare(im_test_out, ">", 0, 255, 0, im_test_out)
        sp.write(im_test_out, os.path.join(db_server._res_dir['test'],name))    
##--------------------------------------------------------------------------------------------------------------------------------------- 



