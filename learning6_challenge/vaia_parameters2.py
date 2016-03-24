# -*- coding: cp1252 -*-
"""
Description: 
Challenge CAMELYON16.
This file contains Peter's parameters for script_classification.py. 

Authors:  Va�a Machairas, Etienne Decenci�re, Peter Naylor, Thomas Walter.

Creation date: 2016-02-24
"""
import os
import pdb
from getpass import getuser

import sklearn.ensemble as ens

import segm_db_access as sdba
import useful_functions as uf
import spp_functors as spp
import op_functors as op
import general_feature_for_pixel_support as pf
#import cytomine_window as cw
import op_functors_geodesic as og
import general_feature_for_SP_support as gf
import general_feature_for_SP_support_geodesic as geo
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
CROSS_VALIDATION = True
LEARNING = False
PREDICTION = LEARNING
EVALUATION = False

##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## on a besoin du chemin d'acc�s de la base: base_path
base = "CAMELYON_1"
base_path = "/home/vaiamachairas/Documents/challengeCAMELYON16/base_tmp/Temp_Vaia/"
#base = "my_verif"
#base_path = "/home/vaiamachairas/Documents/databases/base_verification/"
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## MAIN parameters #############################

## For cross validation:
nber_of_folds = 2
## Number of samples if needed: " REDUCTION2"
NB_SAMPLES = None
NB_SAMPLES = 1000##pour l'instant le laisser d�sactiv� car g�re le nb de slides, pas le nombre d'imagettes. A modifier.
### For features:
## Superpixel functors
wp1 = spp.WaterpixelsFunctor({"step":30, "k":4, "filter_ori":True})
## Some common parameters:
neighborhood = 'V8'
size = 2
se = uf.set_structuring_element(neighborhood, size)
###--------###--------###--------###--------###--------###--------###--------
## List of features:
pixel_features_list = [

###--------###
#### support = pixel
## Identity:
#pf.GeneralFeature(op.IdentityFunctor({}), [0, 1, 2]), 
#pf.GeneralFeature(op.IdentityFunctor({}), [0, 1, 2]), 
## MOMA:
#pf.GeneralFeature(op.ErosionFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2]), 

###--------###

### support = spix (SAF)
## Identity:
#gf.GeneralFeature(op.IdentityFunctor({}), [0, 1, 2],  'mean',  wp1, 'pixel'), 
## Ilastik:
gf.GeneralFeature(op.IlastikGaussianSmoothing({'sigma':5}), [0, 1, 2],  'mean',  wp1, 'pixel'), 
gf.GeneralFeature(op.IlastikLaplacianOfGaussian({'scale':5}), [0, 1, 2],  'mean',  wp1, 'pixel'), 
gf.GeneralFeature(op.IlastikGaussianGradientMagnitude({'sigma':5}), [0, 1, 2],  'mean',  wp1, 'pixel'), 
gf.GeneralFeature(op.IlastikDifferenceOfGaussians({'sigma1':5,  'sigma2':10}), [0, 1, 2],  'mean',  wp1, 'pixel'), 
gf.GeneralFeature(op.IlastikStructureTensorEigenValues({'innerScale':5,  'outerScale':10,  'eigenvalueNumber': 0}), [0, 1, 2],  'mean',  wp1, 'pixel'), 
gf.GeneralFeature(op.IlastikStructureTensorEigenValues({'innerScale':5,  'outerScale':10,  'eigenvalueNumber': 1}), [0, 1, 2],  'mean',  wp1, 'pixel'), 
gf.GeneralFeature(op.IlastikHessianOfGaussianEigenvalues({'scale':5, 'eigenvalueNumber': 0}), [0, 1, 2],  'mean',  wp1, 'pixel'), 
gf.GeneralFeature(op.IlastikHessianOfGaussianEigenvalues({'scale':5, 'eigenvalueNumber': 1}), [0, 1, 2],  'mean',  wp1, 'pixel'), 
## MOMA:
#geo.GeneralFeatureGeodesic(og.GeodesicErosionFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicDilationFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicClosingFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicMorphoGradientFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicTopHatFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp1,  'pixel'), 
##Texture: Haralick, support = spix (SAF):
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'AngularSecondMoment'}), [0, 1, 2],  wp1, 'pixel'),
## Texture: LBP, support = spix (SAF):
#geo.GeneralFeatureGeodesic(og.LBP_bin1({'radius': 2,  'points': 8, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp1,  'pixel'), 

###--------###

### support = cytomine window
#cw.CytomineWindow(cw.FromImageToMatrix_Window({'window_size':1,  'substitution_value':0}),  [0, 1, 2]),

]
###--------###--------###--------###--------###--------###--------###--------
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Classifier:
out_of_bag_score = False  # permet de calculer une erreur d'apprentissage, mais coute cher en temps de calcul
myforest = ens.RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_leaf = 100, max_features='auto', bootstrap=True, n_jobs=4, random_state=42, oob_score=out_of_bag_score)
myforest = ens.RandomForestClassifier(n_estimators=100, criterion='gini', max_features='auto', bootstrap=True, n_jobs=4, random_state=42, oob_score=out_of_bag_score)

##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
