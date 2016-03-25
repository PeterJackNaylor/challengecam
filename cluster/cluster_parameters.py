# -*- coding: cp1252 -*-
"""
Description: 
Challenge CAMELYON16.
This file contains Peter's parameters for script_classification.py. 

Authors:  Vaïa Machairas, Etienne Decencière, Peter Naylor, Thomas Walter.

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
import numpy as np
#import cytomine_window as cw
import op_functors_geodesic as og
import general_feature_for_SP_support as gf
import general_feature_for_SP_support_geodesic as geo
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
CROSS_VALIDATION = False
LEARNING = True
PREDICTION = LEARNING
EVALUATION = False

##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## on a besoin du chemin d'acc�s de la base: base_path
base = "CAMELYON_1"
base_path = "/home/vaiamachairas/Documents/challengeCAMELYON16/base_tmp/Temp_Vaia/"
#base = "my_verif"
#base_path = "/home/vaiamachairas/Documents/databases/base_verification/"

## morphomath
neighborhood_se = 'V6'
##lbp
radius_lpb = 2
points_lbp = 8
## SAF waterpixels
integrator_saf = 'mean'
wp1 = spp.WaterpixelsFunctor({"step":15, "k":4, "filter_ori":True})
wp2 = spp.WaterpixelsFunctor({"step":20, "k":4, "filter_ori":True})
wp3 = spp.WaterpixelsFunctor({"step":30, "k":4, "filter_ori":True})

## For cross validation:
nber_of_folds = 2
## Number of samples if needed: " REDUCTION2"
NB_SAMPLES = None
NB_SAMPLES = 1000##pour l'instant le laisser désactivé car gère le nb de slides, pas le nombre d'imagettes. A modifier.

###--------###--------###--------###--------###--------###--------###--------
## List of features:
pixel_features_list = [
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
gf.GeneralFeature(op.IdentityFunctor({}), [0, 1],  integrator_saf,  wp1, 'pixel'),
gf.GeneralFeature(op.ErosionFunctor({'neighborhood':neighborhood_se, 'size':1}), [0, 1],  integrator_saf,  wp1, 'pixel'),
gf.GeneralFeature(op.ErosionFunctor({'neighborhood':neighborhood_se, 'size':3}), [0, 1],  integrator_saf,  wp1, 'pixel'),
gf.GeneralFeature(op.OpeningFunctor({'neighborhood':neighborhood_se, 'size':1}), [0, 1],  integrator_saf,  wp1, 'pixel'),
gf.GeneralFeature(op.OpeningFunctor({'neighborhood':neighborhood_se, 'size':3}), [0, 1],  integrator_saf,  wp1, 'pixel'),
gf.GeneralFeature(op.OpeningFunctor({'neighborhood':neighborhood_se, 'size':5}), [0, 1],  integrator_saf,  wp1, 'pixel'),
gf.GeneralFeature(op.TopHatInvFunctor({'neighborhood':neighborhood_se, 'size':3}), [0, 1],  integrator_saf,  wp1, 'pixel'),
gf.GeneralFeature(op.TopHatInvFunctor({'neighborhood':neighborhood_se, 'size':5}), [0, 1],  integrator_saf,  wp1, 'pixel'),
gf.GeneralFeature(op.MorphologicalGradientFunctor({'neighborhood':neighborhood_se, 'size':1}), [0, 1],  integrator_saf,  wp1, 'pixel'),
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
gf.GeneralFeature(op.IdentityFunctor({}), [0, 1],  integrator_saf,  wp3, 'pixel'),
gf.GeneralFeature(op.ErosionFunctor({'neighborhood':neighborhood_se, 'size':3}), [0, 1],  integrator_saf,  wp3, 'pixel'),
gf.GeneralFeature(op.OpeningFunctor({'neighborhood':neighborhood_se, 'size':3}), [0, 1],  integrator_saf,  wp3, 'pixel'),
gf.GeneralFeature(op.TopHatInvFunctor({'neighborhood':neighborhood_se, 'size':3}), [0, 1],  integrator_saf,  wp3, 'pixel'),
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
geo.GeneralFeatureGeodesicList(
							[og.HaralickFeature({'direction': 'all',  'feature_name': 'AngularSecondMoment'}), 
							 og.HaralickFeature({'direction': 'all',  'feature_name': 'Contrast'}), 
							 og.HaralickFeature({'direction': 'all',  'feature_name': 'Correlation'}), 
							 og.HaralickFeature({'direction': 'all',  'feature_name': 'SumofSquaresVariance'}), 
							 og.HaralickFeature({'direction': 'all',  'feature_name': 'InverseDifferenceMoment'}), 
							 og.HaralickFeature({'direction': 'all',  'feature_name': 'SumAverage'}), 
							 og.HaralickFeature({'direction': 'all',  'feature_name': 'SumVariance'}), 
							 og.HaralickFeature({'direction': 'all',  'feature_name': 'SumEntropy'}), 
							 og.HaralickFeature({'direction': 'all',  'feature_name': 'Entropy'}), 
							 og.HaralickFeature({'direction': 'all',  'feature_name': 'DifferenceVariance'}), 
							 og.HaralickFeature({'direction': 'all',  'feature_name': 'DifferenceEntropy'}), 
							 og.HaralickFeature({'direction': 'all',  'feature_name': 'InformationMeasureofCorrelation1'}), 
							 og.HaralickFeature({'direction': 'all',  'feature_name': 'InformationMeasureofCorrelation2'})], [0, 1],  wp2, 'pixel'),
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

]

for waterpixpix in [ wp2]:
	for sigma in [1.6]:
		DoG = gf.GeneralFeature(op.IlastikDifferenceOfGaussians({'sigma1':sigma * np.sqrt(2),  'sigma2':sigma / np.sqrt(2)}), [0, 1],  integrator_saf,  waterpixpix, 'pixel')
		ST1 = gf.GeneralFeature(op.IlastikStructureTensorEigenValues({'innerScale':sigma,  'outerScale':2,  'eigenvalueNumber': 0}), [0, 1],  integrator_saf,  waterpixpix, 'pixel') 
		ST2 = gf.GeneralFeature(op.IlastikStructureTensorEigenValues({'innerScale':sigma,  'outerScale':2,  'eigenvalueNumber': 1}), [0, 1],  integrator_saf,  waterpixpix, 'pixel')
		HG1 = gf.GeneralFeature(op.IlastikHessianOfGaussianEigenvalues({'scale':sigma, 'eigenvalueNumber': 0}), [0, 1],  integrator_saf,  waterpixpix, 'pixel')
		HG2 = gf.GeneralFeature(op.IlastikHessianOfGaussianEigenvalues({'scale':sigma, 'eigenvalueNumber': 1}), [0, 1],  integrator_saf,  waterpixpix, 'pixel')

		pixel_features_list.append(DoG)
		pixel_features_list.append(ST1)
		pixel_features_list.append(ST2)
		pixel_features_list.append(HG1)
		pixel_features_list.append(HG2)

###--------###--------###--------###--------###--------###--------###--------
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Classifier:
out_of_bag_score = False  # permet de calculer une erreur d'apprentissage, mais coute cher en temps de calcul
myforest = ens.RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_leaf = 100, max_features='auto', bootstrap=True, n_jobs=4, random_state=42, oob_score=out_of_bag_score)
#myforest = ens.RandomForestClassifier(n_estimators=100, criterion='gini', max_features='auto', bootstrap=True, n_jobs=4, random_state=42, oob_score=out_of_bag_score)

##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
