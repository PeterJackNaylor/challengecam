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
CROSS_VALIDATION = True
LEARNING = False
PREDICTION = LEARNING
EVALUATION = False

##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## on a besoin du chemin d'accès de la base: base_path
base = "CAMELYON_1"

## morphomath
neighborhood_se = 'V6'
##lbp
radius_lpb = 2
points_lbp = 8
## SAF waterpixels
integrator_saf = 'mean'
wp1 = spp.WaterpixelsFunctor({"step":15, "k":4, "filter_ori":True})
wp2 = spp.WaterpixelsFunctor({"step":30, "k":4, "filter_ori":True})
wp3 = spp.WaterpixelsFunctor({"step":50, "k":4, "filter_ori":True})
wp4 = spp.WaterpixelsFunctor({"step":100, "k":4, "filter_ori":True})

## For cross validation:
nber_of_folds = 2
## Number of samples if needed: " REDUCTION2"
NB_SAMPLES = None
NB_SAMPLES = 1000##pour l'instant le laisser désactivé car gère le nb de slides, pas le nombre d'imagettes. A modifier.

###--------###--------###--------###--------###--------###--------###--------
## List of features:
pixel_features_list = [
##------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------
####wp1
##--------------------------------------------------------------------
### Identity
gf.GeneralFeature(op.IdentityFunctor({}), [0, 1, 2],  integrator_saf,  wp1, 'pixel'),
### MOMA geodesic
## erosion
#geo.GeneralFeatureGeodesic(og.GeodesicErosionFunctor({'neighborhood':neighborhood_se, 'size':1,  'integrator': integrator_saf}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicErosionFunctor({'neighborhood':neighborhood_se, 'size':2,  'integrator': integrator_saf}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicErosionFunctor({'neighborhood':neighborhood_se, 'size':3,  'integrator': integrator_saf}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicErosionFunctor({'neighborhood':neighborhood_se, 'size':4,  'integrator': integrator_saf}),  [0, 1, 2],  wp1,  'pixel'), 
### dilation
#geo.GeneralFeatureGeodesic(og.GeodesicDilationFunctor({'neighborhood':neighborhood_se, 'size':1,  'integrator': integrator_saf}),  [0, 1, 2],  wp1,  'pixel'), 
### opening
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood':neighborhood_se, 'size':1,  'integrator': integrator_saf}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood':neighborhood_se, 'size':2,  'integrator': integrator_saf}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood':neighborhood_se, 'size':3,  'integrator': integrator_saf}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood':neighborhood_se, 'size':4,  'integrator': integrator_saf}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood':neighborhood_se, 'size':5,  'integrator': integrator_saf}),  [0, 1, 2],  wp1,  'pixel'), 
### closing
#geo.GeneralFeatureGeodesic(og.GeodesicClosingFunctor({'neighborhood':neighborhood_se, 'size':1,  'integrator': integrator_saf}),  [0, 1, 2],  wp1,  'pixel'), 
### top hat
#geo.GeneralFeatureGeodesic(og.GeodesicTopHatFunctor({'neighborhood':neighborhood_se, 'size':3,  'integrator': integrator_saf}),  [0, 1, 2],  wp1,  'pixel'), 
### top hat inv
#geo.GeneralFeatureGeodesic(og.GeodesicTopHatInvFunctor({'neighborhood':neighborhood_se, 'size':3,  'integrator': integrator_saf}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicTopHatInvFunctor({'neighborhood':neighborhood_se, 'size':5,  'integrator': integrator_saf}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicTopHatInvFunctor({'neighborhood':neighborhood_se, 'size':10,  'integrator': integrator_saf}),  [0, 1, 2],  wp1,  'pixel'), 
### morpho gradient
#geo.GeneralFeatureGeodesic(og.GeodesicMorphoGradientFunctor({'neighborhood':neighborhood_se, 'size':1,  'integrator': integrator_saf}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicMorphoGradientFunctor({'neighborhood':neighborhood_se, 'size':2,  'integrator': integrator_saf}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicMorphoGradientFunctor({'neighborhood':neighborhood_se, 'size':3,  'integrator': integrator_saf}),  [0, 1, 2],  wp1,  'pixel'), 
#### Texture
### Haralick
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'AngularSecondMoment'}), [0, 1, 2],  wp1, 'pixel'),
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'Contrast'}), [0, 1, 2],  wp1, 'pixel'),
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'Correlation'}), [0, 1, 2],  wp1, 'pixel'),
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'SumofSquaresVariance'}), [0, 1, 2],  wp1, 'pixel'),
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'InverseDifferenceMoment'}), [0, 1, 2],  wp1, 'pixel'),
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'SumAverage'}), [0, 1, 2],  wp1, 'pixel'),
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'SumVariance'}), [0, 1, 2],  wp1, 'pixel'),
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'SumEntropy'}), [0, 1, 2],  wp1, 'pixel'),
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'Entropy'}), [0, 1, 2],  wp1, 'pixel'),
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'DifferenceVariance'}), [0, 1, 2],  wp1, 'pixel'),
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'DifferenceEntropy'}), [0, 1, 2],  wp1, 'pixel'),
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'InformationMeasureofCorrelation1'}), [0, 1, 2],  wp1, 'pixel'),
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'InformationMeasureofCorrelation2'}), [0, 1, 2],  wp1, 'pixel'),
### LBP
#geo.GeneralFeatureGeodesic(og.LBP_bin1({'radius': radius_lpb,  'points': points_lbp, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.LBP_bin2({'radius': radius_lpb,  'points': points_lbp, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.LBP_bin3({'radius': radius_lpb,  'points': points_lbp, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.LBP_bin4({'radius': radius_lpb,  'points': points_lbp, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.LBP_bin5({'radius': radius_lpb,  'points': points_lbp, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.LBP_bin6({'radius': radius_lpb,  'points': points_lbp, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp1,  'pixel'), 
###------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
###------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
###--------------------------------------------------------------------
#####wp2
###--------------------------------------------------------------------
#### Identity
#gf.GeneralFeature(op.IdentityFunctor({}), [0, 1, 2],  integrator_saf,  wp2, 'pixel'),
#### MOMA geodesic
### erosion
#geo.GeneralFeatureGeodesic(og.GeodesicErosionFunctor({'neighborhood':neighborhood_se, 'size':1,  'integrator': integrator_saf}),  [0, 1, 2],  wp2,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicErosionFunctor({'neighborhood':neighborhood_se, 'size':2,  'integrator': integrator_saf}),  [0, 1, 2],  wp2,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicErosionFunctor({'neighborhood':neighborhood_se, 'size':3,  'integrator': integrator_saf}),  [0, 1, 2],  wp2,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicErosionFunctor({'neighborhood':neighborhood_se, 'size':4,  'integrator': integrator_saf}),  [0, 1, 2],  wp2,  'pixel'), 
### dilation
#geo.GeneralFeatureGeodesic(og.GeodesicDilationFunctor({'neighborhood':neighborhood_se, 'size':1,  'integrator': integrator_saf}),  [0, 1, 2],  wp2,  'pixel'), 
### opening
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood':neighborhood_se, 'size':1,  'integrator': integrator_saf}),  [0, 1, 2],  wp2,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood':neighborhood_se, 'size':2,  'integrator': integrator_saf}),  [0, 1, 2],  wp2,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood':neighborhood_se, 'size':3,  'integrator': integrator_saf}),  [0, 1, 2],  wp2,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood':neighborhood_se, 'size':4,  'integrator': integrator_saf}),  [0, 1, 2],  wp2,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood':neighborhood_se, 'size':5,  'integrator': integrator_saf}),  [0, 1, 2],  wp2,  'pixel'), 
### closing
#geo.GeneralFeatureGeodesic(og.GeodesicClosingFunctor({'neighborhood':neighborhood_se, 'size':1,  'integrator': integrator_saf}),  [0, 1, 2],  wp2,  'pixel'), 
### top hat
#geo.GeneralFeatureGeodesic(og.GeodesicTopHatFunctor({'neighborhood':neighborhood_se, 'size':3,  'integrator': integrator_saf}),  [0, 1, 2],  wp2,  'pixel'), 
### top hat inv
#geo.GeneralFeatureGeodesic(og.GeodesicTopHatInvFunctor({'neighborhood':neighborhood_se, 'size':3,  'integrator': integrator_saf}),  [0, 1, 2],  wp2,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicTopHatInvFunctor({'neighborhood':neighborhood_se, 'size':5,  'integrator': integrator_saf}),  [0, 1, 2],  wp2,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicTopHatInvFunctor({'neighborhood':neighborhood_se, 'size':10,  'integrator': integrator_saf}),  [0, 1, 2],  wp2,  'pixel'), 
### morpho gradient
#geo.GeneralFeatureGeodesic(og.GeodesicMorphoGradientFunctor({'neighborhood':neighborhood_se, 'size':1,  'integrator': integrator_saf}),  [0, 1, 2],  wp2,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicMorphoGradientFunctor({'neighborhood':neighborhood_se, 'size':2,  'integrator': integrator_saf}),  [0, 1, 2],  wp2,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicMorphoGradientFunctor({'neighborhood':neighborhood_se, 'size':3,  'integrator': integrator_saf}),  [0, 1, 2],  wp2,  'pixel'), 
#### Texture
### Haralick
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'AngularSecondMoment'}), [0, 1, 2],  wp2, 'pixel'),
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'Contrast'}), [0, 1, 2],  wp2, 'pixel'),
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'Correlation'}), [0, 1, 2],  wp2, 'pixel'),
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'SumofSquaresVariance'}), [0, 1, 2],  wp2, 'pixel'),
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'InverseDifferenceMoment'}), [0, 1, 2],  wp2, 'pixel'),
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'SumAverage'}), [0, 1, 2],  wp2, 'pixel'),
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'SumVariance'}), [0, 1, 2],  wp2, 'pixel'),
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'SumEntropy'}), [0, 1, 2],  wp2, 'pixel'),
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'Entropy'}), [0, 1, 2],  wp2, 'pixel'),
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'DifferenceVariance'}), [0, 1, 2],  wp2, 'pixel'),
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'DifferenceEntropy'}), [0, 1, 2],  wp2, 'pixel'),
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'InformationMeasureofCorrelation1'}), [0, 1, 2],  wp2, 'pixel'),
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'InformationMeasureofCorrelation2'}), [0, 1, 2],  wp2, 'pixel'),
## LBP
#geo.GeneralFeatureGeodesic(og.LBP_bin1({'radius': radius_lpb,  'points': points_lbp, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp2,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.LBP_bin2({'radius': radius_lpb,  'points': points_lbp, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp2,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.LBP_bin3({'radius': radius_lpb,  'points': points_lbp, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp2,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.LBP_bin4({'radius': radius_lpb,  'points': points_lbp, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp2,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.LBP_bin5({'radius': radius_lpb,  'points': points_lbp, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp2,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.LBP_bin6({'radius': radius_lpb,  'points': points_lbp, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp2,  'pixel'), 
##------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

##------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------
####wp3
##--------------------------------------------------------------------
### Identity
#gf.GeneralFeature(op.IdentityFunctor({}), [0, 1, 2],  integrator_saf,  wp3, 'pixel'),
### MOMA geodesic
## erosion
#geo.GeneralFeatureGeodesic(og.GeodesicErosionFunctor({'neighborhood':neighborhood_se, 'size':1,  'integrator': integrator_saf}),  [0, 1, 2],  wp3,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicErosionFunctor({'neighborhood':neighborhood_se, 'size':2,  'integrator': integrator_saf}),  [0, 1, 2],  wp3,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicErosionFunctor({'neighborhood':neighborhood_se, 'size':3,  'integrator': integrator_saf}),  [0, 1, 2],  wp3,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicErosionFunctor({'neighborhood':neighborhood_se, 'size':4,  'integrator': integrator_saf}),  [0, 1, 2],  wp3,  'pixel'), 
## dilation
#geo.GeneralFeatureGeodesic(og.GeodesicDilationFunctor({'neighborhood':neighborhood_se, 'size':1,  'integrator': integrator_saf}),  [0, 1, 2],  wp3,  'pixel'), 
## opening
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood':neighborhood_se, 'size':1,  'integrator': integrator_saf}),  [0, 1, 2],  wp3,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood':neighborhood_se, 'size':2,  'integrator': integrator_saf}),  [0, 1, 2],  wp3,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood':neighborhood_se, 'size':3,  'integrator': integrator_saf}),  [0, 1, 2],  wp3,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood':neighborhood_se, 'size':4,  'integrator': integrator_saf}),  [0, 1, 2],  wp3,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood':neighborhood_se, 'size':5,  'integrator': integrator_saf}),  [0, 1, 2],  wp3,  'pixel'), 
## closing
#geo.GeneralFeatureGeodesic(og.GeodesicClosingFunctor({'neighborhood':neighborhood_se, 'size':1,  'integrator': integrator_saf}),  [0, 1, 2],  wp3,  'pixel'), 
## top hat
#geo.GeneralFeatureGeodesic(og.GeodesicTopHatFunctor({'neighborhood':neighborhood_se, 'size':3,  'integrator': integrator_saf}),  [0, 1, 2],  wp3,  'pixel'), 
## top hat inv
#geo.GeneralFeatureGeodesic(og.GeodesicTopHatInvFunctor({'neighborhood':neighborhood_se, 'size':3,  'integrator': integrator_saf}),  [0, 1, 2],  wp3,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicTopHatInvFunctor({'neighborhood':neighborhood_se, 'size':5,  'integrator': integrator_saf}),  [0, 1, 2],  wp3,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicTopHatInvFunctor({'neighborhood':neighborhood_se, 'size':10,  'integrator': integrator_saf}),  [0, 1, 2],  wp3,  'pixel'), 
## morpho gradient
#geo.GeneralFeatureGeodesic(og.GeodesicMorphoGradientFunctor({'neighborhood':neighborhood_se, 'size':1,  'integrator': integrator_saf}),  [0, 1, 2],  wp3,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicMorphoGradientFunctor({'neighborhood':neighborhood_se, 'size':2,  'integrator': integrator_saf}),  [0, 1, 2],  wp3,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicMorphoGradientFunctor({'neighborhood':neighborhood_se, 'size':3,  'integrator': integrator_saf}),  [0, 1, 2],  wp3,  'pixel'), 
##------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

##------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------
####wp4
##--------------------------------------------------------------------
### Identity
#gf.GeneralFeature(op.IdentityFunctor({}), [0, 1, 2],  integrator_saf,  wp4, 'pixel'),
### MOMA geodesic
## erosion
#geo.GeneralFeatureGeodesic(og.GeodesicErosionFunctor({'neighborhood':neighborhood_se, 'size':1,  'integrator': integrator_saf}),  [0, 1, 2],  wp4,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicErosionFunctor({'neighborhood':neighborhood_se, 'size':2,  'integrator': integrator_saf}),  [0, 1, 2],  wp4,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicErosionFunctor({'neighborhood':neighborhood_se, 'size':3,  'integrator': integrator_saf}),  [0, 1, 2],  wp4,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicErosionFunctor({'neighborhood':neighborhood_se, 'size':4,  'integrator': integrator_saf}),  [0, 1, 2],  wp4,  'pixel'), 
## dilation
#geo.GeneralFeatureGeodesic(og.GeodesicDilationFunctor({'neighborhood':neighborhood_se, 'size':1,  'integrator': integrator_saf}),  [0, 1, 2],  wp4,  'pixel'), 
## opening
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood':neighborhood_se, 'size':1,  'integrator': integrator_saf}),  [0, 1, 2],  wp4,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood':neighborhood_se, 'size':2,  'integrator': integrator_saf}),  [0, 1, 2],  wp4,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood':neighborhood_se, 'size':3,  'integrator': integrator_saf}),  [0, 1, 2],  wp4,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood':neighborhood_se, 'size':4,  'integrator': integrator_saf}),  [0, 1, 2],  wp4,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood':neighborhood_se, 'size':5,  'integrator': integrator_saf}),  [0, 1, 2],  wp4,  'pixel'), 
## closing
#geo.GeneralFeatureGeodesic(og.GeodesicClosingFunctor({'neighborhood':neighborhood_se, 'size':1,  'integrator': integrator_saf}),  [0, 1, 2],  wp4,  'pixel'), 
## top hat
#geo.GeneralFeatureGeodesic(og.GeodesicTopHatFunctor({'neighborhood':neighborhood_se, 'size':3,  'integrator': integrator_saf}),  [0, 1, 2],  wp4,  'pixel'), 
## top hat inv
#geo.GeneralFeatureGeodesic(og.GeodesicTopHatInvFunctor({'neighborhood':neighborhood_se, 'size':3,  'integrator': integrator_saf}),  [0, 1, 2],  wp4,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicTopHatInvFunctor({'neighborhood':neighborhood_se, 'size':5,  'integrator': integrator_saf}),  [0, 1, 2],  wp4,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicTopHatInvFunctor({'neighborhood':neighborhood_se, 'size':10,  'integrator': integrator_saf}),  [0, 1, 2],  wp4,  'pixel'), 
## morpho gradient
#geo.GeneralFeatureGeodesic(og.GeodesicMorphoGradientFunctor({'neighborhood':neighborhood_se, 'size':1,  'integrator': integrator_saf}),  [0, 1, 2],  wp4,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicMorphoGradientFunctor({'neighborhood':neighborhood_se, 'size':2,  'integrator': integrator_saf}),  [0, 1, 2],  wp4,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicMorphoGradientFunctor({'neighborhood':neighborhood_se, 'size':3,  'integrator': integrator_saf}),  [0, 1, 2],  wp4,  'pixel'), 
##------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

]

for waterpixpix in [wp1, wp2, wp3, wp4]:
	for sigma in [1]:

		GS = gf.GeneralFeature(op.IlastikGaussianSmoothing({'sigma':sigma}), [0, 1, 2],  'mean',  waterpixpix, 'pixel')
		#LoG = gf.GeneralFeature(op.IlastikLaplacianOfGaussian({'scale':sigma}), [0, 1, 2],  'mean',  waterpixpix, 'pixel') 
		#GGM = gf.GeneralFeature(op.IlastikGaussianGradientMagnitude({'sigma':sigma}), [0, 1, 2],  'mean',  waterpixpix, 'pixel')
		#DoG = gf.GeneralFeature(op.IlastikDifferenceOfGaussians({'sigma1':sigma * np.sqrt(2),  'sigma2':sigma / np.sqrt(2)}), [0, 1, 2],  'mean',  waterpixpix, 'pixel')
		#ST1 = gf.GeneralFeature(op.IlastikStructureTensorEigenValues({'innerScale':sigma,  'outerScale':2,  'eigenvalueNumber': 0}), [0, 1, 2],  'mean',  waterpixpix, 'pixel') 
		#ST2 = gf.GeneralFeature(op.IlastikStructureTensorEigenValues({'innerScale':sigma,  'outerScale':2,  'eigenvalueNumber': 1}), [0, 1, 2],  'mean',  waterpixpix, 'pixel')
		#HG1 = gf.GeneralFeature(op.IlastikHessianOfGaussianEigenvalues({'scale':sigma, 'eigenvalueNumber': 0}), [0, 1, 2],  'mean',  waterpixpix, 'pixel')
		#HG2 = gf.GeneralFeature(op.IlastikHessianOfGaussianEigenvalues({'scale':sigma, 'eigenvalueNumber': 1}), [0, 1, 2],  'mean',  waterpixpix, 'pixel')

		pixel_features_list.append(GS)
		#pixel_features_list.append(LoG)
		#pixel_features_list.append(GGM)
		#pixel_features_list.append(DoG)
		#pixel_features_list.append(ST1)
		#pixel_features_list.append(ST2)
		#pixel_features_list.append(HG1)
		#pixel_features_list.append(HG2)



###--------###--------###--------###--------###--------###--------###--------
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Classifier:
out_of_bag_score = False  # permet de calculer une erreur d'apprentissage, mais coute cher en temps de calcul
myforest = ens.RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_leaf = 100, max_features='auto', bootstrap=True, n_jobs=4, random_state=42, oob_score=out_of_bag_score)
#myforest = ens.RandomForestClassifier(n_estimators=100, criterion='gini', max_features='auto', bootstrap=True, n_jobs=4, random_state=42, oob_score=out_of_bag_score)

##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
