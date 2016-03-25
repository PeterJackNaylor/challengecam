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
import op_functors as op
import general_feature_for_pixel_support as pf
import numpy as np
#import cytomine_window as cw
import op_functors_geodesic as og


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


## For cross validation:
nber_of_folds = 2
## Number of samples if needed: " REDUCTION2"
NB_SAMPLES = None
NB_SAMPLES = 1000##pour l'instant le laisser désactivé car gère le nb de slides, pas le nombre d'imagettes. A modifier.

###--------###--------###--------###--------###--------###--------###--------
## List of features:
pixel_features_list = [
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
pf.GeneralFeature(op.IdentityFunctor({}), [0, 1]), 

pf.GeneralFeature(op.ErosionFunctor({'neighborhood':neighborhood_se, 'size':1}), [0, 1]), 
pf.GeneralFeature(op.ErosionFunctor({'neighborhood':neighborhood_se, 'size':2}), [0, 1]), 
pf.GeneralFeature(op.ErosionFunctor({'neighborhood':neighborhood_se, 'size':3}), [0, 1]), 
pf.GeneralFeature(op.ErosionFunctor({'neighborhood':neighborhood_se, 'size':4}), [0, 1]), 

pf.GeneralFeature(op.OpeningFunctor({'neighborhood':neighborhood_se, 'size':1}), [0, 1]), 
pf.GeneralFeature(op.OpeningFunctor({'neighborhood':neighborhood_se, 'size':2}), [0, 1]), 
pf.GeneralFeature(op.OpeningFunctor({'neighborhood':neighborhood_se, 'size':3}), [0, 1]), 
pf.GeneralFeature(op.OpeningFunctor({'neighborhood':neighborhood_se, 'size':4}), [0, 1]), 
pf.GeneralFeature(op.OpeningFunctor({'neighborhood':neighborhood_se, 'size':5}), [0, 1]), 

pf.GeneralFeature(op.TopHatInvFunctor({'neighborhood':neighborhood_se, 'size':3}), [0, 1]),
pf.GeneralFeature(op.TopHatInvFunctor({'neighborhood':neighborhood_se, 'size':5}), [0, 1]), 
pf.GeneralFeature(op.TopHatInvFunctor({'neighborhood':neighborhood_se, 'size':10}), [0, 1]), 

pf.GeneralFeature(op.MorphologicalGradientFunctor({'neighborhood':neighborhood_se, 'size':1}), [0, 1]), 
pf.GeneralFeature(op.MorphologicalGradientFunctor({'neighborhood':neighborhood_se, 'size':2}), [0, 1]), 
pf.GeneralFeature(op.MorphologicalGradientFunctor({'neighborhood':neighborhood_se, 'size':3}), [0, 1]), 
pf.GeneralFeature(op.MorphologicalGradientFunctor({'neighborhood':neighborhood_se, 'size':4}), [0, 1]), 
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
]

for sigma in [1.6]:
    GS = pf.GeneralFeature(op.IlastikGaussianSmoothing({'sigma':5}), [0, 1])
    LG = pf.GeneralFeature(op.IlastikLaplacianOfGaussian({'scale':5}), [0, 1])
    GGM = pf.GeneralFeature(op.IlastikGaussianGradientMagnitude({'sigma':5}), [0, 1])
    DoG = pf.GeneralFeature(op.IlastikDifferenceOfGaussians({'sigma1':sigma * np.sqrt(2),  'sigma2':sigma / np.sqrt(2)}), [0, 1])
    ST1 = pf.GeneralFeature(op.IlastikStructureTensorEigenValues({'innerScale':sigma,  'outerScale':2,  'eigenvalueNumber': 0}), [0, 1]) 
    ST2 = pf.GeneralFeature(op.IlastikStructureTensorEigenValues({'innerScale':sigma,  'outerScale':2,  'eigenvalueNumber': 1}), [0, 1])
    HG1 = pf.GeneralFeature(op.IlastikHessianOfGaussianEigenvalues({'scale':sigma, 'eigenvalueNumber': 0}), [0, 1])
    HG2 = pf.GeneralFeature(op.IlastikHessianOfGaussianEigenvalues({'scale':sigma, 'eigenvalueNumber': 1}), [0, 1])
    
    pixel_features_list.append(GS)
    pixel_features_list.append(LG)
    pixel_features_list.append(GGM)
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
