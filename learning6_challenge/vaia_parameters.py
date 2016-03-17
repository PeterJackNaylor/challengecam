# -*- coding: cp1252 -*-
"""
Description: learning script parameters
Author: MoSeS-learning project: Vaïa Machairas, Etienne Decencière, Thomas Walter.
Creation date: 2015-04-29
"""
import os
import pdb
from getpass import getuser

import sklearn.ensemble as ens

import segm_db_access as sdba

import useful_functions as uf
import spp_functors as spp
import op_functors as op
import op_functors_geodesic as og
import general_feature_for_SP_support as gf
import general_feature_for_pixel_support as pf
import general_feature_for_SP_support_geodesic as geo
import general_feature_for_pixel_support_geodesic as pfg
import cytomine_window as cw

##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

TRAINING = True
PREDICTION = TRAINING
EVALUATION = True
COMPARISON = False

##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Definition of the database:
#base_path = "/home/vaiamachairas/Documents/databases/essaisbase"
#base = "typeWeiz2"
#base = "WeizmannHorse"
#base = "SmallWeizmannHorse"
#base = "OneCellPerImage"
#base = "WeizmannSingleScale"
#base = "SmallColorWeizmannHorse"
##base = "Coelho"
#base = "ThomasIncomplete"
##base = "LittleThomasIncomplete"
##base = "ICIAR2010_cell"
##base = "ICIAR2010_wall"
#base = "LOreal"
#
#if  base == "typeWeiz2":
#    db_server = sdba.SegmDataBaseStandard("/home/vaiamachairas/Documents/databases/images_simples/baseTypeWeiz2/rectangles/")
#if  base == "typeCell2":
#    db_server = sdba.SegmDataBaseStandard("/home/vaiamachairas/Documents/databases/images_simples/baseTypeCell2/disks/")
#if base == "SmallColorWeizmannHorse":
#    db_server = sdba.SegmDataBaseWeizmannHorseSingleScale("/home/vaiamachairas/Documents/databases/WHD_small_color/")
#if base == "SmallWeizmannHorse":
#    db_server = sdba.SegmDataBaseStandard("/home/vaiamachairas/Documents/databases/WHD_small/")
#if base == "WeizmannHorse":
#    db_server = sdba.SegmDataBaseStandard("/home/vaiamachairas/Documents/databases/WHD/")
#if base == "Coelho":
#    first=0
#    last = 20
#    db_server = sdba.SegmDataBaseCoelhoGNF("/home/vaiamachairas/Documents/databases/Coelho2009_ISBI_NuclearSegmentation/data")
#    #db_server = sdba.SegmDataBaseCoelhoIC100("/home/decencie/images/bdd_segmentation/Coelho2009_ISBI_NuclearSegmentation/data")
#if base == "ThomasIncomplete":
#    db_server = sdba.SegmThomasCellsIncomplete("/home/vaiamachairas/Documents/databases/Thomas_seg/match/")
#if base == "LittleThomasIncomplete":
#    db_server = sdba.SegmThomasCellsIncomplete("/home/vaiamachairas/Documents/databases/Thomas_seg/little_match16/")
#if base == "ICIAR2010_cell":
#    db_server = sdba.SegmICIAR2010Cell("/home/vaiamachairas/Documents/databases/ICIAR2010/cell/")
#if base == "ICIAR2010_wall":
#    db_server = sdba.SegmICIAR2010Cell("/home/vaiamachairas/Documents/databases/ICIAR2010/wall/")
#if base == "LOreal":
#    db_server = sdba.SegmLOreal("/home/vaiamachairas/Documents/databases/LOreal/new/split8/")
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## MAIN parameters #############################

## Superpixel functors
wp_uc_sup = spp.WaterpixelsFunctor({"step":10, "k":4, "filter_ori":False})
wp_uc_sup = spp.SLICSuperpixelsFunctor({"nb_regions": 300, "m": 15})
wp1 = spp.WaterpixelsFunctor({"step":15, "k":4, "filter_ori":True})
wp2 = spp.WaterpixelsFunctor({"step":20, "k":4, "filter_ori":True})
wp3 = spp.WaterpixelsFunctor({"step":50, "k":4, "filter_ori":True})
wp4 = spp.WaterpixelsFunctor({"step":100, "k":4, "filter_ori":True})
slic1 = spp.SLICSuperpixelsFunctor({"nb_regions": 300, "m": 15})

win1 = spp.WindowFunctor({'size': 20})
#wp1 = spp.SLICSuperpixelsFunctor({"nb_regions": 650, "m": 15})

## Number of samples if needed:
NB_SAMPLES = 100000
#NB_SAMPLES = None

## Some common parameters:
neighborhood = 'V8'
size = 2
se = uf.set_structuring_element(neighborhood, size)

## Choice of uc:
pixel_classif = True
spp_classif = not pixel_classif

## List of features:
pixel_features_list = [
###### Cytomine Window and superpixels:
#cw.CytomineWindow(cw.FromImageToMatrix_Window({'window_size':2,  'substitution_value':0}),  [0, 1, 2]),
#cw.CytomineWindow(cw.FromImageToMatrix_Window({'window_size':1,  'substitution_value':0}),  [0, 1, 2]),
#cw.CytomineSuperpixelBin(cw.FromImageToMatrix_Window({'window_size':1,  'substitution_value':0}), wp1 ,  [0, 1, 2]), 
#cw.CytomineSuperpixelBin(cw.FromImageToMatrix_Window({'window_size':1,  'substitution_value':0}), wp2 ,  [0, 1, 2]), 
#cw.CytomineSuperpixelBin(cw.FromImageToMatrix_Window({'window_size':1,  'substitution_value':0}), wp3 ,  [0, 1, 2]), 
#cw.CytomineSuperpixelBin(cw.FromImageToMatrix_Window({'window_size':1,  'substitution_value':0}), wp4 ,  [0, 1, 2]), 
######
#pf.GeneralFeature(op.IdentityFunctor({}), [0, 1, 2]), 
#pf.GeneralFeature(op.IdentityFunctor({}), [0, 1, 2]),
pf.GeneralFeature(op.IdentityFunctor({}), [0, 1, 2]), 
gf.GeneralFeature(op.IdentityFunctor({}), [0, 1, 2],  'mean',  wp1, 'pixel'), 
gf.GeneralFeature(op.IdentityFunctor({}), [0, 1, 2],  'mean',  wp2, 'pixel'), 
gf.GeneralFeature(op.IdentityFunctor({}), [0, 1, 2],  'mean',  wp3, 'pixel'), 
gf.GeneralFeature(op.IdentityFunctor({}), [0, 1, 2],  'mean',  wp4, 'pixel'), 


##gf.GeneralFeature(op.IdentityFunctor({}), [0, 1, 2],  'mean',  win1, 'pixel'), 
##gf.GeneralFeature(op.IdentityFunctor({}), [0, 1, 2],  'mean',  win1, 'pixel'), 

#pf.GeneralFeature(op.IdentityFunctorW({'window_size':6}), [0, 1, 2]),
#pf.GeneralFeature(op.IdentityFunctorW({'window_size':7}), [0, 1, 2]),
#pf.GeneralFeature(op.IdentityFunctorW({'window_size':10}), [0, 1, 2]), 
#pf.GeneralFeature(op.IdentityFunctorW({'window_size':12}), [0, 1, 2]),

### geodesic, support = spix (SAF)
#geo.GeneralFeatureGeodesic(og.GeodesicErosionFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicDilationFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicClosingFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicTopHatFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicMorphoGradientFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp1,  'pixel'), 
#
#geo.GeneralFeatureGeodesic(og.GeodesicErosionFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp2,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicDilationFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp2,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp2,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicClosingFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp2,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicTopHatFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp2,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicMorphoGradientFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp2,  'pixel'), 
#
#geo.GeneralFeatureGeodesic(og.GeodesicErosionFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp3,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicDilationFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp3,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp3,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicClosingFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp3,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicTopHatFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp3,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicMorphoGradientFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp3,  'pixel'), 
#
#geo.GeneralFeatureGeodesic(og.GeodesicErosionFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp4,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicDilationFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp4,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp4,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicClosingFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp4,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicTopHatFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp4,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicMorphoGradientFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  wp4,  'pixel'), 


### geodesic, support = window partition
#geo.GeneralFeatureGeodesic(og.GeodesicErosionFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  win1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicDilationFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  win1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicOpeningFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  win1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicClosingFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  win1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicTopHatFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  win1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.GeodesicMorphoGradientFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2],  win1,  'pixel'), 

#### geodesic, support = window
#pfg.GeneralFeatureGeodesicPixel(og.GeodesicErosionFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2], {"size": 6}), 
#pfg.GeneralFeatureGeodesicPixel(og.GeodesicDilationFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2], {"size": 6}), 
#pfg.GeneralFeatureGeodesicPixel(og.GeodesicOpeningFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2], {"size": 6}), 
#pfg.GeneralFeatureGeodesicPixel(og.GeodesicClosingFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2], {"size": 6}), 
#pfg.GeneralFeatureGeodesicPixel(og.GeodesicTopHatFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2], {"size": 6}), 
#pfg.GeneralFeatureGeodesicPixel(og.GeodesicMorphoGradientFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2], {"size": 6}), 
#
#pfg.GeneralFeatureGeodesicPixel(og.GeodesicErosionFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2], {"size": 7}), 
#pfg.GeneralFeatureGeodesicPixel(og.GeodesicDilationFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2], {"size": 7}), 
#pfg.GeneralFeatureGeodesicPixel(og.GeodesicOpeningFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2], {"size": 7}), 
#pfg.GeneralFeatureGeodesicPixel(og.GeodesicClosingFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2], {"size": 7}), 
#pfg.GeneralFeatureGeodesicPixel(og.GeodesicTopHatFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2], {"size": 7}), 
#pfg.GeneralFeatureGeodesicPixel(og.GeodesicMorphoGradientFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2], {"size": 7}), 
#
#pfg.GeneralFeatureGeodesicPixel(og.GeodesicErosionFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2], {"size": 10}), 
#pfg.GeneralFeatureGeodesicPixel(og.GeodesicDilationFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2], {"size": 10}), 
#pfg.GeneralFeatureGeodesicPixel(og.GeodesicOpeningFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2], {"size": 10}), 
#pfg.GeneralFeatureGeodesicPixel(og.GeodesicClosingFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2], {"size": 10}), 
#pfg.GeneralFeatureGeodesicPixel(og.GeodesicTopHatFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2], {"size": 10}), 
#pfg.GeneralFeatureGeodesicPixel(og.GeodesicMorphoGradientFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2], {"size": 10}), 
#
#pfg.GeneralFeatureGeodesicPixel(og.GeodesicErosionFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2], {"size": 12}), 
#pfg.GeneralFeatureGeodesicPixel(og.GeodesicDilationFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2], {"size": 12}), 
#pfg.GeneralFeatureGeodesicPixel(og.GeodesicOpeningFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2], {"size": 12}), 
#pfg.GeneralFeatureGeodesicPixel(og.GeodesicClosingFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2], {"size": 12}), 
#pfg.GeneralFeatureGeodesicPixel(og.GeodesicTopHatFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2], {"size": 12}), 
#pfg.GeneralFeatureGeodesicPixel(og.GeodesicMorphoGradientFunctor({'neighborhood': 'V4', 'size':5,  'integrator': 'mean'}),  [0, 1, 2], {"size": 12}), 


##


#### non_geodesic, support = pixel
#pf.GeneralFeature(op.ErosionFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2]), 
#pf.GeneralFeature(op.DilationFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2]), 
#pf.GeneralFeature(op.OpeningFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2]), 
#pf.GeneralFeature(op.ClosingFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2]), 
#pf.GeneralFeature(op.TopHatFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2]), 
#pf.GeneralFeature(op.MorphologicalGradientFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2]), 

#### non_geodesic, support = spix (SAF)
#gf.GeneralFeature(op.ErosionFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  wp1, 'pixel'),
#gf.GeneralFeature(op.DilationFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  wp1, 'pixel'),
#gf.GeneralFeature(op.OpeningFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  wp1, 'pixel'),
#gf.GeneralFeature(op.ClosingFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  wp1, 'pixel'),
#gf.GeneralFeature(op.TopHatFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  wp1, 'pixel'),
#gf.GeneralFeature(op.MorphologicalGradientFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  wp1, 'pixel'),
#
#
#gf.GeneralFeature(op.ErosionFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  wp2, 'pixel'),
#gf.GeneralFeature(op.DilationFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  wp2, 'pixel'),
#gf.GeneralFeature(op.OpeningFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  wp2, 'pixel'),
#gf.GeneralFeature(op.ClosingFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  wp2, 'pixel'),
#gf.GeneralFeature(op.TopHatFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  wp2, 'pixel'),
#gf.GeneralFeature(op.MorphologicalGradientFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  wp2, 'pixel'),
#
#gf.GeneralFeature(op.ErosionFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  wp3, 'pixel'),
#gf.GeneralFeature(op.DilationFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  wp3, 'pixel'),
#gf.GeneralFeature(op.OpeningFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  wp3, 'pixel'),
#gf.GeneralFeature(op.ClosingFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  wp3, 'pixel'),
#gf.GeneralFeature(op.TopHatFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  wp3, 'pixel'),
#gf.GeneralFeature(op.MorphologicalGradientFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  wp3, 'pixel'),
#
#gf.GeneralFeature(op.ErosionFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  wp4, 'pixel'),
#gf.GeneralFeature(op.DilationFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  wp4, 'pixel'),
#gf.GeneralFeature(op.OpeningFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  wp4, 'pixel'),
#gf.GeneralFeature(op.ClosingFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  wp4, 'pixel'),
#gf.GeneralFeature(op.TopHatFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  wp4, 'pixel'),
#gf.GeneralFeature(op.MorphologicalGradientFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  wp4, 'pixel'),


### non_geodesic, support =  window
#pf.GeneralFeature(op.ErosionFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 10}), [0, 1, 2]), 
#pf.GeneralFeature(op.DilationFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 10}), [0, 1, 2]), 
#pf.GeneralFeature(op.OpeningFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 10}), [0, 1, 2]), 
#pf.GeneralFeature(op.ClosingFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 10}), [0, 1, 2]), 
#pf.GeneralFeature(op.TopHatFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 10}), [0, 1, 2]), 
#pf.GeneralFeature(op.MorphologicalGradientFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 10}), [0, 1, 2]), 
#
#pf.GeneralFeature(op.ErosionFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 6}), [0, 1, 2]), 
#pf.GeneralFeature(op.DilationFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 6}), [0, 1, 2]), 
#pf.GeneralFeature(op.OpeningFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 6}), [0, 1, 2]), 
#pf.GeneralFeature(op.ClosingFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 6}), [0, 1, 2]), 
#pf.GeneralFeature(op.TopHatFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 6}), [0, 1, 2]), 
#pf.GeneralFeature(op.MorphologicalGradientFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 6}), [0, 1, 2]), 
#
#pf.GeneralFeature(op.ErosionFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 7}), [0, 1, 2]), 
#pf.GeneralFeature(op.DilationFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 7}), [0, 1, 2]), 
#pf.GeneralFeature(op.OpeningFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 7}), [0, 1, 2]), 
#pf.GeneralFeature(op.ClosingFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 7}), [0, 1, 2]), 
#pf.GeneralFeature(op.TopHatFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 7}), [0, 1, 2]), 
#pf.GeneralFeature(op.MorphologicalGradientFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 7}), [0, 1, 2]), 
#
#pf.GeneralFeature(op.ErosionFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 10}), [0, 1, 2]), 
#pf.GeneralFeature(op.DilationFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 10}), [0, 1, 2]), 
#pf.GeneralFeature(op.OpeningFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 10}), [0, 1, 2]), 
#pf.GeneralFeature(op.ClosingFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 10}), [0, 1, 2]), 
#pf.GeneralFeature(op.TopHatFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 10}), [0, 1, 2]), 
#pf.GeneralFeature(op.MorphologicalGradientFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 10}), [0, 1, 2]), 
#
#pf.GeneralFeature(op.ErosionFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 12}), [0, 1, 2]), 
#pf.GeneralFeature(op.DilationFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 12}), [0, 1, 2]), 
#pf.GeneralFeature(op.OpeningFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 12}), [0, 1, 2]), 
#pf.GeneralFeature(op.ClosingFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 12}), [0, 1, 2]), 
#pf.GeneralFeature(op.TopHatFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 12}), [0, 1, 2]), 
#pf.GeneralFeature(op.MorphologicalGradientFunctorW({'neighborhood':'V4', 'size':5,  'window_size': 12}), [0, 1, 2]), 

### non_geodesic, support = window partition
#gf.GeneralFeature(op.ErosionFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  win1, 'pixel'),
#gf.GeneralFeature(op.DilationFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  win1, 'pixel'),
#gf.GeneralFeature(op.OpeningFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  win1, 'pixel'),
#gf.GeneralFeature(op.ClosingFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  win1, 'pixel'),
#gf.GeneralFeature(op.TopHatFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  win1, 'pixel'),
#gf.GeneralFeature(op.MorphologicalGradientFunctor({'neighborhood':'V4', 'size':5}), [0, 1, 2],  'mean',  win1, 'pixel'),

##
##### Texture
##### Haralick
#### support = spix (SAF)
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'AngularSecondMoment'}), [0, 1, 2],  wp2, 'pixel'),
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'Contrast'}), [0, 1, 2],  wp2, 'pixel'),
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'Correlation'}), [0, 1, 2],  wp2, 'pixel'),
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'SumofSquaresVariance'}), [0, 1, 2],  wp2, 'pixel'),
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'InverseDifferenceMoment'}), [0, 1, 2],  wp2, 'pixel'),
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'SumAverage'}), [0, 1, 2],  wp2, 'pixel'),
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'SumVariance'}), [0, 1, 2],  wp2, 'pixel'),
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'SumEntropy'}), [0, 1, 2],  wp2, 'pixel'),
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'Entropy'}), [0, 1, 2],  wp2, 'pixel'),
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'DifferenceVariance'}), [0, 1, 2],  wp2, 'pixel'),
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'DifferenceEntropy'}), [0, 1, 2],  wp2, 'pixel'),
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'InformationMeasureofCorrelation1'}), [0, 1, 2],  wp2, 'pixel'),
geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'InformationMeasureofCorrelation2'}), [0, 1, 2],  wp2, 'pixel'),

#### support = window partition
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'AngularSecondMoment'}), [0, 1, 2],  win1, 'pixel'),
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'Contrast'}), [0, 1, 2],  win1, 'pixel'),
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'Correlation'}), [0, 1, 2],  win1, 'pixel'),
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'SumofSquaresVariance'}), [0, 1, 2],  win1, 'pixel'),
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'InverseDifferenceMoment'}), [0, 1, 2],  win1, 'pixel'),
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'SumAverage'}), [0, 1, 2],  win1, 'pixel'),
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'SumVariance'}), [0, 1, 2],  win1, 'pixel'),
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'SumEntropy'}), [0, 1, 2],  win1, 'pixel'),
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'Entropy'}), [0, 1, 2],  win1, 'pixel'),
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'DifferenceVariance'}), [0, 1, 2],  win1, 'pixel'),
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'DifferenceEntropy'}), [0, 1, 2],  win1, 'pixel'),
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'InformationMeasureofCorrelation1'}), [0, 1, 2],  win1, 'pixel'),
#geo.GeneralFeatureGeodesic(og.HaralickFeature({'direction': 'all',  'feature_name': 'InformationMeasureofCorrelation2'}), [0, 1, 2],  win1, 'pixel'),

#### support = window
#pfg.GeneralFeatureGeodesicPixel(og.HaralickFeature({'direction': 'all',  'feature_name': 'AngularSecondMoment'}),  [0, 1, 2], {"size": 10}),
#pfg.GeneralFeatureGeodesicPixel(og.HaralickFeature({'direction': 'all',  'feature_name': 'Contrast'}),  [0, 1, 2], {"size": 10}),
#pfg.GeneralFeatureGeodesicPixel(og.HaralickFeature({'direction': 'all',  'feature_name': 'Correlation'}),  [0, 1, 2], {"size": 10}),
#pfg.GeneralFeatureGeodesicPixel(og.HaralickFeature({'direction': 'all',  'feature_name': 'SumofSquaresVariance'}),  [0, 1, 2], {"size": 10}),
#pfg.GeneralFeatureGeodesicPixel(og.HaralickFeature({'direction': 'all',  'feature_name': 'InverseDifferenceMoment'}),  [0, 1, 2], {"size": 10}),
#pfg.GeneralFeatureGeodesicPixel(og.HaralickFeature({'direction': 'all',  'feature_name': 'SumAverage'}),  [0, 1, 2], {"size": 10}),
#pfg.GeneralFeatureGeodesicPixel(og.HaralickFeature({'direction': 'all',  'feature_name': 'SumVariance'}),  [0, 1, 2], {"size": 10}),
#pfg.GeneralFeatureGeodesicPixel(og.HaralickFeature({'direction': 'all',  'feature_name': 'SumEntropy'}),  [0, 1, 2], {"size": 10}),
#pfg.GeneralFeatureGeodesicPixel(og.HaralickFeature({'direction': 'all',  'feature_name': 'DifferenceVariance'}),  [0, 1, 2], {"size": 10}),
#pfg.GeneralFeatureGeodesicPixel(og.HaralickFeature({'direction': 'all',  'feature_name': 'DifferenceEntropy'}),  [0, 1, 2], {"size": 10}),
#pfg.GeneralFeatureGeodesicPixel(og.HaralickFeature({'direction': 'all',  'feature_name': 'InformationMeasureofCorrelation1'}),  [0, 1, 2], {"size": 10}),
#pfg.GeneralFeatureGeodesicPixel(og.HaralickFeature({'direction': 'all',  'feature_name': 'InformationMeasureofCorrelation2'}),  [0, 1, 2], {"size": 10}),
##
##### LBP
#### support = spix (SAF)
geo.GeneralFeatureGeodesic(og.LBP_bin1({'radius': 2,  'points': 8, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp2,  'pixel'), 
geo.GeneralFeatureGeodesic(og.LBP_bin2({'radius': 2,  'points': 8, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp2,  'pixel'), 
geo.GeneralFeatureGeodesic(og.LBP_bin3({'radius': 2,  'points': 8, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp2,  'pixel'), 
geo.GeneralFeatureGeodesic(og.LBP_bin4({'radius': 2,  'points': 8, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp2,  'pixel'), 
geo.GeneralFeatureGeodesic(og.LBP_bin5({'radius': 2,  'points':8, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp2,  'pixel'), 
geo.GeneralFeatureGeodesic(og.LBP_bin6({'radius': 2,  'points': 8, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp2,  'pixel'), 

#geo.GeneralFeatureGeodesic(og.LBP_bin1({'radius': 3,  'points': 15, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.LBP_bin2({'radius': 3,  'points': 15, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.LBP_bin3({'radius': 3,  'points': 15, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.LBP_bin4({'radius': 3,  'points': 15, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.LBP_bin5({'radius': 3,  'points': 15, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.LBP_bin6({'radius': 3,  'points': 15, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp1,  'pixel'), 
#
#geo.GeneralFeatureGeodesic(og.LBP_bin1({'radius': 4,  'points': 15, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.LBP_bin2({'radius': 4,  'points': 15, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.LBP_bin3({'radius': 4,  'points': 15, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.LBP_bin4({'radius': 4,  'points': 15, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.LBP_bin5({'radius': 4,  'points': 15, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.LBP_bin6({'radius': 4,  'points': 15, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  wp1,  'pixel'), 


#### support = window partition
#geo.GeneralFeatureGeodesic(og.LBP_bin1({'radius': 3,  'points': 8, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  win1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.LBP_bin2({'radius': 3,  'points': 8, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  win1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.LBP_bin3({'radius': 3,  'points': 8, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  win1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.LBP_bin4({'radius': 3,  'points': 8, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  win1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.LBP_bin5({'radius': 3,  'points': 8, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  win1,  'pixel'), 
#geo.GeneralFeatureGeodesic(og.LBP_bin6({'radius': 3,  'points': 8, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2],  win1,  'pixel'), 

### support = window
#pfg.GeneralFeatureGeodesicPixel(og.LBP_bin1({'radius': 3,  'points': 8, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2], {"size": 10}),
#pfg.GeneralFeatureGeodesicPixel(og.LBP_bin2({'radius': 3,  'points': 8, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2], {"size": 10}),
#pfg.GeneralFeatureGeodesicPixel(og.LBP_bin3({'radius': 3,  'points': 8, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2], {"size": 10}),
#pfg.GeneralFeatureGeodesicPixel(og.LBP_bin4({'radius': 3,  'points': 8, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2], {"size": 10}),
#pfg.GeneralFeatureGeodesicPixel(og.LBP_bin5({'radius': 3,  'points': 8, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2], {"size": 10}),
#pfg.GeneralFeatureGeodesicPixel(og.LBP_bin6({'radius': 3,  'points': 8, 'ignore_zeros': True,  'preserve_shape': False}),  [0, 1, 2], {"size": 10}),

#### Contour
##geo.GeneralFeatureGeodesic(og.DiffIntContoursStd({'neighborhood':'V4'}), [0, 1, 2],  wp1,  'pixel'), 
##geo.GeneralFeatureGeodesic(og.DiffIntContoursStd({'neighborhood':'V4'}), [0, 1, 2],  wp2,  'pixel'), 
##geo.GeneralFeatureGeodesic(og.DiffIntContoursStd({'neighborhood':'V4'}), [0, 1, 2],  slic1,  'pixel'), 


]
superpixel_features_list = [
gf.GeneralFeature(op.IdentityFunctor({}), [0, 1, 2], 'mean', wp_uc_sup, 'superpixel'), 
gf.GeneralFeature(op.MorphologicalGradientFunctor({'neighborhood':'V4', 'size':10}), [0, 1, 2],  'mean',  wp_uc_sup,  'superpixel'), 
#geo.GeneralFeatureGeodesic(og.Haralick_AngularSecondMoment({'direction':1}), [0, 1, 2],  wp_uc_sup, 'superpixel'),
geo.GeneralFeatureGeodesic(og.DiffIntContoursStd({'neighborhood':'V8'}), [0, 1, 2],  wp_uc_sup,  'superpixel'), 
]
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Classifier:
out_of_bag_score = False  # permet de calculer une erreur d'apprentissage, mais coute cher en temps de calcul
myforest = ens.RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_leaf = 100, max_features='auto', bootstrap=True, n_jobs=4, random_state=42, oob_score=out_of_bag_score)
#myforest = ens.RandomForestClassifier(n_estimators=100, criterion='gini', max_features='auto', bootstrap=True, n_jobs=4, random_state=42, oob_score=out_of_bag_score)

##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Output directory:
#output_dir0 = os.path.join(db_server.get_input_dir(), "resultats/")
#output_dir = os.path.join(output_dir0, "test/")
#if os.path.isdir(output_dir0) is not True:
#    try:
#        os.mkdir(output_dir0)
#    except OSError:
#        print "Could not make output results dir"
#        raise
#if os.path.isdir(output_dir) is not True:
#    try:
#        os.mkdir(output_dir)
#    except OSError:
#        print "Could not make output results subset dir"
#        raise
