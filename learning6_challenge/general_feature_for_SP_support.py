# -*- coding: cp1252 -*-
"""
Description:
This file contains classes  SPCSOperator and GeneralFeature which enable respectively to compute 
the "features" images and the data matrix associated for classification of UC (units of classification). 
These classes are to be used when dealing with features whose computational support is the superpixel.
Author: MoSeS-learning project: Vaïa Machairas, Etienne Decencière, Thomas Walter.
Creation date: 2015-06
"""

##---------------------------------------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------------------------------------
import pdb
import numpy as np
import smilPython as sp
import useful_functions as uf
##---------------------------------------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------------------------------------
def apply_with_integrator(vals_map, spp_lab, integrator_code):
    """
    Permet de calculer la valeur intégrée de chaque superpixel
    et de créer l'image de superpixels associée.
    """
    im_out = sp.Image(spp_lab)
    blobs = sp.computeBlobs(spp_lab)
    myLUT =  sp.Map_UINT16_UINT16()
    if integrator_code == "mean" or integrator_code == "std":
        if integrator_code == "mean":
            choice = 0
        else:
            choice = 1
        for lbl in blobs.keys():
            myLUT[lbl] = int(vals_map[lbl][choice])
    else:
        for lbl in blobs.keys():
            myLUT[lbl] = int(vals_map[lbl])
    sp.applyLookup(spp_lab, myLUT, im_out)
    return im_out
##---------------------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------------------------- 
class SPCSOperator(object):
    """
    Classe intermédiaire qui permet de calculer les images de superpixels contenant les valeurs après application
    de l'opérateur.
    """
    def __init__(self, operator_functor, channels_list, integrator, spp_method):
        """
        Constructeur de la classe.
        Inputs:
        operator_functor (functor)
        channels_list (list)
        integrator (string)
        spp_method (functor)
        """
        self._operator_functor = operator_functor
        self._channels_list = channels_list
        self._integrator = integrator
        self._spp_method = spp_method

    def __call__(self, original_image):
        """
        Output:
        dictionnaire contenant pour chaque canal choisi l'image de superpixels,
        avec une seule valeur par superpixel.
        """
        image_sp = self._spp_method(original_image)
        blobs = sp.computeBlobs(image_sp)
        dict_integrator = {
            "mean": sp.measMeanVals,
            "std": sp.measMeanVals,
            "mode": sp.measModeVals,
            "min": sp.measMinVals,
            "max": sp.measMaxVals,
            "vol": sp.measVolumes,
            }        
        ### 
        dic_im_res={}
        if original_image.getTypeAsString()=="RGB":
            if self._channels_list == None:
                self._channels_list = [0,1,2]
            image_slices = sp.Image()
            sp.splitChannels(original_image, image_slices)
            image_transformed = sp.Image(image_slices.getSlice(0))
            for i in self._channels_list:
                self._operator_functor(image_slices.getSlice(i), image_transformed)
                vals_map = dict_integrator[self._integrator](image_transformed, blobs)
                im_out = apply_with_integrator(vals_map, image_sp, self._integrator)
                dic_im_res[i] = im_out
        elif original_image.getTypeAsString()=="UINT8" or original_image.getTypeAsString()=="UINT16":
            self._channels_list = [0]
            image_transformed = sp.Image(original_image)
            self._operator_functor(original_image, image_transformed)
            vals_map = dict_integrator[self._integrator](image_transformed, blobs)
            im_out = apply_with_integrator(vals_map, image_sp, self._integrator)
            dic_im_res[0] =  im_out
        else:
            raise TypeError('pb')

        return dic_im_res

    def get_name(self):
        list_names = []
        for i in self._channels_list:
            list_names +=[self._operator_functor.get_name()+"_"+self._spp_method.get_name()+"_"+ self._integrator + "_for_channel_" + str(i)]
        return list_names

##---------------------------------------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------------------------------------
class GeneralFeature(SPCSOperator):
    def __init__(self, operator_functor,  channels_list, integrator, spp_method, uc):
        SPCSOperator.__init__(self, operator_functor, channels_list, integrator, spp_method)
        self._uc = uc

    def __call__(self, original_image):
        dic_sp_val = SPCSOperator.__call__(self, original_image)
        vals = np.array([])
        cache_blobs = False
        X_feature_t = np.array([])
        for elem in dic_sp_val.keys():
            if self._uc == "superpixel":
                image_sp = self._spp_method(original_image)
                blobs = sp.computeBlobs(image_sp)
                list_vals = []
#                if cache_blobs == False:
#                    blobs = sp.computeBlobs(dic_sp_val[elem])
#                    cache_blobs = True
                the_vals = sp.measMinVals(dic_sp_val[elem],  blobs)
                for lbl in blobs.keys():
                    list_vals +=[int(the_vals[lbl])]
                array_vals = np.array(list_vals)
                
            elif self._uc == "pixel":
                array_vals = np.ravel(dic_sp_val[elem].getNumArray())
            else:
                raise TypeError("uc (string) should be equal to superpixel or pixel")
            X_feature_t = uf.my_concatenation(X_feature_t, array_vals)
            
        return X_feature_t


    def get_name(self):
        list_names = SPCSOperator.get_name(self)
        return list_names
##---------------------------------------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------------------------------------
