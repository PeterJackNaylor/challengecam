# -*- coding: cp1252 -*-
"""
Description:
This file contains classes  SPCSOperator and GeneralFeature which enable respectively to compute 
the "features" images and the data matrix associated for classification of pixels. 
These classes are to be used when dealing with features whose computational support is not a superpixel.
Author: MoSeS-learning project: Vaïa Machairas, Etienne Decencière, Thomas Walter.
Creation date: 2015-08-24
"""

##---------------------------------------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------------------------------------
import pdb
import numpy as np
import smilPython as sp
import useful_functions as uf
##---------------------------------------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------------------------------------
class SPCSOperator(object):
    """
    Classe intermédiaire qui permet de calculer les images contenant les valeurs après application
    de l'opérateur.
    """
    def __init__(self, operator_functor, channels_list):
        """
        Constructeur de la classe.
        Inputs:
        operator_functor (functor)
        channels_list (list)
        """
        self._operator_functor = operator_functor
        self._channels_list = channels_list

    def __call__(self, original_image):
        """
        Output:
        dictionnaire contenant pour chaque canal choisi l'image transformée après application de l'opérateur.
        """       
        ### 
        dic_im_res={}
        #pdb.set_trace()
        if original_image.getTypeAsString()=="RGB":
            if self._channels_list == None:
                self._channels_list = [0,1,2]
            image_slices = sp.Image()
            sp.splitChannels(original_image, image_slices)
            for i in self._channels_list:
                image_transformed = sp.Image(image_slices.getSlice(0))
                self._operator_functor(image_slices.getSlice(i), image_transformed)
                dic_im_res[i] = image_transformed
        elif original_image.getTypeAsString()=="UINT8" or original_image.getTypeAsString()=="UINT16":
            self._channels_list = [0]
            image_transformed = sp.Image(original_image)
            self._operator_functor(original_image, image_transformed)
            dic_im_res[0] =  image_transformed
        else:
            raise TypeError('pb')

        return dic_im_res

    def get_name(self):
        list_names = []
        for i in self._channels_list:
            list_names +=[self._operator_functor.get_name()+"_for_channel_" + str(i)]
        return list_names

##---------------------------------------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------------------------------------
class GeneralFeature(SPCSOperator):
    def __init__(self, operator_functor,  channels_list):
        SPCSOperator.__init__(self, operator_functor, channels_list)

    def __call__(self, original_image):
        dic_sp_val = SPCSOperator.__call__(self, original_image)
        length = len(dic_sp_val.keys())
        vals = np.array([])
        X_feature_t = np.array([])
        for elem in dic_sp_val.keys():
            array_vals = np.ravel(dic_sp_val[elem].getNumArray())
            X_feature_t = uf.my_concatenation(X_feature_t, array_vals)
        return X_feature_t
