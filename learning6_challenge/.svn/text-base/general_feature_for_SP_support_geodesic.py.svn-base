# -*- coding: cp1252 -*-
"""
Description:
This file contains classes  SPCSGeodesicOperator and GeodesicGeneralFeature which enable respectively to compute 
the "features" images and the data matrix associated for classification of UC (units of classification). 
These classes are to be used when dealing with features whose computational support is the superpixel, where operators are applied in a geodesic manner.
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
class SPCSGeodesicOperator(object):
    """
    Classe intermédiaire qui permet de calculer les images de superpixels contenant les valeurs après application
    de l'opérateur de façon géodésique.
    """
    def __init__(self, operator_functor, channels_list, spp_method):
        """
        Constructeur de la classe.
        Inputs:
        operator_functor (functor): les opérateurs s'appliquent sur des images en NdG et doivent impérativement sortir qu'une seule "valeur" (ex: val, histo, etc).
        channels_list (list)
        spp_method (functor)
        """
        self._operator_functor = operator_functor
        self._channels_list = channels_list
        self._spp_method = spp_method
        
    def __call__(self, original_image):
        """
        Plusieurs étapes:
        1) calculer les superpixels de l'image originale
        2) calculer le dictionnaire des imagettes de superpixels
        3) appliquer l'opérateur sur chacune de ces imagettes
        4) intégrer sur le superpixel pour n'avoir qu'une seule valeur (si besoin)
        --> inclus dans l'opérateur
        5) calculer la nouvelle image entière des superpixels,  où cette fois-ci la valeur de chaque SP n'est pas son label mais celle calculée en 4. 
        [Note: plusieurs images si plusieurs cannaux sélectionnés dans channels_list. Output: dictionnaire de ces images finales.]
        --> output plut^ot un dictionnaire car pas besoin des images.
        
        
        Ouput:
        dic_inter: un dictionnaire tel que:
                - chaque clé est un numéro de superpixel ex: i
                - chaque valeur est un dictionnaire associé au superpixel i, contenant pour chaque cannal j (clés) la valeur du feature. 
        """
        
        ### Etape 1: calcul des SP
        image_sp = self._spp_method(original_image)
        blobs = sp.computeBlobs(image_sp)
        barys = sp.measBarycenters(image_sp,  blobs)
        
        ### Etape 2 :  cacul du dictionnaire intermédiaire
        
        ### Inititation listes:
        if original_image.getTypeAsString()=="RGB":
            if self._channels_list == None:
                self._channels_list = [0,1,2]
        elif original_image.getTypeAsString()=="UINT8" or original_image.getTypeAsString()=="UINT16":
            self._channels_list = [0]
        else:
            raise TypeError('pb')
        dic_final = {}
        for i in self._channels_list:
            dic_final[i] = [] ## cette liste contiendra la valeur pour chaque superpixel
        ###
        dic_inter = {}
        nb_sp = len(blobs.keys()) ## nombre de superpixels
        bboxes_coord = sp.measBoundBoxes(image_sp) ## dictionnary of coordinates of the two points (top left, bottom right) of each bounding box.
        sim_sp = sp.Image(image_sp)## pour garder temporairement l'imagette du superpixel
        
        for elem in range(nb_sp):
            elem += 1
            sp.crop(image_sp, bboxes_coord[elem][0],  bboxes_coord[elem][1],  bboxes_coord[elem][2] - bboxes_coord[elem][0] + 1,  bboxes_coord[elem][3] - bboxes_coord[elem][1] + 1,  sim_sp)
            sp.subNoSat(sim_sp,  elem,  sim_sp)
            sp.test(sim_sp,  0, 65535, sim_sp) ## imagette masque du superpixel i
#            sim_sp.save('imagettes/bin_mask_'+str(elem)+'.png')
#            if sim_sp.getSize()[0] == 1 and sim_sp.getSize()[1] == 1 :
#                print "sup_" + str(elem) +"pos_" + str(bboxes_coord[elem][0]) + "_" + str(bboxes_coord[elem][1])
#                image_sp.save("essais/sup_" + str(elem) +"pos_" + str(bboxes_coord[elem][0]) + "_" + str(bboxes_coord[elem][1]) + "_SP.png")
#                original_image.save("essais/sup_" + str(elem) +"pos_" + str(bboxes_coord[elem][0]) + "_" + str(bboxes_coord[elem][1]) + "_orig.png")
            if original_image.getTypeAsString()=="RGB":
                image_slices = sp.Image()
                sp.splitChannels(original_image, image_slices)
                dic_orig_slices={}
                for i in self._channels_list:
                    sim_orig_slice= sp.Image(image_slices.getSlice(i))
                    sp.crop(image_slices.getSlice(i), bboxes_coord[elem][0],  bboxes_coord[elem][1],  bboxes_coord[elem][2] - bboxes_coord[elem][0] + 1,  bboxes_coord[elem][3] - bboxes_coord[elem][1] + 1,  sim_orig_slice)
                    sp.add(sim_orig_slice, 1, sim_orig_slice)
                    sp.test(sim_sp>0, sim_orig_slice, 0, sim_orig_slice)
                    dic_orig_slices[i] = sim_orig_slice
                    #sim_orig_slice.save('imagettes/orig_slice_'+str(i)+'_for_sup_'+str(elem)+'.png')
            elif original_image.getTypeAsString()=="UINT8" or original_image.getTypeAsString()=="UINT16":
                dic_orig_slices={}
                sim_orig_slice= sp.Image(original_image)
                sp.crop(original_image, bboxes_coord[elem][0],  bboxes_coord[elem][1],  bboxes_coord[elem][2] - bboxes_coord[elem][0] + 1,  bboxes_coord[elem][3] - bboxes_coord[elem][1] + 1,  sim_orig_slice)
                sp.add(sim_orig_slice, 1, sim_orig_slice)
                sp.test(sim_sp>0, sim_orig_slice, 0, sim_orig_slice)
                dic_orig_slices[0] = sim_orig_slice
            dic_inter[elem] = (sim_sp,  dic_orig_slices,  barys[elem])
            
            
        ### Etape 3: application de l'opérateur sur chaque imagette:
            dic_elem_val_={}
            for i in self._channels_list:
                dic_final[i] += [self._operator_functor(dic_orig_slices[i])]
        return dic_final
       
    def get_name(self):
        list_names = []
        for i in self._channels_list:
            list_names +=[self._operator_functor.get_name()+"_"+self._spp_method.get_name()+"_for_channel_" + str(i)]
        return list_names
##---------------------------------------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------------------------------------
class GeneralFeatureGeodesic(SPCSGeodesicOperator):
    def __init__(self, operator_functor, channels_list, spp_method,  uc):
        SPCSGeodesicOperator.__init__(self, operator_functor, channels_list, spp_method)
        self._uc = uc
        
    def __call__(self,  original_image):
        dic_channels_spvals = SPCSGeodesicOperator.__call__(self, original_image)
        vals = np.array([])
        X_feature_t = np.array([])

        for elem in dic_channels_spvals.keys(): ##elem: numéro d'un cannal
            vals_list =[dic_channels_spvals[elem][i] for i in range(len(dic_channels_spvals[elem]))] ## i : numéro d'un superpixel
            array_vals = np.array(vals_list)
            if self._uc == "pixel":
                image_sp = self._spp_method(original_image)
                arr_pix_sp_labels = np.ravel(image_sp.getNumArray())
                arr_pix_sp = [vals_list[arr_pix_sp_labels[j]-1] for j in range(len(arr_pix_sp_labels))]
                array_vals = np.array(arr_pix_sp)
            X_feature_t = uf.my_concatenation(X_feature_t, array_vals) 
            
        return X_feature_t
        
    def get_name(self):
        list_names = SPCSGeodesicOperator.get_name(self)
        return list_names
##---------------------------------------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------------------------------------

