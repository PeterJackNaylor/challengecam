# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:05:29 2016

@author: naylor
"""

import pdb
import numpy as np
from debug_mode import debug_mess
import cPickle as pickle
import random
import smilPython as sm
import useful_functions as uf
import os
import openslide
from find_ROI import ROI
from image_classification import GetImage

class SegmDataBaseCommon(object):
    """Segmentation data base server common class."""

    def __init__(self, dir_in):
        self._im_dir = {}
        self._im_dir["general"] = dir_in
        self._segm_dir = {}
        self._res_dir = {}

    def get_input_dir(self):
        return self._im_dir["general"]

    def iter(self, code, first=None, last=None):
        for im_file in os.listdir(self._im_dir[code])[first:last]:
            #debug_mess("Processing file %s" % os.path.basename(im_file))
            outputs = self._im_and_segm(os.path.join(self._im_dir[code], im_file), self._segm_dir[code])
            yield outputs[0],  outputs[1],  outputs[2]

    def iter2(self, code):
        for im_file in os.listdir(self._im_dir[code]):
            #debug_mess("Processing file %s" % os.path.basename(im_file))
            outputs = self._im_and_segm2(os.path.join(self._im_dir[code], im_file),   self._segm_dir[code], self._res_dir[code])
            yield outputs[0],  outputs[1],  outputs[2]
            
    def train(self):
        raise NameError("Train set not available for this database")

    def val(self):
        raise NameError("Validation set not available for this database")

    def test(self):
        raise NameError("Test set not available for this database")

    def _im_and_segm(self, im_file, segm_dir):
        """Return  image and the corresponding segmentations.

        In this version, we suppose that there is a single segmentation for the input file,
        which has the same name and is found in the segmentation folder.

        Args:
           im_file: complete file name.
           segm_dir: folder containing the image segmentations.

        Returns:
           im: original image
           segm_list: list containing the corresponding GT segmentations
           file_name: original file name
        """
        #pdb.set_trace()
        im = sm.Image(im_file)
        segm_list = []
        file_name =  os.path.basename(im_file)
        print file_name
        im_segm = sm.Image(os.path.join(segm_dir, file_name))
        self.segm_post_process(im_segm)
        segm_list.append(im_segm)
        return im, segm_list, file_name
        
    def _im_and_segm2(self, im_file, segm_dir,  res_dir):
        """Return  image and the corresponding segmentations.

        In this version, we suppose that there is a single segmentation for the input file,
        which has the same name and is found in the segmentation folder.

        Args:
           im_file: complete file name.
           segm_dir: folder containing the image segmentations.

        Returns:
           im: original image
           segm_list: list containing the corresponding GT segmentations
           file_name: original file name
        """
        segm_list = []
        file_name =  os.path.basename(im_file)
        im_segm = sm.Image(os.path.join(segm_dir, file_name))
        self.segm_post_process(im_segm)
        segm_list.append(im_segm)
        im_pred = sm.Image(os.path.join(res_dir, file_name))
        return  segm_list, im_pred, file_name

    def nb_im(self, code):
        """Number of images in subbase given by *code*"""
        return len(os.listdir(self._im_dir[code]))

    def segm_post_process(self, im_segm):
        pass

#__init__  get_input_dir  iter iter2 train val test _im_and_segm _im_and_segm2 nb_im segm_post_process
def from_pil_to_sm_image(Pil_image,type_im='not GT'):
    matrix=np.array(Pil_image)
    dim=len(matrix.shape)
    if type_im=='not GT':
        if dim<3:
            if dim==2:
                print 'dim == 2 is unexepected...'
            image = sm.Image(matrix.shape[1], matrix.shape[0])
            im_array = image.getNumArray()
            im_array[:,:] = np.transpose(matrix)
        else:
            image = sm.Image("RGB",matrix.shape[1], matrix.shape[0], min(matrix.shape[2],3))  ###il faudrait la mettre en RGB!!!
            image_temp= sm.Image()
            sm.splitChannels(image,image_temp)
            im_array = image_temp.getNumArray()
            for i in range(min(matrix.shape[2],3)):
                im_array[:,:,i] = np.transpose(matrix[:,:,i])
            sm.mergeChannels(image_temp,image)
    else:
        if dim<3:        
            image = sm.Image(matrix.shape[1], matrix.shape[0])
            im_array = image.getNumArray()
            im_array[:,:] = np.transpose(matrix)
        else:
            image= sm.Image('UINT8',matrix.shape[1],matrix.shape[0])
            im_array = image.getNumArray()
            im_array[:,:] = np.transpose(matrix[:,:,0])
    return(image)


class SegmChallengeCamelyon16(SegmDataBaseCommon):
    """
    Base du challenge CAMELYON16 (ISBI16).
    """
    def __init__(self,  dir_in):
        SegmDataBaseCommon.__init__(self, dir_in)
        self._im_dir["train"] = os.path.join(dir_in, "images/train")
        self._im_dir["val"] = os.path.join(dir_in, "images/val")
        self._im_dir["test"] = os.path.join(dir_in, "images/test")
        self._segm_dir["train"] = os.path.join(dir_in, "GT/train")
        self._segm_dir["val"] = os.path.join(dir_in, "GT/val")
        self._segm_dir["test"] = os.path.join(dir_in, "GT/test")
        self._res_dir["train"] = os.path.join(dir_in, "resultats/train")
        self._res_dir["test"] = os.path.join(dir_in, "resultats/test")
        self._res_dir["val"] = os.path.join(dir_in, "resultats/val")

    def train_training(self):
        return self.iter_training("train")
    
    def train_final_prediction(self):
        return self.iter_final_prediction("train")

    def test_final_prediction(self):
        return self.iter_final_prediction("test")
        

##----------------------------------------------------------------
    def segm_post_process(self, im_segm):
        """
        This function enables to pre-process the GT for it to contain only labels {0,1} and in UINT8 format.
        """
        pdb.set_trace()
        sm.compare(im_segm, ">", 0, 1, 0, im_segm)   
##----------------------------------------------------------------
    def get_image(self, im_file):
        """
        Return, for a given slide which complete path_name is "im_file", a list whom each element is itself a list of two elements:
        - an imagette (smil image, RGB)
        - the name of this imagette
        
        Be careful, the whole slide must be processed later on, so the set of imagettes must cover the whole slide.

        Args:
           im_file: complete file name.
           segm_dir: folder containing the image segmentations.

        Returns:
            list_of_imagettes: list of size the number of imagettes selected for the given slide.
        """
        ROI_pos=ROI(im_file,ref_level=2,disk_size=4,thresh=220,black_spots=20,number_of_pixels_max=700000,verbose=False) ### RAJOUTER ARGUMENT EN OPTION
        file_name =  os.path.basename(im_file)
        [base_name, ext] = file_name.rsplit(".", 1)
        list_of_imagettes=[]
        for para in ROI_pos[0:2]: ##### change for a better imagettes' selection method___PREDICSTION--> Etienne's method
            outputs=[]
            outputs.append(from_pil_to_sm_image(GetImage(im_file,para))) 
            outputs.append(base_name+"_"+str(para[0])+"_"+str(para[1])+"_"+str(para[2])+"_"+str(para[3])+"."+ext)
            list_of_imagettes.append((outputs[0],  outputs[1]))
        return  list_of_imagettes
##----------------------------------------------------------------
    def get_im_and_segm(self, im_file, segm_dir):
        """
        Return, for a given slide which complete path_name is "im_file", a list whom each element is itself a list of three elements:
        - an imagette (smil image, RGB)
        - its corresponding groundtruth (smil image, UINT8)
        - the name of this imagette
        
        Indeed, not the whole slide will be processed later on, only some parts of it ("imagettes"), for training and validation of the method.

        Args:
           im_file: complete file name.
           segm_dir: folder containing the image segmentations.

        Returns:
            list_of_imagettes_and_corr_gt: list of size the number of imagettes selected for the given slide.
        """
        ROI_pos=ROI(im_file,ref_level=2,disk_size=4,thresh=220,black_spots=20,number_of_pixels_max=700000,verbose=False) ### RAJOUTER ARGUMENT EN OPTION
        file_name =  os.path.basename(im_file)
        [base_name, ext] = file_name.rsplit(".", 1)
        ct=0
        list_of_imagettes_and_corr_gt=[]
        for para in ROI_pos[0:2]: ##### change for a better imagettes' selection method
            outputs=[]
            outputs.append(from_pil_to_sm_image(GetImage(im_file,para)))                                                                                                                            
            if 'Tumor' in im_file:
                outputs.append(from_pil_to_sm_image(GetImage(os.path.join(segm_dir, base_name + "_Mask" + "." + ext),para),'GT'))
            else:
                w=outputs[0].getSize()[0]
                h=outputs[0].getSize()[1]
                d=outputs[0].getSize()[2]
                outputs.append( sm.Image('UINT8',w,h,d) )
            outputs.append(base_name+"_"+str(para[0])+"_"+str(para[1])+"_"+str(para[2])+"_"+str(para[3])+"."+ext)
            ct += 1
            list_of_imagettes_and_corr_gt.append((outputs[0],  outputs[1],  outputs[2]))
        return  list_of_imagettes_and_corr_gt
##----------------------------------------------------------------
    def iter_training(self, code, first=None, last=None):
        """
        How to iter on the database during training.
        Each slide of the train slides' database is reduced to a pertinent set of small imagettes (crops of the slide) for training and validation of the model.
        
        Returns for each imagette: (imagette, corresponding GT, name_imagette)
        """
        for im_file in os.listdir(self._im_dir[code])[first:last]:
            #debug_mess("Processing file %s" % os.path.basename(im_file))
            list_of_imagettes_and_corr_gt = self.get_im_and_segm(os.path.join(self._im_dir[code], im_file), self._segm_dir[code])
            for i in range(len(list_of_imagettes_and_corr_gt)):
                yield list_of_imagettes_and_corr_gt[i][0],  list_of_imagettes_and_corr_gt[i][1],  list_of_imagettes_and_corr_gt[i][2]
##----------------------------------------------------------------
    def iter_final_prediction(self, code, first=None, last=None):
        """
        How to iter on the database during prediction.
        Each slide of the train slides' database is reduced to a covering set of small imagettes (crops of the slide) for final prediction.
        
        Returns for each imagette: (imagette, name_imagette)
        """
        for im_file in os.listdir(self._im_dir[code])[first:last]:
            #debug_mess("Processing file %s" % os.path.basename(im_file))
            list_of_imagettes = self.get_image(os.path.join(self._im_dir[code], im_file))
            for i in range(len(list_of_imagettes)):
                yield list_of_imagettes[i][0],  list_of_imagettes[i][1]
##----------------------------------------------------------------


