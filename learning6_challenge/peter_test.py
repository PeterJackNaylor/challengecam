# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 13:15:51 2016

@author: naylor
"""
import numpy as np
from debug_mode import debug_mess
import cPickle as pickle
import random
import smilPython as sm
import useful_functions as uf
import os
import openslide
from BasicOperations import ROI
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
            yield outputs[0],  outputs[1],  outputs[2],  os.path.basename(im_file)

    def iter2(self, code):
        for im_file in os.listdir(self._im_dir[code]):
            #debug_mess("Processing file %s" % os.path.basename(im_file))
            outputs = self._im_and_segm2(os.path.join(self._im_dir[code], im_file),   self._segm_dir[code], self._res_dir[code])
            yield outputs[0],  outputs[1],  outputs[2],  os.path.basename(im_file)
            
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


class SegmChallengeCamelyon16(SegmDataBaseCommon):
    """
    Base du challenge CAMELYON16 (ISBI16).
    """
    def __init__(self,  dir_in):
        SegmDataBaseCommon.__init__(self, dir_in)
        self._im_dir["train"] = os.path.join(dir_in, "Normal")
        self._im_dir["val"] = os.path.join(dir_in, "images/val")
        self._im_dir["test"] = os.path.join(dir_in, "images/test")
        self._segm_dir["train"] = os.path.join(dir_in, "GT/train")
        self._segm_dir["val"] = os.path.join(dir_in, "GT/val")
        self._segm_dir["test"] = os.path.join(dir_in, "GT/test")
        self._res_dir["train"] = os.path.join(dir_in, "resultats/train")
        self._res_dir["test"] = os.path.join(dir_in, "resultats/test")
        self._res_dir["val"] = os.path.join(dir_in, "resultats/val")
        
        self._im_dir["otsu_train"] = os.path.join(dir_in, "images/train")
        self._im_dir["otsu_test"] = os.path.join(dir_in, "images/test")
        self._im_dir["otsu_val"] = os.path.join(dir_in, "images/val")
        self._segm_dir["otsu_train"] = os.path.join(dir_in, "GT/train")
        self._segm_dir["otsu_test"] = os.path.join(dir_in, "GT/test")
        self._segm_dir["otsu_val"] = os.path.join(dir_in, "GT/val")
        self._res_dir['otsu_train'] = os.path.join(dir_in, "resultats/otsu/train")
        self._res_dir['otsu_test'] = os.path.join(dir_in, "resultats/otsu/test")
        self._res_dir['otsu_val'] = os.path.join(dir_in, "resultats/otsu/val")
    
    def iter(self, code, first=None, last=None):
        for im_file in os.listdir(self._im_dir[code])[first:last]:
            #debug_mess("Processing file %s" % os.path.basename(im_file))
            slide , ROI_pos = self._im_and_segm(os.path.join(self._im_dir[code], im_file), self._segm_dir[code])
            file_name =  os.path.basename(im_file)
            [base_name, ext] = file_name.rsplit(".", 1)
            for para in ROI_pos:
                outputs=np.zeros(3)
                outputs[0]=sm.Image(GetImage(os.path.join(self._im_dir[code], im_file),para))
                if 'Tumor' in im_file:
                    outputs[1] = [sm.Image(GetImage(os.path.join(self._segm_dir[code], base_name + "_Mask" + "." + ext),para))]
                else:
                    w=outputs[0].getSize()[0]
                    h=outputs[0].getSize()[1]
                    d=outputs[0].getSize()[2]
                    outputs[1] = [sm.Image(w,h,d)]
                outputs[2] = file_name
            yield outputs[0],  outputs[1],  outputs[2],  os.path.basename(im_file)
    
    def segm_post_process(self, im_segm):
        
        sm.compare(im_segm, ">", 0, 1, 0, im_segm)
#        pdb.set_trace()
#        image_slices = sm.Image()
#        sm.splitChannels(im_segm, image_slices)
#        sm.compare(image_slices.getSlice(0), ">", 0, 1, 0, image_slices.getSlice(0))
#        return image_slices.getSlice(0)

    def _im_and_segm(self, im_file, segm_dir):
        """Return  image and the corresponding segmentations.

        Args:
           im_file: complete file name.
           segm_dir: folder containing the image segmentations.

        Returns:
           im: original image
           segm_list: list containing the corresponding GT segmentations
           file_name: original file name
        """
        slide = openslide.openslide(im_file)
        ROI_pos=ROI(im_file,ref_level=2,disk_size=4,thresh=220,black_spots=20,number_of_pixels_max=700000,verbose=False) ### RAJOUTER ARGUMENT EN OPTION
        return(slide,ROI_pos)        
        #im = sm.Image(im_file)
        #segm_list = []
        #file_name =  os.path.basename(im_file)
        #[base_name, ext] = file_name.rsplit(".", 1)
        #im_segm = sm.Image(os.path.join(segm_dir, base_name + "_Mask" + "." + ext))
        #self.segm_post_process(im_segm)
        #segm_list.append(im_segm)
        #return im, segm_list, file_name

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
        [base_name, ext] = file_name.rsplit(".", 1)
        im_segm = sm.Image(os.path.join(segm_dir, base_name + "_Mask" + "." + ext))
        self.segm_post_process(im_segm)
        segm_list.append(im_segm)
        im_pred = sm.Image(os.path.join(res_dir, file_name))
        return  segm_list, im_pred, file_name
