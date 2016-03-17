#-*- coding: utf-8 -*-
"""
Description: 
Challenge CAMELYON16.
Classes to access segmentation databases.

Authors:  Vaïa Machairas, Etienne Decencière, Peter Naylor, Thomas Walter.

Creation date: 2016-02-24
"""
import pdb
import os
import smilPython as sm

from debug_mode import debug_mess

def bin_jpg(im_segm):
    """Binarization.

    Some segmentation images, such as in The Weizmann Horse Database,
    are coded with lossy jpeg... Therefore after decoding we have no more a binary
    image. An appropriate threshold should solve the problem.
    """
    sm.compare(im_segm, ">", 128, 255, 0, im_segm)

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

##--------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------
class SegmDataBaseStandard(SegmDataBaseCommon):
    """Segmentation data base server

    This is the standard version. The data base has the following structure:

    dir_in/images/train
                  /val
                  /test
           /GT/train
              /val
              /test

    A single ground-truth image is available for each input image."""

    def __init__(self, dir_in):
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
        
        self._im_dir["otsu_train"] = os.path.join(dir_in, "images/train")
        self._im_dir["otsu_test"] = os.path.join(dir_in, "images/test")
        self._im_dir["otsu_val"] = os.path.join(dir_in, "images/val")
        self._segm_dir["otsu_train"] = os.path.join(dir_in, "GT/train")
        self._segm_dir["otsu_test"] = os.path.join(dir_in, "GT/test")
        self._segm_dir["otsu_val"] = os.path.join(dir_in, "GT/val")
        self._res_dir['otsu_train'] = os.path.join(dir_in, "resultats/otsu/train")
        self._res_dir['otsu_test'] = os.path.join(dir_in, "resultats/otsu/test")
        self._res_dir['otsu_val'] = os.path.join(dir_in, "resultats/otsu/val")

    def train(self):
        return self.iter("train")

    def val(self):
        return self.iter("val")

    def test(self):
        return self.iter("test")
    
    def res_train(self):
        return self.iter2("train")

    def res_test(self):
        return self.iter2("test")

    def res_val(self):
        return self.iter2("val")
    
    def otsu_res_train(self):
        return self.iter2("otsu_train")
        
    def otsu_res_test(self):
        return self.iter2("otsu_test")
        
    def otsu_res_val(self):
        return self.iter2("otsu_val")
##--------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------
class SegmChallengeCamelyon16(SegmDataBaseStandard):
    """
    Base du challenge CAMELYON16 (ISBI16).
    """
    def __init__(self,  dir_in):
        SegmDataBaseStandard.__init__(self, dir_in)

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
        im = sm.Image(im_file)
        
        segm_list = []
        file_name =  os.path.basename(im_file)
        [base_name, ext] = file_name.rsplit(".", 1)
        im_segm = sm.Image(os.path.join(segm_dir, base_name + "_Mask" + "." + ext))
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
        [base_name, ext] = file_name.rsplit(".", 1)
        im_segm = sm.Image(os.path.join(segm_dir, base_name + "_Mask" + "." + ext))
        self.segm_post_process(im_segm)
        segm_list.append(im_segm)
        im_pred = sm.Image(os.path.join(res_dir, file_name))
        return  segm_list, im_pred, file_name

##--------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------
