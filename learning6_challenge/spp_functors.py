# -*- coding: cp1252 -*-
"""
Description: encapsulation of superpixel methods
Author: MoSeS-learning project: Vaïa Machairas, Etienne Decencière, Thomas Walter.
Creation date: 2015-04-27
"""
import pdb
import smilPython as sp
import numpy as np
import scipy.ndimage

import demo_waterpixels_smil as wp
import morphee as mm

import sys
sys.path.append("/home/vaiamachairas/src/SLIC/slic-python-master")
import slic

class SuperpixelFunctorBase(object):
    def __init__(self, params):
        """
        Args:
           params: dictionary giving the parameters.
        """
        self._params = params
        self._cache_im = None
        self._cache_seg = None

    def get_name(self):
        return self.__class__.__name__

    def check_cache(self, im):
#        if self._cache_im is not None and sp.equ(im, self._cache_im):
#            return True
#        else:
#            return False
        return False

    def update_cache(self, im, seg):
        self._cache_im = im
        self._cache_seg = seg


    def __call__(self, im):
        """Compute the superpixels on im.

        Args:
           im: original input image.
        Returns:
           Resulting superpixel segmentation.
        """
        pass



class WaterpixelsFunctor(SuperpixelFunctorBase):
    def __init__(self, params):
        """
        Args:
           params: dictionrary containing the keys "step", "k" and "filter_ori".
        """
        SuperpixelFunctorBase.__init__(self, params)

    def get_name(self):
        return "wp_step%i_k%i_filter%s" % (self._params["step"], self._params["k"], str(self._params["filter_ori"]))

    def __call__(self, im):
        if self.check_cache(im) is False:
            if 1:
                im_tmp = sp.Image(im)
                im_tmp << im
                seg =  wp.demo_m_waterpixels(im_tmp, self._params["step"], self._params["k"], self._params["filter_ori"])[0]
                self.update_cache(im, seg)
                #seg.showLabel()
                print "we did it"
        else:
            im.show()
            self._cache_im.show()
        return self._cache_seg

class SLICSuperpixelsFunctor(SuperpixelFunctorBase):
    def __init__(self, params):
        """
        Args:
           params: dictionrary containing the keys "nb_regions" and "m" (regularity parameter).
        """
        SuperpixelFunctorBase.__init__(self, params)

    def get_name(self):
        return "slic_%i_regions_m%i" % (self._params["nb_regions"], self._params["m"])

    def __call__(self, im):
        if self.check_cache(im) is False:
            arrIn = np.zeros(( im.getSize()[0], im.getSize()[1], 3  ))
            if im.getTypeAsString()=="RGB":
                image_slices = sp.Image()
                sp.splitChannels(im, image_slices)
                for i in range(3):
                    arrtmp = image_slices.getSlice(i)
                    arrIn[:, :, i] = arrtmp.getNumArray()
            else:
                for i in range(3):
                    arrIn[:, :, i] = im.getNumArray()
            region_labels = slic.slic_n(np.uint8(arrIn), self._params["nb_regions"],   self._params["m"])
            imout = sp.Image(region_labels.shape[0],  region_labels.shape[1])
            arrOut = imout.getNumArray()
            for i in range(region_labels.shape[0]):
                for j in range(region_labels.shape[1]):
                    arrOut[i, j] = region_labels[i, j]
            ##
            copie16 = sp.Image(imout, "UINT16")
            sp.copy(imout,  copie16)
            self.update_cache(im, copie16)
        return self._cache_seg

##---------------------------------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------------------------------- 
class WindowFunctor(SuperpixelFunctorBase):
    def __init__(self,  params):
        """
        Args:
        params : dictionary containing the key "size" (length of the side of the square window).
        """
        SuperpixelFunctorBase.__init__(self, params)
        
    def get_name(self):
        return "window_size_%i"%(self._params["size"])
    
    def __call__(self,  im):
        im_size = im.getSize()
        _ ,  im_morphm = ws.im_labelled_square_grid_points_v2( im_size,  self._params["size"], 0)
        imout_smil = sp.Image("UINT16", im_size[0], im_size[1], im_size[2])
        sp.copy(sp.MorphmInt(im_morphm),  imout_smil)
        self.update_cache(im, imout_smil)
        return self._cache_seg
