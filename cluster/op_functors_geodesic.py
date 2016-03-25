# -*- coding: cp1252 -*-
"""
Description: encapsulation of operators for geodesic features.
Author: MoSeS-learning project: Vaïa Machairas, Etienne Decencière, Thomas Walter.
Creation date: 2015-08-26
"""
import pdb
import numpy as np
import smilPython as sp
import useful_functions as uf
import mahotas as mh
import scipy.ndimage

##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class OperatorFunctorBase(object):
    """
    General class for operators.
    """
    def __init__(self, params):
        """
        Args:
           params: dictionary giving the parameters.
        """
        self._params = params
        self.feats = None

    def get_name(self):
        return self.__class__.__name__

    def __call__(self, imIn ):
        """Apply the operator on imIn.

        Args:
           imIn: original input image.
        """
        pass
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Each operator is to be applied on a grey level "small" image corresponding to the encapsulation of a superpixel. Output: one value.
#List of operators functors (classes):
#- Haralick:
    # * Haralick_AngularSecondMoment
    # * Haralick_Contrast
    # * Haralick_Correlation
    # * Haralick_SumofSquaresVariance
    # * Haralick_InverseDifferenceMoment
    # * Haralick_SumAverage
    # * Haralick_SumVariance
    # * Haralick_SumEntropy
    # * Haralick_Entropy
    # * Haralick_DifferenceVariance
    # * Haralick_DifferenceEntropy
    # * Haralick_InformationMeasureofCorrelation1
    # * Haralick_InformationMeasureofCorrelation2
    # * Haralick_MaximalCorrelationCoefficient
# - LBP
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Haralick(OperatorFunctorBase):
    def __init__(self, params):
        """
        Args:
           params: dictionrary containing the key "direction" (1,2,3 or 4).
        """
        OperatorFunctorBase.__init__(self, params)
        self._direction_number = self._params["direction"]
        self._feature_name = self._params["feature_name"]
        self._dic_features = {
                              'AngularSecondMoment': 0, 
                              'Contrast':1, 
                              'Correlation':2, 
                              'SumofSquaresVariance':3, 
                              'InverseDifferenceMoment':4, 
                              'SumAverage':5, 
                              'SumVariance':6, 
                              'SumEntropy':7,
                              'Entropy':8, 
                              'DifferenceVariance':9, 
                              'DifferenceEntropy':10, 
                              'InformationMeasureofCorrelation1':11, 
                              'InformationMeasureofCorrelation2':12, 
                              'MaximalCorrelationCoefficient':13, 
        }
        
    def get_name(self):
        return 'Haralick'
        
    def __call__(self,  imIn):
        #print "size: ",  imIn.getSize()
        if imIn.getSize()[0] == 1 or imIn.getSize()[1] == 1 :
            to_add = rd.randint(0, 100000)

            imIn.save('./temp/temporary_save'+str(to_add)+'.png')
            arr = scipy.ndimage.imread('temporary_save'+str(to_add)+'.png')
        else:
#            arr = np.transpose(imIn.getNumArray())

            arr = imIn.getNumArray()
        feats = mh.features.haralick(arr, ignore_zeros=True, preserve_haralick_bug=False, compute_14th_feature=False, return_mean=False, return_mean_ptp=False, use_x_minus_y_variance=False)
        #pdb.set_trace()
        return feats
        
class HaralickFeature(Haralick):
    def __init__(self,  params, erase=False):
        """
        Args:
        params: dictionary containing the keys "direction" (1,2,3,4 or 'all') and  "feature_name" ("AngularSecondMoment", ...).
        """
        Haralick.__init__(self,  params)
        self.erase = erase

    def get_name(self):
        return self._feature_name+"_dir_"+str(self._direction_number)
            
    
    def __call__(self,  imIn, feats=None):
        #feats = Haralick({'direction': self._direction_number,  'feature_name': self._feature_name}).__call__(imIn)
        if self.erase:
            self.feats = Haralick({'direction': self._direction_number,  'feature_name': self._feature_name}).__call__(imIn)
        else:
            if feats is None: 
                self.feats = Haralick({'direction': self._direction_number,  'feature_name': self._feature_name}).__call__(imIn)
            else:
                self.feats = feats

        feats = self.feats
        try:
            if self._direction_number == 'all':
                val_out1 =round(feats[0,  self._dic_features[self._feature_name]], 5)
                val_out2 =round(feats[1,  self._dic_features[self._feature_name]], 5)
                val_out3 =round(feats[2,  self._dic_features[self._feature_name]], 5)
                val_out4 =round(feats[3,  self._dic_features[self._feature_name]], 5)
                val_out = (val_out1 + val_out2 + val_out3 + val_out4) / float(4)
            else:
                val_out =round(feats[self._direction_number - 1,  self._dic_features[self._feature_name]], 5)
        except:
            print "Warning: default value None when not possible to compute the feature."
            val_out = None
        return val_out


##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class LBP(OperatorFunctorBase):
    def __init__(self, params):
        """
        Args:
           params: dictionrary containing the keys:
          - "radius": in pixels
          - "points": number of points
          - ignore_zeros (bool)
        """
        OperatorFunctorBase.__init__(self, params)
        self._radius = self._params["radius"]
        self._points = self._params["points"]
        self._ignore_zeros = self._params["ignore_zeros"]
        self._preserve_shape = self._params["preserve_shape"]
        
    def get_name(self):
        return 'LBP_radius_%i_points_%i_ignorezeros_%s_preserveshape_%s'%(self._radius,  self._points,  str(self._ignore_zeros),  str(self._preserve_shape))
    
    def __call__(self,  imIn):
        """
        Output:
        histogram with 6 bins (1D-ndarray f size 6)
        """
        
        try:
            arr = imIn.getNumArray()
            arr = arr.T
            histogram = mh.features.lbp(arr,  self._radius,  self._points, ignore_zeros = self._ignore_zeros)
        except:
            to_add = rd.randint(0, 100000)
            imIn.save('./temp/temporary_save'+str(to_add)+'.png')
            arr = scipy.ndimage.imread('temporary_save'+str(to_add)+'.png')
            histogram = mh.features.lbp(arr,  self._radius,  self._points,  ignore_zeros = self._ignore_zeros)
        return histogram
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class LBP_bin1(LBP):
    def __init__(self, params):
        """
        Args:
           params: dictionrary containing the keys:
          - "radius": in pixels
          - "points": number of points
          - ignore_zeros (bool)
        """
        LBP.__init__(self, params)
        
    def get_name(self):
        return 'LBP1_radius_%i_points_%i_ignorezeros_%s_preserveshape_%s'%(self._radius,  self._points,  str(self._ignore_zeros),  str(self._preserve_shape))

    def __call__(self,  imIn):
        histo_val = LBP({'radius': self._radius,  'points': self._points, 'ignore_zeros': self._ignore_zeros,  'preserve_shape': self._preserve_shape}).__call__(imIn)
        #pdb.set_trace()
        return histo_val[0]

##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class LBP_bin2(LBP):
    def __init__(self, params):
        """
        Args:
           params: dictionrary containing the keys:
          - "radius": in pixels
          - "points": number of points
          - ignore_zeros (bool)
        """
        LBP.__init__(self, params)
        
    def get_name(self):
        return 'LBP2_radius_%i_points_%i_ignorezeros_%s_preserveshape_%s'%(self._radius,  self._points,  str(self._ignore_zeros),  str(self._preserve_shape))

    def __call__(self,  imIn):
        histo_val = LBP({'radius': self._radius,  'points': self._points, 'ignore_zeros': self._ignore_zeros,  'preserve_shape': self._preserve_shape}).__call__(imIn)
        return histo_val[1]

##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class LBP_bin3(LBP):
    def __init__(self, params):
        """
        Args:
           params: dictionrary containing the keys:
          - "radius": in pixels
          - "points": number of points
          - ignore_zeros (bool)
        """
        LBP.__init__(self, params)
        
    def get_name(self):
        return 'LBP3_radius_%i_points_%i_ignorezeros_%s_preserveshape_%s'%(self._radius,  self._points,  str(self._ignore_zeros),  str(self._preserve_shape))

    def __call__(self,  imIn):
        histo_val = LBP({'radius': self._radius,  'points': self._points, 'ignore_zeros': self._ignore_zeros,  'preserve_shape': self._preserve_shape}).__call__(imIn)
        return histo_val[2]

##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class LBP_bin4(LBP):
    def __init__(self, params):
        """
        Args:
           params: dictionrary containing the keys:
          - "radius": in pixels
          - "points": number of points
          - ignore_zeros (bool)
        """
        LBP.__init__(self, params)
        
    def get_name(self):
        return 'LBP4_radius_%i_points_%i_ignorezeros_%s_preserveshape_%s'%(self._radius,  self._points,  str(self._ignore_zeros),  str(self._preserve_shape))

    def __call__(self,  imIn):
        histo_val = LBP({'radius': self._radius,  'points': self._points, 'ignore_zeros': self._ignore_zeros,  'preserve_shape': self._preserve_shape}).__call__(imIn)
        return histo_val[3]

##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class LBP_bin5(LBP):
    def __init__(self, params):
        """
        Args:
           params: dictionrary containing the keys:
          - "radius": in pixels
          - "points": number of points
          - ignore_zeros (bool)
        """
        LBP.__init__(self, params)
        
    def get_name(self):
        return 'LBP5_radius_%i_points_%i_ignorezeros_%s_preserveshape_%s'%(self._radius,  self._points,  str(self._ignore_zeros),  str(self._preserve_shape))

    def __call__(self,  imIn):
        histo_val = LBP({'radius': self._radius,  'points': self._points, 'ignore_zeros': self._ignore_zeros,  'preserve_shape': self._preserve_shape}).__call__(imIn)
        return histo_val[4]

##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class LBP_bin6(LBP):
    def __init__(self, params):
        """
        Args:
           params: dictionrary containing the keys:
          - "radius": in pixels
          - "points": number of points
          - ignore_zeros (bool)
        """
        LBP.__init__(self, params)
        
    def get_name(self):
        return 'LBP6_radius_%i_points_%i_ignorezeros_%s_preserveshape_%s'%(self._radius,  self._points,  str(self._ignore_zeros),  str(self._preserve_shape))

    def __call__(self,  imIn):
        histo_val = LBP({'radius': self._radius,  'points': self._points, 'ignore_zeros': self._ignore_zeros,  'preserve_shape': self._preserve_shape}).__call__(imIn)
        return histo_val[5]
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class DiffIntContoursStd(OperatorFunctorBase):
    """
    "Contour" feature. The aim is to study how features of contours (instead of regions) impact performance when regions are used as CS.
    Feature which enables to compute the difference of standard deviation between the interior and the contour area of the superpixel.
    """
    def __init__(self, params):
        """
        Args:
        params: dictionary containing the keys:
        - "neighborhood": of structuring element
        """
        OperatorFunctorBase.__init__(self, params)
        self._neighborhood = self._params["neighborhood"]

    def get_name(self):
        return 'DiffIntContoursStd'

    def __call__(self,  imIn0):
        imIn = sp.Image(imIn0.getSize()[0]+2,  imIn0.getSize()[1]+2)
        sp.copy(imIn0,  imIn, 1, 1) # cette étape sert pour pouvoir éroder les pixels qui touchent le bord.
        se = uf.set_structuring_element(self._neighborhood, 1)
        im_bin = sp.Image(imIn)
        sp.test(imIn,  1, 0, im_bin)
        imwrk1= sp.Image(im_bin) # interior zone
        imwrk2 = sp.Image(im_bin) # contour zone (inside SP)
        sp.erode(im_bin,  imwrk1,  se)
        sp.sub(im_bin,  imwrk1,  imwrk2)
        sp.test(imwrk1, imIn,  0, imwrk1)
        sp.test(imwrk2, imIn,  0, imwrk2)
        histo1 = sp.histogram(imwrk1,  im_bin)
        histo2 = sp.histogram(imwrk2,  im_bin)
        list1 = uf.from_histogram_to_list(histo1,  True)
        list2 = uf.from_histogram_to_list(histo2,  True)
        #pdb.set_trace()
        print "list1: ",  len(list1)
        print "list2: ",  len(list2)
        if len(list2)==0 or len(list1)==0:
            imwrk1.showLabel()
            imwrk2.showLabel()
            #pdb.set_trace()
            return None
        else:
            return np.abs(np.std(list1) - np.std(list2))

##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def geodesicErosion(imIn, se_neigh, se_size):
    se = uf.set_structuring_element(se_neigh, 1)
    imOut = sp.Image(imIn)
    sp.copy(imIn,  imOut)
    imwrk = sp.Image(imOut)
    maxval = sp.maxVal(imIn)
    for _ in range(se_size):
        sp.test(imOut, imOut,  maxval,  imwrk)
        sp.erode(imwrk,  imOut,  se)
        sp.test(imIn,  imOut, 0 ,  imOut )
    return imOut

    
def geodesicDilation(imIn, se_neigh,  se_size):
    se = uf.set_structuring_element(se_neigh, 1)
    imOut = sp.Image(imIn)
    sp.copy(imIn,  imOut)
    imwrk = sp.Image(imOut)
    for i in range(se_size):
        sp.dilate(imOut,  imwrk, se)
        sp.test(imIn,  imwrk, 0 ,  imOut ) 
    return imOut


def geodesicOpening(imIn, se_neigh,  se_size):
    imwrk = geodesicErosion(imIn,  se_neigh,  se_size)
    imOut = geodesicDilation(imwrk,  se_neigh,  se_size)
    return imOut

def geodesicClosing(imIn, se_neigh,  se_size):
    imwrk = geodesicDilation(imIn,  se_neigh,  se_size)
    imOut = geodesicErosion(imwrk,  se_neigh,  se_size)
    return imOut

def geodesicTopHat(imIn, se_neigh,  se_size):
    imOut = sp.Image(imIn)
    imwrk = geodesicOpening(imIn,  se_neigh,  se_size)
    sp.sub(imIn, imwrk,  imOut)
    return imOut

def geodesicTopHatInv(imIn, se_neigh,  se_size):
    imIn2 = sp.Image(imIn)
    sp.copy(imIn,  imIn2)
    sp.inv(imIn2,  imIn2)
    sp.test(imIn, imIn2, 0,  imIn2)
    imOut = sp.Image(imIn)
    imwrk = geodesicOpening(imIn2,  se_neigh,  se_size)
    sp.sub(imIn2, imwrk,  imOut)
    return imOut

def geodesicMorphoGradient(imIn, se_neigh,  se_size):
    ime = geodesicErosion(imIn,  se_neigh,  se_size)
    imd = geodesicDilation(imIn,  se_neigh,  se_size)
    imOut = sp.Image(imIn)
    sp.sub(imd,  ime,  imOut)
    return imOut
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class GeodesicErosionFunctor(OperatorFunctorBase):
    """
    Compute the morphological erosion of the image. Contrary to the feature ErosionFunctor contained in op_functors.py,  
    this version enables to compute the erosion inside a support (e.g. superpixel) without taking into account what happens outside this very support.
    """
    def __init__(self,  params):
        """
        Args:
        params: dictionary containing the keys:
        - "neighborhood": of structuring element
        - "size": of structuring element
        - "integrator": how to compute one value from all values in the image (e.g. 'mean', 'std', 'max', 'min').
        """
        OperatorFunctorBase.__init__(self, params)

    def get_name(self):
        return "geodesic_erosion_%s_size_%i" % (self._params["neighborhood"], self._params["size"])

    def __call__(self, imIn):
        imOut = geodesicErosion(imIn,  self._params["neighborhood"],  self._params["size"])
        val_out = uf.from_image_to_value(imOut, self._params['integrator'],  True )
        val_out = round(val_out, 5)
        return val_out

##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class GeodesicDilationFunctor(OperatorFunctorBase):
    """
    Compute the morphological dilation of the image. Contrary to the feature ErosionFunctor contained in op_functors.py,  
    this version enables to compute the dilation inside a support (e.g. superpixel) without taking into account what happens outside this very support.
    """
    def __init__(self,  params):
        """
        Args:
        params: dictionary containing the keys:
        - "neighborhood": of structuring element
        - "size": of structuring element
        - "integrator": how to compute one value from all values in the image (e.g. 'mean', 'std', 'max', 'min').
        """
        OperatorFunctorBase.__init__(self, params)

    def get_name(self):
        return "geodesic_dilation_%s_size_%i_%s" % (self._params["neighborhood"], self._params["size"],  self._params["integrator"])

    def __call__(self, imIn):
        imOut = geodesicDilation(imIn, self._params["neighborhood"],  self._params["size"])
        val_out = uf.from_image_to_value(imOut, self._params['integrator'],  True )
        val_out = round(val_out, 5)
        return val_out

##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class GeodesicOpeningFunctor(OperatorFunctorBase):
    """
    Compute the morphological opening of the image. Contrary to the feature ErosionFunctor contained in op_functors.py,  
    this version enables to compute the opening inside a support (e.g. superpixel) without taking into account what happens outside this very support.
    """
    def __init__(self,  params):
        """
        Args:
        params: dictionary containing the keys:
        - "neighborhood": of structuring element
        - "size": of structuring element
        - "integrator": how to compute one value from all values in the image (e.g. 'mean', 'std', 'max', 'min').
        """
        OperatorFunctorBase.__init__(self, params)

    def get_name(self):
        return "geodesic_opening_%s_size_%i_%s" % (self._params["neighborhood"], self._params["size"],  self._params["integrator"])

    def __call__(self, imIn):
        imOut = geodesicOpening(imIn, self._params["neighborhood"],  self._params["size"])
        val_out = uf.from_image_to_value(imOut, self._params['integrator'],  True )
        val_out = round(val_out, 5)
        return val_out

##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class GeodesicClosingFunctor(OperatorFunctorBase):
    """
    Compute the morphological closing of the image. Contrary to the feature ErosionFunctor contained in op_functors.py,  
    this version enables to compute the closing inside a support (e.g. superpixel) without taking into account what happens outside this very support.
    """
    def __init__(self,  params):
        """
        Args:
        params: dictionary containing the keys:
        - "neighborhood": of structuring element
        - "size": of structuring element
        - "integrator": how to compute one value from all values in the image (e.g. 'mean', 'std', 'max', 'min').
        """
        OperatorFunctorBase.__init__(self, params)

    def get_name(self):
        return "geodesic_closing_%s_size_%i_%s" % (self._params["neighborhood"], self._params["size"],  self._params["integrator"])

    def __call__(self, imIn):
        imOut = geodesicClosing(imIn, self._params["neighborhood"],  self._params["size"])
        val_out = uf.from_image_to_value(imOut, self._params['integrator'],  True )
        val_out = round(val_out, 5)
        return val_out

##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class GeodesicTopHatFunctor(OperatorFunctorBase):
    """
    Compute the morphological tophat of the image. Contrary to the feature ErosionFunctor contained in op_functors.py,  
    this version enables to compute the tophat inside a support (e.g. superpixel) without taking into account what happens outside this very support.
    """
    def __init__(self,  params):
        """
        Args:
        params: dictionary containing the keys:
        - "neighborhood": of structuring element
        - "size": of structuring element
        - "integrator": how to compute one value from all values in the image (e.g. 'mean', 'std', 'max', 'min').
        """
        OperatorFunctorBase.__init__(self, params)

    def get_name(self):
        return "geodesic_top_hat_%s_size_%i_%s" % (self._params["neighborhood"], self._params["size"],  self._params["integrator"])

    def __call__(self, imIn):
        imOut = geodesicTopHat(imIn, self._params["neighborhood"],  self._params["size"])
        val_out = uf.from_image_to_value(imOut, self._params['integrator'],  True )
        val_out = round(val_out, 5)
        return val_out

##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class GeodesicTopHatInvFunctor(OperatorFunctorBase):
    """
    Compute the morphological tophat of the image. Contrary to the feature ErosionFunctor contained in op_functors.py,  
    this version enables to compute the tophat inside a support (e.g. superpixel) without taking into account what happens outside this very support.
    """
    def __init__(self,  params):
        """
        Args:
        params: dictionary containing the keys:
        - "neighborhood": of structuring element
        - "size": of structuring element
        - "integrator": how to compute one value from all values in the image (e.g. 'mean', 'std', 'max', 'min').
        """
        OperatorFunctorBase.__init__(self, params)

    def get_name(self):
        return "geodesic_top_hat_inv_%s_size_%i_%s" % (self._params["neighborhood"], self._params["size"],  self._params["integrator"])

    def __call__(self, imIn):
        imOut = geodesicTopHatInv(imIn, self._params["neighborhood"],  self._params["size"])
        val_out = uf.from_image_to_value(imOut, self._params['integrator'],  True )
        val_out = round(val_out, 5)
        return val_out

##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class GeodesicMorphoGradientFunctor(OperatorFunctorBase):
    """
    Compute the morphological morphological gradient of the image. Contrary to the feature ErosionFunctor contained in op_functors.py,  
    this version enables to compute the morphological gradient inside a support (e.g. superpixel) without taking into account what happens outside this very support.
    """
    def __init__(self,  params):
        """
        Args:
        params: dictionary containing the keys:
        - "neighborhood": of structuring element
        - "size": of structuring element
        - "integrator": how to compute one value from all values in the image (e.g. 'mean', 'std', 'max', 'min').
        """
        OperatorFunctorBase.__init__(self, params)

    def get_name(self):
        return "geodesic_morphological_gradient_%s_size_%i_%s" % (self._params["neighborhood"], self._params["size"],  self._params["integrator"])

    def __call__(self, imIn):
        imOut = geodesicMorphoGradient(imIn, self._params["neighborhood"],  self._params["size"])
        val_out = uf.from_image_to_value(imOut, self._params['integrator'],  True )
        val_out = round(val_out, 5)
        return val_out
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Features d'Ilastik (from Vigra vigra.filters):
## gaussianSmoothing
## laplacianOfGaussian
## gaussianGradientMagnitude
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
