# -*- coding: cp1252 -*-
"""
Description: encapsulation of operators for non-geodesic features.
Author: MoSeS-learning project: Vaïa Machairas, Etienne Decencière, Thomas Walter.
Creation date: 2015-07-02
"""
import pdb
import numpy as np

import smilPython as sp
import vigra
import useful_functions as uf
from BasicOperations import make_vigra_image

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

    def get_name(self):
        return self.__class__.__name__

    def __call__(self, imIn, imOut ):
        """Apply the operator on imIn.

        Args:
           imIn: original input image.
           imOut: image transformed.

        """
        pass
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Each operator is to be applied on a grey level image.
#List of operators functors (classes):
# identity
# erosion
# dilation
# opening
# closing
# top hat
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class IdentityFunctor(OperatorFunctorBase):
    def __init__(self, params):
        """
        Args:
           params: dictionrary containing the keys "neighborhood" and "size" for structuring element.
        """
        OperatorFunctorBase.__init__(self, params)

    def get_name(self):
        return "identity" 

    def __call__(self, imIn,  imOut):
        sp.copy(imIn,  imOut)
        return 
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class ErosionFunctor(OperatorFunctorBase):
    def __init__(self, params):
        """
        Args:
           params: dictionrary containing the keys "neighborhood" and "size" for structuring element.
        """
        OperatorFunctorBase.__init__(self, params)
        self._se = uf.set_structuring_element(self._params["neighborhood"], self._params["size"])

    def get_name(self):
        return "erosion_%s_size_%i" % (self._params["neighborhood"], self._params["size"])

    def __call__(self, imIn,  imOut):
        sp.erode(imIn,  imOut,  self._se)
        return 
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class DilationFunctor(OperatorFunctorBase):
    def __init__(self, params):
        """
        Args:
           params: dictionrary containing the keys "neighborhood" and "size" for structuring element.
        """
        OperatorFunctorBase.__init__(self, params)
        self._se = uf.set_structuring_element(self._params["neighborhood"], self._params["size"])

    def get_name(self):
        return "dilation_%s_size_%i" % (self._params["neighborhood"], self._params["size"])

    def __call__(self, imIn,  imOut):
        sp.dilate(imIn,  imOut,  self._se)
        return 
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class OpeningFunctor(OperatorFunctorBase):
    def __init__(self, params):
        """
        Args:
           params: dictionrary containing the keys "neighborhood" and "size" for structuring element.
        """
        OperatorFunctorBase.__init__(self, params)
        self._se = uf.set_structuring_element(self._params["neighborhood"], self._params["size"])

    def get_name(self):
        return "opening_%s_size_%i" % (self._params["neighborhood"], self._params["size"])

    def __call__(self, imIn,  imOut):
        sp.open(imIn,  imOut,  self._se)
        return 
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class ClosingFunctor(OperatorFunctorBase):
    def __init__(self, params):
        """
        Args:
           params: dictionrary containing the keys "neighborhood" and "size" for structuring element.
        """
        OperatorFunctorBase.__init__(self, params)
        self._se = uf.set_structuring_element(self._params["neighborhood"], self._params["size"])

    def get_name(self):
        return "closing_%s_size_%i" % (self._params["neighborhood"], self._params["size"])

    def __call__(self, imIn,  imOut):
        sp.close(imIn,  imOut,  self._se)
        return 
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class TopHatFunctor(OperatorFunctorBase):
    def __init__(self, params):
        """
        Args:
           params: dictionrary containing the keys "neighborhood" and "size" for structuring element.
        """
        OperatorFunctorBase.__init__(self, params)
        self._se = uf.set_structuring_element(self._params["neighborhood"], self._params["size"])

    def get_name(self):
        return "top_hat_%s_size_%i" % (self._params["neighborhood"], self._params["size"])

    def __call__(self, imIn,  imOut):
        sp.topHat(imIn,  imOut,  self._se)
        return 
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class TopHatInvFunctor(OperatorFunctorBase):
    def __init__(self, params):
        """
        Args:
           params: dictionrary containing the keys "neighborhood" and "size" for structuring element.
        """
        OperatorFunctorBase.__init__(self, params)
        self._se = uf.set_structuring_element(self._params["neighborhood"], self._params["size"])

    def get_name(self):
        return "top_hat_inv_%s_size_%i" % (self._params["neighborhood"], self._params["size"])

    def __call__(self, imIn,  imOut):
        imIn2 = sp.Image(imIn)
        sp.copy(imIn, imIn2)
        sp.inv(imIn2,  imIn2)
        sp.topHat(imIn2,  imOut,  self._se)
        return 
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class MorphologicalGradientFunctor(OperatorFunctorBase):
    def __init__(self, params):
        """
        Args:
           params: dictionrary containing the keys "neighborhood" and "size" for structuring element.
        """
        OperatorFunctorBase.__init__(self, params)
        self._se = uf.set_structuring_element(self._params["neighborhood"], self._params["size"])

    def get_name(self):
        return "morphological_gradient_%s_size_%i" % (self._params["neighborhood"], self._params["size"])

    def __call__(self, imIn,  imOut):
        sp.gradient(imIn,  imOut,  self._se)
        return 
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Temporaire: on a besoin d'appliquer des opérateurs non géodésiques à des fenetres pour classifier des pixels....l'astuce est de
## construire les memes operateurs non geodesiques mais en ajoutant un filtre moyenneur de taille la taille de la fenetre que l'on souhaitait 
## puis de recuperer la valeur pixel par pixel (pf).
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def apply_uniform_filter(imIn, imOut, window_size):
    arrIn = np.zeros([imIn.getSize()[0],  imIn.getSize()[1]])
    arrIn[:, :] = imIn.getNumArray()
    arrOut = imOut.getNumArray()
    arrOut[:, :] = scipy.ndimage.uniform_filter(arrIn, window_size)[:, :]
    return
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class IdentityFunctorW(OperatorFunctorBase):
    def __init__(self, params):
        """
        Args:
           params: dictionrary containing the keys "window_size".
        """
        OperatorFunctorBase.__init__(self, params)

    def get_name(self):
        return "identityW_%i"%(self._params["window_size"]) 

    def __call__(self, imIn,  imOut):
        imtmp = sp.Image(imOut)
        sp.copy(imIn,  imtmp)
        apply_uniform_filter(imtmp,  imOut, self._params["window_size"])
        return 
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class ErosionFunctorW(OperatorFunctorBase):
    def __init__(self, params):
        """
        Args:
           params: dictionrary containing the keys "neighborhood" and "size" for structuring element, and "window_size" for the window of integration.
        """
        OperatorFunctorBase.__init__(self, params)
        self._se = uf.set_structuring_element(self._params["neighborhood"], self._params["size"])

    def get_name(self):
        return "erosionW_%s_size_%i_%i" % (self._params["neighborhood"], self._params["size"],  self._params["window_size"])

    def __call__(self, imIn,  imOut):
        imtmp = sp.Image(imOut)
        sp.erode(imIn,  imtmp,  self._se)
        apply_uniform_filter(imtmp,  imOut, self._params["window_size"])
        return         
    
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class DilationFunctorW(OperatorFunctorBase):
    def __init__(self, params):
        """
        Args:
           params: dictionrary containing the keys "neighborhood" and "size" for structuring element, and "window_size" for the window of integration.
        """
        OperatorFunctorBase.__init__(self, params)
        self._se = uf.set_structuring_element(self._params["neighborhood"], self._params["size"])

    def get_name(self):
        return "dilation_%s_size_%i_%i" % (self._params["neighborhood"], self._params["size"],  self._params["window_size"])

    def __call__(self, imIn,  imOut):
        imtmp = sp.Image(imOut)
        sp.dilate(imIn,  imtmp,  self._se)
        apply_uniform_filter(imtmp,  imOut, self._params["window_size"])
        return 
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class OpeningFunctorW(OperatorFunctorBase):
    def __init__(self, params):
        """
        Args:
           params: dictionrary containing the keys "neighborhood" and "size" for structuring element, and "window_size" for the window of integration.
        """
        OperatorFunctorBase.__init__(self, params)
        self._se = uf.set_structuring_element(self._params["neighborhood"], self._params["size"])

    def get_name(self):
        return "opening_%s_size_%i_%i" % (self._params["neighborhood"], self._params["size"],  self._params["window_size"])

    def __call__(self, imIn,  imOut):
        imtmp = sp.Image(imOut)
        sp.open(imIn,  imtmp,  self._se)
        apply_uniform_filter(imtmp,  imOut, self._params["window_size"])
        return 
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class ClosingFunctorW(OperatorFunctorBase):
    def __init__(self, params):
        """
        Args:
           params: dictionrary containing the keys "neighborhood" and "size" for structuring element, and "window_size" for the window of integration.
        """
        OperatorFunctorBase.__init__(self, params)
        self._se = uf.set_structuring_element(self._params["neighborhood"], self._params["size"])

    def get_name(self):
        return "closing_%s_size_%i_%i" % (self._params["neighborhood"], self._params["size"],  self._params["window_size"])

    def __call__(self, imIn,  imOut):
        imtmp = sp.Image(imOut)
        sp.close(imIn,  imtmp,  self._se)
        apply_uniform_filter(imtmp,  imOut, self._params["window_size"])
        return 
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class TopHatFunctorW(OperatorFunctorBase):
    def __init__(self, params):
        """
        Args:
           params: dictionrary containing the keys "neighborhood" and "size" for structuring element, and "window_size" for the window of integration.
        """
        OperatorFunctorBase.__init__(self, params)
        self._se = uf.set_structuring_element(self._params["neighborhood"], self._params["size"])

    def get_name(self):
        return "top_hat_%s_size_%i_%i" % (self._params["neighborhood"], self._params["size"],  self._params["window_size"])

    def __call__(self, imIn,  imOut):
        imtmp = sp.Image(imOut)
        sp.topHat(imIn,  imtmp,  self._se)
        apply_uniform_filter(imtmp,  imOut, self._params["window_size"])
        return 
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class MorphologicalGradientFunctorW(OperatorFunctorBase):
    def __init__(self, params):
        """
        Args:
           params: dictionrary containing the keys "neighborhood" and "size" for structuring element, and "window_size" for the window of integration.
        """
        OperatorFunctorBase.__init__(self, params)
        self._se = uf.set_structuring_element(self._params["neighborhood"], self._params["size"])

    def get_name(self):
        return "morphological_gradient_%s_size_%i_%i" % (self._params["neighborhood"], self._params["size"],  self._params["window_size"])

    def __call__(self, imIn,  imOut):
        imtmp = sp.Image(imOut)
        sp.gradient(imIn,  imtmp,  self._se)
        apply_uniform_filter(imtmp,  imOut, self._params["window_size"])
        return 
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Features d'Ilastik:
# vigra.filters.gaussianSmoothing
# vigra.filters.laplacianOfGaussian
# vigra.filters.gaussianGradientMagnitude
# difference of gaussians
# vigra.structureTensorEigenValues
#
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class IlastikGaussianSmoothing(OperatorFunctorBase):
    def __init__(self, params):
        """
        Args:
           params: dictionary containing the key "sigma".
        """
        OperatorFunctorBase.__init__(self, params)

    def get_name(self):
        return "Ilastik_GaussianSmoothing_sigma_%f" % (self._params["sigma"])

    def __call__(self, imIn,  imOut):
        image_numpy = np.float32(imIn.getNumArray())
        image_vigra = make_vigra_image(image_numpy)
        image_vigra_out = vigra.filters.gaussianSmoothing(image_vigra,  self._params["sigma"])
        image_numpy_out = imOut.getNumArray()
        #pdb.set_trace()
        image_numpy_out[:, :] = np.uint8(np.abs(np.array(image_vigra_out[:, :, 0])))
        return 
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class IlastikLaplacianOfGaussian(OperatorFunctorBase):
    def __init__(self, params):
        """
        Args:
           params: dictionary containing the key "scale".
        """
        OperatorFunctorBase.__init__(self, params)

    def get_name(self):
        return "Ilastik_LaplacianOfGaussian_scale_%f" % (self._params["scale"])

    def __call__(self, imIn,  imOut):
        image_numpy = np.float32(imIn.getNumArray())
        image_vigra = make_vigra_image(image_numpy)
        image_vigra_out = vigra.filters.laplacianOfGaussian(image_vigra,  self._params["scale"])
        image_numpy_out = imOut.getNumArray()
        #pdb.set_trace()
        image_numpy_out[:, :] = np.uint8(np.abs(np.array(image_vigra_out[:, :, 0])))
        return 
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class IlastikGaussianGradientMagnitude(OperatorFunctorBase):
    def __init__(self, params):
        """
        Args:
           params: dictionary containing the key "sigma".
        """
        OperatorFunctorBase.__init__(self, params)

    def get_name(self):
        return "Ilastik_GaussianGradientMagnitude_sigma_%f" % (self._params["sigma"])

    def __call__(self, imIn,  imOut):
        image_numpy = np.float32(imIn.getNumArray())
        image_vigra = make_vigra_image(image_numpy)
        image_vigra_out = vigra.filters.gaussianGradientMagnitude(image_vigra,  self._params["sigma"])
        image_numpy_out = imOut.getNumArray()
        #pdb.set_trace()
        image_numpy_out[:, :] = np.uint8(np.abs(np.array(image_vigra_out[:, :, 0])))
        return 
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class IlastikDifferenceOfGaussians(OperatorFunctorBase):
    def __init__(self, params):
        """
        Args:
           params: dictionary containing the keys "sigma1" and "sigma2".
        """
        OperatorFunctorBase.__init__(self, params)

    def get_name(self):
        return "Ilastik_DifferenceOfGaussians_sigma1_%f_sigma2_%f" % (self._params["sigma1"], self._params["sigma1"] )

    def __call__(self, imIn,  imOut):
        image_numpy = np.float32(imIn.getNumArray())
        image_vigra = make_vigra_image(image_numpy)
        image_vigra_out1 = vigra.filters.gaussianSmoothing(image_vigra,  self._params["sigma1"])
        image_vigra_out2 = vigra.filters.gaussianSmoothing(image_vigra,  self._params["sigma2"])
        image_vigra_out = image_vigra_out1 - image_vigra_out2
        image_numpy_out = imOut.getNumArray()
        image_numpy_out[:, :] = np.uint8(np.abs(np.array(image_vigra_out[:, :, 0])))
        return 
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class IlastikStructureTensorEigenValues(OperatorFunctorBase):
    def __init__(self, params):
        """
        Args:
           params: dictionary containing the keys "innerScale" and "outerScale", "eigenvalueNumber".
        """
        OperatorFunctorBase.__init__(self, params)

    def get_name(self):
        return "Ilastik_StructureTensorEigenValues_%i_innerScale_%f_outerScale_%f" % (self._params["eigenvalueNumber"],  self._params["innerScale"], self._params["outerScale"] )

    def __call__(self, imIn,  imOut):
        image_numpy = np.float32(imIn.getNumArray())
        image_vigra = make_vigra_image(image_numpy)
        image_vigra_out = vigra.filters.structureTensorEigenvalues(image_vigra,  self._params["innerScale"],  self._params["outerScale"])
        image_numpy_out = imOut.getNumArray()
        image_numpy_out[:, :] = np.uint8(np.abs(np.array(image_vigra_out[:, :, self._params["eigenvalueNumber"]])))
        return 
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class IlastikHessianOfGaussianEigenvalues(OperatorFunctorBase):
    def __init__(self, params):
        """
        Args:
           params: dictionary containing the keys "scale" and "eigenvalueNumber".
        """
        OperatorFunctorBase.__init__(self, params)

    def get_name(self):
        return "Ilastik_HessianOfGaussianEigenvalues_%i_scale_%f" % (self._params["eigenvalueNumber"],  self._params["scale"])

    def __call__(self, imIn,  imOut):
        image_numpy = np.float32(imIn.getNumArray())
        image_vigra = make_vigra_image(image_numpy)
        image_vigra_out = vigra.filters.hessianOfGaussianEigenvalues(image_vigra,  self._params["scale"])
        image_numpy_out = imOut.getNumArray()
        image_numpy_out[:, :] = np.uint8(np.abs(np.array(image_vigra_out[:, :, self._params["eigenvalueNumber"]])))
        return 
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