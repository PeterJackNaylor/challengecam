# -*- coding: cp1252 -*-
"""
Description: 
Challenge CAMELYON16.
Useful functions.

Authors:  Vaïa Machairas, Etienne Decencière, Peter Naylor, Thomas Walter.

Creation date: 2016-02-24
"""
import pdb
import numpy as np
import smilPython as sp
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def my_concatenation(x,y):
    """
    Enables to concatenate two matrices (potentially empty) along the vertical axis.

    Inputs:
    x, y: two matrices with the same number of rows.

    Output:
    out_arr: concatenation matrice of x and y
    """
    if y.shape[0]==0:
        raise TypeError("The second array should not be empty.")
    if x is None or x.shape[0]==0:
        out_arr = y
    else:
        if (len(x.shape)==1) and (len(y.shape)==1):
            out_arr = np.concatenate((x[np.newaxis,:],y[np.newaxis,:]))
        elif len(x.shape)==1:
            out_arr = np.concatenate((x[np.newaxis,:],y))
        elif len(y.shape)==1:
            out_arr = np.concatenate((x,y[np.newaxis,:]))
        else:
            if x.shape[1]!= y.shape[1]:
                #pdb.set_trace()
                print "x: ",  x
                print "y: ",  y
            out_arr = np.concatenate((x,y))

    return out_arr
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def features_choice(dic, feature_list):
    if feature_list == 'all':
        the_list = dic.values()
    else:
        the_list=[]
        for elem in feature_list:
            the_list+= [dic[elem]]
    return the_list

##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def set_structuring_element(neighborhood, size):
    """
    Enables to create the structuring element.
    """
    if neighborhood=='V4':
        se = sp.CrossSE(size)
    elif neighborhood=='V6':
        se = sp.HexSE(size)
    elif neighborhood=='V8':
        se = sp.SquSE(size)
    else:
        raise TypeError("Only V4, V6 or V8 neighborhoods are available")

    return se
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def my_func_to_sort_dico(name_and_score):
    return name_and_score[1]
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def SoS(lign_i,  lign_j):
    """
    Enables to compute the sum of squares of differences between values of the two ligns, for all features whose values are available (i.e. not missing).
    """
    length = len(list(lign_i))
    sum = 0
    for k in range(length):
        if lign_i[k] != None and lign_j[k] != None:
            sum += np.power( lign_i[k]-lign_j[k] ,  2)
    return sum

##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def from_histogram_to_list(H,  avoid_zero_bin = False):
    """
    Inputs:
    H: histogram (map)
    avoid_zero_bin (bool): wether to remove the key 0.
    
    Outputs:
    liste or liste0 (list)
    
    Converts an histogram to a list.
    For example: H={'0':1, '1':5, '2':4} will give list = [0, 1, 1, 1, 1, 1, 2, 2, 2, 2] (or [1, 1, 1, 1, 1, 2, 2, 2, 2] if avoid_zero_bin is True). 
    """
    liste = []
    liste0 = []
    for key in H.keys():
        if H[key] != 0:
            if key != 0:
                for nb in range(H[key]):
                    liste += [key]
            else:
                for nb in range(H[key]):
                    liste0 += [key]
    if avoid_zero_bin == False:
        liste0 += liste
        return liste0
    else:
        return liste
    
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def from_list_to_value(list, integrator):
    """
    Enables to compute one value from a list of values.
    Inputs:
    -list: list of values
    - integrator: 'mean', 'max', 'std', 'min'.
    Output:
    - val_out: one value
    """
    if integrator == 'mean':
        val_out = np.mean(list)
    elif integrator == 'std':
        val_out = np.std(list)
    elif integrator == 'max':
        val_out = np.max(list)
    elif integrator == 'min':
        val_out = np.min(list)
    else:
        raise TypeError('The integrator should be taken among the following list: mean, std, max, min.')    
    return val_out

##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def from_image_to_value(image, integrator,  avoid_zero_bin = False):
    """
    Enables to compute one value from a histogram.
    Inputs:
    -image
    - integrator: 'mean', 'max', 'std', 'min'.
    Output:
    - val_out: one value
    """  
    H= sp.histogram(image)
    list_H = from_histogram_to_list(H, avoid_zero_bin)
    val_out = from_list_to_value(list_H,  integrator)
    return val_out

##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def visu_TFPN(im_pred,  im_GT):
    """
    Enables to see the:
   - true positives in green
   - true negatives in white
   - false positives in red
   - false negatives in blue
    """
    
    dic_color={'TP': [0, 255, 0],  'TN': [255, 255, 255], 'FP': [255, 0, 0],  'FN': [0, 0, 255] }
    sp.test(im_pred,  1,  0,  im_pred)
    sp.test(im_GT,  1,  0,  im_GT)

    imtmp = sp.Image(im_GT)
    iminv = sp.Image(im_GT)
    imTP = sp.Image(im_GT)
    imFN = sp.Image(im_GT)
    imTN = sp.Image(im_GT)
    imFP = sp.Image(im_GT)
    
    sp.test(im_pred == im_GT,  1,  0,  imtmp)
    sp.test(im_pred == im_GT,  0,  1,  iminv)
    
    sp.test(im_pred>0,  imtmp,  0,  imTP)
    sp.test(im_pred<1,  iminv,  0,  imFN)
    sp.test(im_GT<1,  imtmp,  0,  imTN)
    sp.test(im_GT<1, iminv,  0,  imFP)
    
    imR = sp.Image(im_GT)
    imG = sp.Image(im_GT)
    imB = sp.Image(im_GT)
    
    sp.test(imTP,  dic_color['TP'][0], 0,  imR)
    sp.test(imTP,  dic_color['TP'][1], 0,  imG)
    sp.test(imTP,  dic_color['TP'][2], 0,  imB)
    
    sp.test(imFN, dic_color['FN'][0],  imR,  imR)
    sp.test(imFN, dic_color['FN'][1],  imG,  imG)
    sp.test(imFN, dic_color['FN'][2],  imB,  imB)
    
    sp.test(imTN, dic_color['TN'][0],  imR,  imR)
    sp.test(imTN, dic_color['TN'][1],  imG,  imG)
    sp.test(imTN, dic_color['TN'][2],  imB,  imB)
    
    sp.test(imFP, dic_color['FP'][0],  imR,  imR)
    sp.test(imFP, dic_color['FP'][1],  imG,  imG)
    sp.test(imFP, dic_color['FP'][2],  imB,  imB)
    
    imOut = sp.Image(imR,  'RGB')
    im_slices = sp.Image()
    sp.splitChannels(imOut,  im_slices)
    sp.copy(imR,  im_slices.getSlice(0))
    sp.copy(imG,  im_slices.getSlice(1))
    sp.copy(imB,  im_slices.getSlice(2))
    sp.mergeChannels(im_slices,  imOut)
    
    return imOut
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def compute_TFPN(im_pred,  im_GT):
    """
    Enables to compute TP: true positives and FP: false positives.
    im_pred and im_GT must be in binary form.
    """
    imtmp = sp.Image(im_GT)
    iminv = sp.Image(im_GT)
    imTP = sp.Image(im_GT)
    imFP = sp.Image(im_GT)
    sp.test(im_pred == im_GT,  1,  0,  imtmp)
    sp.test(im_pred == im_GT,  0,  1,  iminv)
    sp.test(im_pred>0,  imtmp,  0,  imTP)
    sp.test(im_GT<1, iminv,  0,  imFP)
    return sp.vol(imTP), sp.vol(imFP)
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def image_enlargement(original_image, window_size,  substitution_value=0):
    """
    This function enables to create a enlarged image containing at its center the original image and filling the new boarders with a substitution value.
    
    Inputs:
    - original_image: original image (smil)
    - window_size (int): number of pixels (depth) to be added on the boarders of the original_image
    - substitution_value: how to fill the pixels added on the boarders (0 by default)
    
    Output:
    - enlarged_image: image of size [original_image.getSize()[0]+2*window_size, original_image.getSize()[1]+2*window_size]
    """
    enlarged_image = sp.Image(original_image.getSize()[0]+2*window_size, original_image.getSize()[1]+2*window_size)
    sp.fill(enlarged_image,  0)
    sp.copy(original_image,  enlarged_image,  window_size,  window_size)
    return enlarged_image
    
def translation_3D_matrix(enlarged_image,  window_size):
    """"
    This function enables to create a 3D matrix where each plane corresponds to a translation of the enlarged_image. 
    The translations are defined in the window of radius window_size. The order is the same as np.ravel.
    """
    enlarged_image_numpy = np.transpose(enlarged_image.getNumArray())
    the_3D_matrix = np.zeros([enlarged_image.getSize()[1]- 2*window_size, enlarged_image.getSize()[0]- 2*window_size, np.power(2*window_size+1, 2)])
    d=0
    for i in range(2*window_size+1):
        for j in range(2*window_size+1):
            the_3D_matrix[:, :,  d] = np.transpose(enlarged_image_numpy[i:i+enlarged_image.getSize()[1] - 2*window_size,  j:j+enlarged_image.getSize()[0] - 2*window_size])
            d+=1
    return the_3D_matrix
