# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 15:36:56 2016

@author: naylor
"""
from scipy import ndimage
import random
import numpy as np
import openslide
from skimage.morphology import disk,closing
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import smilPython as sm
#from Evaluation_FROC import computeEvaluationMask,computeEvaluationMask_Peter

import openslide
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from skimage import measure
import os
import sys
import pdb 
import FIMM_histo.deconvolution as deconv
from sklearn.cluster import KMeans
from PIL import Image

def computeEvaluationMask_Peter(pixelarray,resolution,level):
    distance = nd.distance_transform_edt(255 - pixelarray[:,:])
    Threshold = 75/(resolution * pow(2, level) * 2) # 75µm is the equivalent size of 5 tumor cells
    binary = distance < Threshold
    filled_image = nd.morphology.binary_fill_holes(binary)
    evaluation_mask = measure.label(filled_image, connectivity = 2) 
    return evaluation_mask

def GetImage(c,para):
    ## Returns cropped image given a set of parameters
    if len(para)!=5:
            print "Not enough parameters..."
    elif isinstance(c,str):
        sample=openslide.open_slide(c).read_region((para[0],para[1]),para[4],(para[2],para[3]))
    else:
        sample=c.read_region((para[0],para[1]),para[4],(para[2],para[3]))

    #pdb.set_trace()
    # do color deconvolution on the sample image. 
    dec = deconv.Deconvolution()
    dec.params['image_type'] = 'HEDab'
    
    np_img = np.array(sample)
    dec_img = dec.colorDeconv(np_img[:,:,:3])
    
    new_img = Image.fromarray(dec_img.astype('uint8'))

    return(new_img)
    

def get_X_Y(slide,x_0,y_0,level):
    ## Gives you the coordinates for the level 0 image for a given couple of pixel

    size_x_0=slide.level_dimensions[level][0]
    size_y_0=slide.level_dimensions[level][1]
    size_x_1=float(slide.level_dimensions[0][0])
    size_y_1=float(slide.level_dimensions[0][1])
  
    x_1=x_0*size_x_1/size_x_0
    y_1=y_0*size_y_1/size_y_0
    
    return int(x_1),int(y_1)

def get_X_Y_from_0(slide,x_1,y_1,level):
    ## Gives you the coordinates for the level 'level' image for a given couple of pixel from resolution 0

    size_x_0=slide.level_dimensions[level][0]
    size_y_0=slide.level_dimensions[level][1]
    size_x_1=float(slide.level_dimensions[0][0])
    size_y_1=float(slide.level_dimensions[0][1])
  
    x_0=x_1*size_x_0/size_x_1
    y_0=y_1*size_y_0/size_y_1
  

    return int(x_0),int(y_0)
                
                
def get_size(slide,size_x,size_y,level_from,level_to):
    ## Gives comparable size from one level to an other    
    
    ds = slide.level_downsamples
    scal=float(ds[level_from])/ds[level_to]
    size_x_new = int(float(size_x)*scal)
    size_y_new = int(float(size_y)*scal)
    return(size_x_new,size_y_new)

def White_score(slide,para,thresh):
    crop=GetImage(slide,para) 
    #pdb.set_trace()
    crop=np.array(crop)[:,:,0]
    binary = crop > thresh
    nber_ones=sum(sum(binary))
    nber_total=binary.shape[0]*binary.shape[1]
    return(float(nber_ones)/nber_total)
    
    
def Best_Finder_rec(slide,level,x_0,y_0,size_x,size_y,image_name,ref_level,list_roi,number_of_pixels_max,thresh,verbose,marge=0):
    ## This function enables you to cut up a portion of a slide to a given resolution. 
    ## It will try and minimize the number of create images for a given portion of the image
    if size_x*size_y==0:
        print 'Warning: width or height is null..'
        return([])
    else:
        if level==ref_level:          
            if size_x*size_y<number_of_pixels_max: ##size of level 3
                if marge>0:
                    if isinstance(marge,int):  ## if it is int, then it ill add that amount of pixels
                        extra_pixels=marge
                    elif isinstance(marge,float): ## if it is float it will multiply it with respect to its size. 
                        extra_pixels=int(np.ceil(marge*min(size_x,size_y)))
                    size_x +=extra_pixels/2
                    size_y +=extra_pixels/2
                    width_xp,height_xp=get_size(slide,extra_pixels/2,extra_pixels/2,level,0)
                    x_0 = max(x_0 - width_xp,0)
                    y_0 = max(y_0 - height_xp,0)
                para=[x_0,y_0,size_x,size_y,level]
                if White_score(slide,para,thresh)<0.5:
                    list_roi.append(para)
                return(list_roi)
            else:
                            
                size_x_new=int(size_x*0.5)
                size_y_new=int(size_y*0.5)
                    
                diese_str="#"*level*10
                if verbose:
                    print diese_str +"split level "+ str(level)
                
                width_x_0,height_y_0=get_size(slide,size_x,size_y,level,0)                
                x_1=x_0+int(width_x_0*0.5)
                y_1=y_0+int(height_y_0*0.5)
                
                image_name=image_name+"_Split_id_"+str(random.randint(0, 1000))            
                list_roi=Best_Finder_rec(slide,level,x_0,y_0,size_x_new,size_y_new,image_name,ref_level,list_roi,number_of_pixels_max,thresh,verbose,marge)
                list_roi=Best_Finder_rec(slide,level,x_1,y_0,size_x_new,size_y_new,image_name,ref_level,list_roi,number_of_pixels_max,thresh,verbose,marge)
                list_roi=Best_Finder_rec(slide,level,x_0,y_1,size_x_new,size_y_new,image_name,ref_level,list_roi,number_of_pixels_max,thresh,verbose,marge)
                list_roi=Best_Finder_rec(slide,level,x_1,y_1,size_x_new,size_y_new,image_name,ref_level,list_roi,number_of_pixels_max,thresh,verbose,marge)
                return(list_roi)
        else:
            if level > 1:
                if size_x*size_y>number_of_pixels_max: ##size of level 3
                    
                    size_x_new,size_y_new=get_size(slide,size_x,size_y,level,level-1)
                    size_x_new=int(size_x_new*0.5)
                    size_y_new=int(size_y_new*0.5)
                    
                    if verbose:
                        diese_str="#"*level*10
                        print diese_str +"split level "+ str(level)
                    
                    width_x_0,height_y_0=get_size(slide,size_x,size_y,level,0)
                    
                    x_1=x_0+int(width_x_0*0.5)
                    y_1=y_0+int(height_y_0*0.5)
                    
                    list_roi=Best_Finder_rec(slide,level-1,x_0,y_0,size_x_new,size_y_new,image_name,ref_level,list_roi,number_of_pixels_max,thresh,verbose,marge)
                    list_roi=Best_Finder_rec(slide,level-1,x_1,y_0,size_x_new,size_y_new,image_name,ref_level,list_roi,number_of_pixels_max,thresh,verbose,marge)
                    list_roi=Best_Finder_rec(slide,level-1,x_0,y_1,size_x_new,size_y_new,image_name,ref_level,list_roi,number_of_pixels_max,thresh,verbose,marge)
                    list_roi=Best_Finder_rec(slide,level-1,x_1,y_1,size_x_new,size_y_new,image_name,ref_level,list_roi,number_of_pixels_max,thresh,verbose,marge)
                    return(list_roi)
                else:
                    
                    size_x_new,size_y_new=get_size(slide,size_x,size_y,level,level-1)
                    
                    list_roi=Best_Finder_rec(slide,level-1,x_0,y_0,size_x_new,size_y_new,image_name,ref_level,list_roi,number_of_pixels_max,thresh,verbose,marge)
                    return(list_roi)
            else:
                print "Not enough variability on second split"

def Mask_ROI_cl(im,disk_size,thresh=None,black_spots=None,with_morph=False):
    ## We first do a closing of the image before cutting it.
    l=np.array([0,0])
    if not isinstance(im,l.__class__):
        numpy_array=np.array(im)
    else:
        numpy_array=im 
    if len(numpy_array.shape)==3:
        numpy_array=numpy_array[:,:,0:3].mean(axis=2)

    selem = disk(disk_size)
    closed = closing(numpy_array, selem)
    if thresh is None:
        thresh = threshold_otsu(closed)
    binary = closed > thresh
    if binary.dtype=='bool':
        binary=binary+0
    if black_spots is not None:    
        binary2 = closed > black_spots
        binary2 = binary2 + 0
        binary = binary - binary2 
    else:
        binary -=1
    binary=binary * -1
    if with_morph:
        return(binary,closed)
    else:
        return(binary)

def contour(image_A,image_B,seuil=0.1):   ##### il faut parametrer ce seuillage
    nb_A=image_A.shape[0]*image_A.shape[1]
    if nb_A!=image_B.shape[0]*image_B.shape[1]:
        raise NameError("Not same size between both images")

    image_A[image_A>0]=1
    image_B[image_B>0]=1
    
    s=np.abs(image_A-image_B)
    sum_s=sum(sum(s))
    if (float(sum_s)/nb_A) >seuil:
        return True
    else:
        return False


def find_square(slide,x_i,y_i,level_resolution,nber_pixels,current_level):
    #for a given pixel, returns a square centered on this pixel of a certain h and w
    ### I could add a white filter, so that shit images stay small

    #pdb.set_trace()
    x_0,y_0=get_X_Y(slide,x_i,y_i,current_level)

    h=np.ceil(np.sqrt(nber_pixels))
    w=h
    w_0,h_0=get_size(slide,w,h,level_resolution,0)
    new_x=max(x_0-w_0/2,0)
    new_y=max(y_0-h_0/2,0)
    return([int(new_x),int(new_y),int(w),int(h),int(level_resolution)])
   
def Sample_imagette(im_bin,N,slide,level_resolution,nber_pixels,current_level,mask):
    ### I should modify this function, so that the squares don't fall on each other.. 
    y,x=np.where(im_bin>0)
    n=len(x)
    indices=range(n)
    random.shuffle(indices)
    #indices=indices[0:N]
    result=[]
    i=0
    #pdb.set_trace()
    while i<n and len(result)<N:
        x_i=x[indices[i]]
        y_i=y[indices[i]]
        if mask[y_i,x_i]==0:
            para=find_square(slide,x_i,y_i,level_resolution,nber_pixels,current_level)
            result.append(para)
            x_,y_=get_X_Y_from_0(slide,para[0],para[1],current_level)
            w_,h_=get_size(slide,para[2],para[3],level_resolution,current_level)
            add=int(w_/2)      ### allowing how much overlapping?
            mask[max((y_-add),0):min((y_+add+h_),im_bin.shape[0]),max((x_-add),0):min((x_+add+w_),im_bin.shape[1])]=1
        i +=1            
    return(result,mask)


def Peter_Ipython_plot(im_to_plot,title="", cmap=None,size=12):
    fig, ax = plt.subplots(figsize=(size,size))
    if hasattr(im_to_plot,"__class__"):
        if str(im_to_plot.__class__)=="<class 'vigra.arraytypes.VigraArray'>":
            im_to_plot=im_to_plot.transpose()
            im_to_plot=np.array(im_to_plot)
    if cmap is None:
        im = ax.imshow(im_to_plot,origin='lower')
    else:
        im = ax.imshow(im_to_plot,origin='lower',cmap=cmap)
        fig.colorbar(im)
    ax.set_title(title, size=20)
    plugins.connect(fig, plugins.MousePosition(fontsize=14))  
    plt.show()

def ROI(name,ref_level=4, disk_size=4, thresh=None, black_spots=None,
        number_of_pixels_max=1000000, verbose=False, marge=0, method='grid',
        mask_address=None, contour_size=3, N_squares=100, seed=None):   
    ## creates a grid of the all interesting places on the image

    if seed is None:
        random.seed(seed)

    if '/' in name:
        cut=name.split('/')[-1]
        folder=cut.split('.')[0]
    else:
        folder=name.split(".")[0] 
    slide = openslide.open_slide(name)
    list_roi=[]
    #pdb.set_trace()

    if method=='grid':
        lowest_res=len(slide.level_dimensions)-2
    
        s=np.array(slide.read_region((0,0),lowest_res,slide.level_dimensions[lowest_res]))[:,:,0]
        
        binary=Mask_ROI_cl(s,disk_size,thresh=thresh,black_spots=black_spots)
        stru = [[1,1,1],[1,1,1],[1,1,1]]
        blobs, number_of_blobs = ndimage.label(binary,structure=stru)
        for i in range(1,number_of_blobs+1):
            y,x=np.where(blobs == i)
            x_0=min(x)
            y_0=min(y)
            w=max(x)-x_0
            h=max(y)-y_0               
            new_x,new_y=get_X_Y(slide,x_0,y_0,lowest_res)
            list_roi=Best_Finder_rec(slide,lowest_res,new_x,new_y,w,h,"./"+folder+"/"+folder,ref_level,list_roi,number_of_pixels_max,thresh,verbose) 
            

    elif method=='grid_etienne':
        lowest_res=len(slide.level_dimensions)-2
    
        s=np.array(slide.read_region((0,0),lowest_res,slide.level_dimensions[lowest_res]))[:,:,0]
        
        binary=Mask_ROI_cl(s,disk_size,thresh=thresh,black_spots=black_spots)
        stru = [[1,1,1],[1,1,1],[1,1,1]]
        blobs, number_of_blobs = ndimage.label(binary,structure=stru)
        for i in range(1,number_of_blobs+1):
            y,x=np.where(blobs == i)
            x_0=min(x)
            y_0=min(y)
            w=max(x)-x_0
            h=max(y)-y_0               
            new_x,new_y=get_X_Y(slide,x_0,y_0,lowest_res)
            list_roi=Best_Finder_rec(slide,lowest_res,new_x,new_y,w,h,"./"+folder+"/"+folder,ref_level,list_roi,number_of_pixels_max,thresh,verbose,marge)
            
            
    elif method=='SP_ROI':        
        if 'Tumor' in name:
            if mask_address is None:
                #print "you have not given a mask address"
                mask_address='blabla'
                raise NameError('No address alone fetching implemented yet')
            slide_mask=openslide.open_slide(mask_address)
            lowest_res=min(len(slide.level_dimensions)-2,len(slide_mask.level_dimensions)-1)
            s_m=np.array(slide_mask.read_region((0,0),lowest_res,slide_mask.level_dimensions[lowest_res]))[:,:,0]

        else:
            lowest_res=len(slide.level_dimensions)-2

        s=np.array(slide.read_region((0,0),lowest_res,slide.level_dimensions[lowest_res]))[:,:,0]
        
        if 'Tumor' not in name:
            s_m=np.zeros(shape=s.shape,dtype='uint8')
        
        binary=Mask_ROI_cl(s,disk_size,thresh=thresh,black_spots=black_spots)
        binary[binary>0]=255
        binary=computeEvaluationMask_Peter(binary,0.243,lowest_res)
        uniq, counts = np.unique(binary,return_counts=True)
        background_val = uniq[np.argmax(counts)]
        binary[binary!=background_val]=background_val+1
        binary -=background_val
        if 'Tumor' in name:
            s_m   =computeEvaluationMask_Peter(s_m   ,0.243,lowest_res)
            
            for i in range(1,max(np.unique(s_m))+1):
                indices= s_m==i
                x,y=np.where(indices)
                x_0=min(x)
                y_0=min(y)
                w=max(x)-x_0
                h=max(y)-y_0
                GT_i=s_m[x_0:x_0+w,y_0:y_0+h].copy()               
                blob_i=binary[x_0:x_0+w,y_0:y_0+h].copy()
                if not contour(GT_i,blob_i):
                    val=-1
                else:
                    val= 1
                s_m[x,y]=val
            transitions_blobs=s_m.copy()
            no_transitions_blobs=s_m.copy()
            
            transitions_blobs[transitions_blobs!=1]=0
            no_transitions_blobs[no_transitions_blobs!=-1]=0
        else:
            transitions_blobs=s_m.copy()
            no_transitions_blobs=s_m.copy()
        contour_GT=ndimage.morphology.morphological_gradient(transitions_blobs,size=(contour_size,contour_size))
        
        inside=transitions_blobs-contour_GT-no_transitions_blobs
        inside[inside<0]=0

        outside=binary-inside
        outside[outside<0]=0

        contour_binary      = ndimage.morphology.morphological_gradient(outside,size=(contour_size,contour_size))
        list_roi            = []
        mask                = np.zeros(shape=(inside.shape[0],inside.shape[1]),dtype='uint8')
        if 'Tumor' in name:
            list_inside, mask         = Sample_imagette(inside,N_squares/4,slide,ref_level,number_of_pixels_max,lowest_res,mask)
            #Peter_Ipython_plot(mask,size=5,title="Masque ou les nouveaux carres ne peuvent pas etre tirer")
            #visualise_cut(slide,list_inside,res_to_view=5,size=7,title="Interieur du ground_truth") 
            
            list_outside, mask        = Sample_imagette(outside,N_squares/4,slide,ref_level,number_of_pixels_max,lowest_res,mask)
            #Peter_Ipython_plot(mask,size=5,title="Masque ou les nouveaux carres ne peuvent pas etre tirer")
            #visualise_cut(slide,list_outside,res_to_view=5,size=7,title="en dehors du ground truth") 

            list_contour_GT, mask     = Sample_imagette(contour_GT,N_squares/4,slide,ref_level,number_of_pixels_max,lowest_res,mask)
            #Peter_Ipython_plot(mask,size=5,title="Masque ou les nouveaux carres ne peuvent pas etre tirer")
            #visualise_cut(slide,list_contour_GT,res_to_view=5,size=7,title="frontiere tumor/non tumor") 

            list_contour_binary, mask = Sample_imagette(contour_binary,N_squares/4,slide,ref_level,number_of_pixels_max,lowest_res,mask)
            #Peter_Ipython_plot(mask,size=5,title="Masque ou les nouveaux carres ne peuvent pas etre tirer")
            #visualise_cut(slide,list_contour_binary,res_to_view=5,size=7,title="frontiere tissue/fond") 

            list_roi                  = list_inside+list_outside+list_contour_GT+list_contour_binary

        else:
            list_outside, mask        = Sample_imagette(outside,N_squares/2,slide,ref_level,number_of_pixels_max,lowest_res,mask)
            #Peter_Ipython_plot(mask,size=5,title="Masque ou les nouveaux carres ne peuvent pas etre tirer")
            #visualise_cut(slide,list_outside,res_to_view=5,size=7,title="en dehors du ground truth")  

            list_contour_binary, mask = Sample_imagette(contour_binary,N_squares/2,slide,ref_level,number_of_pixels_max,lowest_res,mask)
            #Peter_Ipython_plot(mask,size=5,title="Masque ou les nouveaux carres ne peuvent pas etre tirer")
            #visualise_cut(slide,list_contour_binary,res_to_view=5,size=7,title="frontiere tissue/fond") 
            
            list_roi                  = list_outside+list_contour_binary
    else:
        raise NameError("Not known method")


    list_roi=np.array(list_roi)
    return(list_roi)






def visualise_cut(slide,list_pos,res_to_view=None,color='red',size=12,title=""):
    if res_to_view is None:
        res_to_view=slide.level_count-3
    whole_slide=np.array(slide.read_region((0,0),res_to_view,slide.level_dimensions[res_to_view]))
    max_x,max_y=slide.level_dimensions[res_to_view]
    fig = plt.figure(figsize=(size,size ))
    ax = fig.add_subplot(111, aspect='equal')
    ax.imshow(whole_slide)#,origin='lower')
    for para in list_pos:
        top_left_x,top_left_y=get_X_Y_from_0(slide,para[0],para[1],res_to_view)
        w,h=get_size(slide,para[2],para[3],para[4],res_to_view)
        p=patches.Rectangle((top_left_x,max_y-top_left_y-h), w, h, fill=False, edgecolor=color)
        p=patches.Rectangle((top_left_x,top_left_y), w, h, fill=False, edgecolor=color)
        ax.add_patch(p)
    ax.set_title(title, size=20)
    plt.show()

def make_vigra_image(X):
    if not isinstance(X,np.array([0]).__class__):
        X=np.array(X,dtype='float32')
    if len(X.shape) == 2:
        X = X.reshape((X.shape[0], X.shape[1], 1))
    elif len(X.shape) < 2 or len(X.shape) > 3:
        print 'shape of array: ', X.shape
        print 'conversion was not successful.'
        return None        
    img = vigra.taggedView(X, vigra.defaultAxistags('xyc'))
    return img
    
def from_vigra_image_make_image(X):
    if X.shape[2]==1:
        return(np.array(X).reshape(X.shape[0:2]))
    else:
        return(np.array(X))


def change_res_smil(image,wanted_size):
    if 'smilPython' in str(im1):
        sx,sy,sz=image.getSize()
        imOut = sm.Image(sx,sy)
        sm.resize(image,sx,sy,imOut)
        return(imOut)
def change_res_np(image,wanted_size,order=3):
    if "numpy" in str(type(image)):
        imOut=nd.zoom(image, wanted_size, order=order)
        return(imOut)

def predict_WSI(slide,training_res,pred_WSI_res,classifier_vaia):
    if slide is str:
        slide = openslide.open_slide(slide)

    ROI_para = ROI(name,ref_level=training_res, disk_size=4, thresh=None, black_spots=None,
                   number_of_pixels_max=1000000, verbose=False, marge=0.5, method='grid_etienne')
    WSI_pred=np.zeros(shape=(slide.level_dimensions[pred_WSI_res][0],slide.level_dimensions[pred_WSI_res][1],2))
    for para in ROI_para:
        sub_image = slide.read_region((para[0],para[1]),para[4],(para[2],para[3]))
        ### prediction  ###

        image_pred 
        to_insert = change_res_np(image_pred)
        x0, y0 = get_X_Y_from_0(slide,para[0],para[1],pred_WSI_res) 
        size_x,size_y = get_size(slide, para[2], para[3], training_res, pred_WSI_res)
        WSI_pred[x0:(x0+size_x),y0:(y0+size_y),0] += to_insert[0:size_x,0:size_y]  ###we maybe have to invert x and y
        WSI_pred[x0:(x0+size_x),y0:(y0+size_y),0] += 1

    zeros = np.where(WSI_pred[:,:,1]==0)
    WSI_pred[zeros,0] = WSI_pred[zeros,0] / WSI_pred[zeros,1]

    return(WSI_pred[:,:,0])


def subsample(Y,version,version_para):
    n = len(Y)
    val, freq = np.unique(Y, return_counts=True )
    iter_obj = [(val[i],freq[i]) for i in range(len(val))]
    if version == 'default':
        ## this is a purely random susampling
        ## checking for right arguments
        if 'n_sub' not in version_para:
            raise NameError("missing parameter n_sub in input dictionnary")
        else:
            n_val = len(val)
            n_sub = version_para['n_sub'] / n_val

        list_res = []

        for values,frequency in iter_obj:
            index_val = np.where(Y == values)[0]
            random.shuffle(index_val)
            list_res.append(index_val[0:min(n_sub,frequency)])
        res = np.concatenate(tuple(list_res))
    elif version == 'kmeans':
        ## this is a purely random susampling
        ## checking for right arguments
        if 'n_sub' not in version_para:
            raise NameError("missing parameter n_sub in input dictionnary")
        else:
            n_val = len(val)
            n_sub = version_para['n_sub'] / n_val
        if 'k' not in version_para:
            raise NameError("Missing parameter k in input dictionnary")
        else:
            k = version_para['k']
        if 'X' not in version_para:
            raise NameError('Missing data X for kmeans clustering')
            X = version_para['X']

        res = []
        for values,frequency in iter_obj:
            index_val = np.where(Y == values)[0]
            X_temp = X[index_val,:]
            kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
            groups = kmeans.fit_predict(X_temp)
            #val_, freq_ = np.unique(groups, return_counts=True)
            for id_group in range(50):
                
                index_subgroup = np.where(groups == id_group)[0]
                freq_ = len(index_subgroup)

                index_to_pic = index_val[index_subgroup]
                random.shuffle(index_to_pic)
                res += list(index_to_pic[min(freq_,n_sub)])
    return res




def from_list_string_to_list_Tumor(lists, first_part):
    res = []
    res_bis = []
    lists = lists.split(', ')
    first_el = lists[0][1::]
    last_el  = lists[-1][0:-1]
    res.append(first_el)
    for i in range(1,len(lists)-1):
        res.append(lists[i])
    res.append(last_el)
    for el in res:
        n = 3-len(el)
        res_bis.append(first_part+"_"+n*"0"+el)
    return res_bis
