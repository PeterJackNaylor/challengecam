# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:11:11 2016

@author: naylor
"""
import os
import pdb
import matplotlib
import numpy as np
import openslide
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk
from skimage.filters import threshold_otsu
from scipy import ndimage
import random
import pandas as pd
import vigra



def get_X_Y(slide,x_0,y_0,level):
    ###gives you the coordinates for the level 0 image for a given couple of pixel

    size_x_0=slide.level_dimensions[level][0]
    size_y_0=slide.level_dimensions[level][1]
    size_x_1=float(slide.level_dimensions[0][0])
    size_y_1=float(slide.level_dimensions[0][1])
  
    x_1=x_0*size_x_1/size_x_0
    y_1=y_0*size_y_1/size_y_0
    
    return int(x_1),int(y_1)                
                
def get_size(slide,size_x,size_y,level_from,level_to):
    ## gives comparable size from one level to an other    
    
    ds = slide.level_downsamples
    scal=float(ds[level_from])/ds[level_to]
    size_x_new = int(float(size_x)*scal)
    size_y_new = int(float(size_y)*scal)
    return(size_x_new,size_y_new)
                
def variability_val(crop_array):
    val=np.std(crop_array[:,:,0])+np.std(crop_array[:,:,1])+np.std(crop_array[:,:,2])
    return(val/3.)
    
    
    
### He goes and find the mask adresse on his own, it has to be close to the tumor image
def Worst_Slicer(name,lamb,ref_level=0,Mask_=False):        
    if '/' in name:
        cut=name.split('/')[-1]
        folder=cut.split('.')[0]
    else:
        folder=name.split(".")[0]
    if Mask_:
        pieces=name.split('/')[:-2]
        folder_mask=folder+"_Mask"
        Mask_adresse=""
        for i in range(len(pieces)):
            Mask_adresse+=pieces[i]+"/"
        Mask_adresse+=folder_mask.split("_")[0]+"_Mask"+"/"+folder_mask+".tif"
    else: 
        Mask_adresse=None
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    slide = openslide.open_slide(name)
        
    level =slide.level_count-2
    size_x=int(slide.level_dimensions[level][0])
    size_y=int(slide.level_dimensions[level][1])
    
    Best_Slicer_rec(slide,level,0,0,size_x,size_y,lamb,"./"+folder+"/"+folder,ref_level,Mask_adresse)


def Best_Slicer_rec(slide,level,x_0,y_0,size_x,size_y,lamb,image_name,ref_level,Mask_adresse=None):
    if level==ref_level:          
        if size_x*size_y<1000000: ##size of level 3
            croped=slide.read_region((x_0,y_0), level, (size_x,size_y) )
            test=variability_val(np.array(croped))
            if test>lamb:
                croped.save(image_name+"_"+str(x_0)+"_"+str(y_0)+".png")
                if Mask_adresse is not None:
                    slide_mask = openslide.open_slide(Mask_adresse)
                    croped_mask=slide_mask.read_region((x_0,y_0), level, (size_x,size_y) )
                    croped_mask.save(image_name+"_"+str(x_0)+"_"+str(y_0)+"_Mask"+".png")
        else:
                        
            size_x_new=int(size_x*0.5)
            size_y_new=int(size_y*0.5)
                
            diese_str="#"*level*10
            print diese_str +"split level "+ str(level)
                            
            x_1=x_0+size_x_new
            y_1=y_0+size_y_new
            
            image_name=image_name+"_Split_id_"+str(random.randint(0, 1000))            
            Best_Slicer_rec(slide,level,x_0,y_0,size_x_new,size_y_new,lamb,image_name,ref_level,Mask_adresse)
            Best_Slicer_rec(slide,level,x_1,y_0,size_x_new,size_y_new,lamb,image_name,ref_level,Mask_adresse)
            Best_Slicer_rec(slide,level,x_0,y_1,size_x_new,size_y_new,lamb,image_name,ref_level,Mask_adresse)
            Best_Slicer_rec(slide,level,x_1,y_1,size_x_new,size_y_new,lamb,image_name,ref_level,Mask_adresse)

    else:
        croped=slide.read_region((x_0,y_0), level, (size_x,size_y) )
        test=variability_val(np.array(croped))       
        
        if test>lamb or level > 1:
            if size_x*size_y>1000000: ##size of level 3
                
                size_x_new,size_y_new=get_size(slide,size_x,size_y,level,level-1)
                size_x_new=int(size_x_new*0.5)
                size_y_new=int(size_y_new*0.5)
                
                diese_str="#"*level*10
                print diese_str +"split level "+ str(level)
                
                width_x_0,height_y_0=get_size(slide,size_x,size_y,level,0)
                
                x_1=x_0+int(width_x_0*0.5)
                y_1=y_0+int(height_y_0*0.5)
                
                Best_Slicer_rec(slide,level-1,x_0,y_0,size_x_new,size_y_new,lamb,image_name,ref_level,Mask_adresse)
                Best_Slicer_rec(slide,level-1,x_1,y_0,size_x_new,size_y_new,lamb,image_name,ref_level,Mask_adresse)
                Best_Slicer_rec(slide,level-1,x_0,y_1,size_x_new,size_y_new,lamb,image_name,ref_level,Mask_adresse)
                Best_Slicer_rec(slide,level-1,x_1,y_1,size_x_new,size_y_new,lamb,image_name,ref_level,Mask_adresse)
                
            else:
                
                size_x_new,size_y_new=get_size(slide,size_x,size_y,level,level-1)
                Best_Slicer_rec(slide,level-1,x_0,y_0,size_x_new,size_y_new,lamb,image_name,ref_level,Mask_adresse)
                
        else:
            print "Not enough variability on second split"
            

def Mask_ROI_op(im,disk_size,thresh=None,black_spots=None,with_morph=False):
    l=np.array([0,0])
    if not isinstance(im,l.__class__):
        numpy_array=np.array(im)
    else:
        numpy_array=im 
    if len(numpy_array.shape)==3:
        numpy_array=numpy_array[:,:,0:3].mean(axis=2)        
    selem = disk(disk_size)
    openin = opening(numpy_array, selem)
    if thresh is None:
        thresh = threshold_otsu(openin)
    binary = openin > thresh
    if binary.dtype=='bool':
        binary=binary+0
    if black_spots is not None:    
        binary2 = openin > black_spots
        binary2 = binary2 + 0
        binary = binary - binary2 
    else:
        binary -=1
    binary=binary * -1
    if with_morph:
        return(binary,openin)
    else:
        return(binary)                     
        

def Mask_ROI_cl(im,disk_size,thresh=None,black_spots=None,with_morph=False):
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

def Cut_ROI(name,ref_level=4,disk_size=4,Mask_=False):
    if '/' in name:
        cut=name.split('/')[-1]
        folder=cut.split('.')[0]
    else:
        folder=name.split(".")[0] 
    if Mask_:
        pieces=name.split('/')[:-2]
        folder_mask=folder+"_Mask"
        Mask_adresse=""
        for i in range(len(pieces)):
            Mask_adresse+=pieces[i]+"/"
        Mask_adresse+=folder_mask.split("_")[0]+"_Mask"+"/"+folder_mask+".tif"
    else:
        Mask_adresse=None
    if not os.path.exists(folder):
        os.makedirs(folder)
    slide = openslide.open_slide(name)
    
    lowest_res=slide.level_count-2
    s=np.array(slide.read_region((0,0),lowest_res,slide.level_dimensions[lowest_res]))[:,:,1]
    
    binary=Mask_ROI_cl(s,disk_size,220)
    
    stru = [[1,1,1],[1,1,1],[1,1,1]]
    blobs, number_of_blobs = ndimage.label(binary,structure=stru)   
    
    for i in range(1,number_of_blobs):
        y,x=np.where(blobs == i)
        x_0=min(x)
        y_0=min(y)
        w=max(x)-x_0
        h=max(y)-y_0               
        
        new_x,new_y=get_X_Y(slide,x_0,y_0,lowest_res)
        Best_Slicer_rec(slide,lowest_res,new_x,new_y,w,h,-1,"./"+folder+"/"+folder,ref_level,Mask_adresse)

def Best_Finder_rec(slide,level,x_0,y_0,size_x,size_y,lamb,image_name,ref_level,list_roi,number_of_pixels_max,verbose):
    if size_x*size_y==0:
        print 'Warning: width or height is null..'
        return([])
    else:
        if level==ref_level:          
            if size_x*size_y<number_of_pixels_max: ##size of level 3
                croped=slide.read_region((x_0,y_0), level, (size_x,size_y) ) ## maybe can be removed? 
                test=variability_val(np.array(croped))
                
                if test>lamb:
                    list_roi.append([x_0,y_0,size_x,size_y,level])
                    return(list_roi)
            else:
                            
                size_x_new=int(size_x*0.5)
                size_y_new=int(size_y*0.5)
                    
                diese_str="#"*level*10
                if verbose:
                    print diese_str +"split level "+ str(level)
                                
                x_1=x_0+size_x_new
                y_1=y_0+size_y_new
                
                image_name=image_name+"_Split_id_"+str(random.randint(0, 1000))            
                list_roi=Best_Finder_rec(slide,level,x_0,y_0,size_x_new,size_y_new,lamb,image_name,ref_level,list_roi,number_of_pixels_max,verbose)
                list_roi=Best_Finder_rec(slide,level,x_1,y_0,size_x_new,size_y_new,lamb,image_name,ref_level,list_roi,number_of_pixels_max,verbose)
                list_roi=Best_Finder_rec(slide,level,x_0,y_1,size_x_new,size_y_new,lamb,image_name,ref_level,list_roi,number_of_pixels_max,verbose)
                list_roi=Best_Finder_rec(slide,level,x_1,y_1,size_x_new,size_y_new,lamb,image_name,ref_level,list_roi,number_of_pixels_max,verbose)
                return(list_roi)
        else:
            croped=slide.read_region((x_0,y_0), level, (size_x,size_y) )
            test=variability_val(np.array(croped))       
            
            if test>lamb or level > 1:
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
                    
                    list_roi=Best_Finder_rec(slide,level-1,x_0,y_0,size_x_new,size_y_new,lamb,image_name,ref_level,list_roi,number_of_pixels_max,verbose)
                    list_roi=Best_Finder_rec(slide,level-1,x_1,y_0,size_x_new,size_y_new,lamb,image_name,ref_level,list_roi,number_of_pixels_max,verbose)
                    list_roi=Best_Finder_rec(slide,level-1,x_0,y_1,size_x_new,size_y_new,lamb,image_name,ref_level,list_roi,number_of_pixels_max,verbose)
                    list_roi=Best_Finder_rec(slide,level-1,x_1,y_1,size_x_new,size_y_new,lamb,image_name,ref_level,list_roi,number_of_pixels_max,verbose)
                    return(list_roi)
                else:
                    
                    size_x_new,size_y_new=get_size(slide,size_x,size_y,level,level-1)
                    list_roi=Best_Finder_rec(slide,level-1,x_0,y_0,size_x_new,size_y_new,lamb,image_name,ref_level,list_roi,number_of_pixels_max,verbose)
                    return(list_roi)
            else:
                print "Not enough variability on second split"

def ROI(name,ref_level=4,disk_size=4,thresh=None,black_spots=None,number_of_pixels_max=1000000,verbose=False):   
    if '/' in name:
        cut=name.split('/')[-1]
        folder=cut.split('.')[0]
    else:
        folder=name.split(".")[0] 
    slide = openslide.open_slide(name)
    
    lowest_res=slide.level_count-2
    s=np.array(slide.read_region((0,0),lowest_res,slide.level_dimensions[lowest_res]))[:,:,0]
    
    binary=Mask_ROI_cl(s,disk_size,thresh=thresh,black_spots=black_spots)
    
    stru = [[1,1,1],[1,1,1],[1,1,1]]
    
    blobs, number_of_blobs = ndimage.label(binary,structure=stru)   
    list_roi=[]  ### pd.DataFrame(columns=['x_0','y_0','w','h','res'])
    for i in range(1,number_of_blobs):
        y,x=np.where(blobs == i)
        x_0=min(x)
        y_0=min(y)
        w=max(x)-x_0
        h=max(y)-y_0               
        new_x,new_y=get_X_Y(slide,x_0,y_0,lowest_res)
        list_roi=Best_Finder_rec(slide,lowest_res,new_x,new_y,w,h,-1,"./"+folder+"/"+folder,ref_level,list_roi,number_of_pixels_max,verbose)
    list_roi=np.array(list_roi)
    return(list_roi)
    
    
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
    