# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 10:49:17 2016

@author: naylor
"""

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import vigra
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score
import pdb
import openslide
from BasicOperations import make_vigra_image,Mask_ROI_op
import random

def GetImage(c,para):
    if len(para)!=5:
        print "Not enough parameters..."
    else:
        sample=openslide.open_slide(c).read_region((para[0],para[1]),para[4],(para[2],para[3]))
        return(sample)
    
def Stratefied_subsampling(y,percentage):
    n_get=int(y.shape[0]*percentage*0.5)
    s=y.value_counts()
    sub_index=[]          
    for id_ in s.index:
        val_=int(s[id_])
        n_get=min([n_get,val_])
    for id_ in s.index:
        y_temp=y.ix[y==id_]
        if len(y_temp.index)==n_get:
            sub_index=sub_index+y_temp.index.values.tolist()
        else:
            sub_index=sub_index+random.sample(y_temp.index,n_get)
    return(sub_index)

class Filter(object):
    
    def __init__(self, filter_fun ,args,name):
        self.filter=filter_fun
        self.args=args
        self.name=name
        
    def apply_filter(self,image):
        
        return(self.filter(image,**self.args))
        
class RandomForest_PixelClassification(RandomForestClassifier):
    def __init__(self, n_range,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 n_estimators=10):
        super(RandomForest_PixelClassification, self).__init__(  bootstrap=bootstrap, class_weight=class_weight, criterion=criterion,
                                                        max_depth=max_depth, max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                                                        min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                                        min_weight_fraction_leaf=min_weight_fraction_leaf, n_estimators=n_range[0], n_jobs=n_jobs,
                                                        oob_score=oob_score, random_state=random_state, verbose=verbose,
                                                        warm_start=warm_start)
        self.n_range=n_range
        self.MSE=None
    def image_to_data(self,image,list_filters,disk_size=None,Mask=None,thresh=160,return_ROI=False):
        #### takes a smaller image in order to launch it to transform it into a dataset.
        #it only returns instances that are in the ROI with open and a disk size of size 20
        if Mask is None:
            mask_val=0
        else:
            mask_val=len(np.unique(Mask))
        if mask_val>2:
            print "Mask is not a binary image"
        else:
            n_c=image.shape[2]
            data=pd.DataFrame(index=range(image.shape[0]*image.shape[1]))
            for filter_ in list_filters:
                f=filter_.apply_filter(image)
                for c in range(min([n_c,3])):
                    f_c=f[:,:,c].flatten()
                    data[filter_.name+"_dim_"+str(c)]=f_c
            for c in range(min([n_c,3])):
                f_c=image[:,:,c].flatten()
                data["Pixel"+"_dim_"+str(c)]=f_c
            if Mask is not None:
                data['y']=Mask[:,:,0].flatten()
            if disk_size is not None:
                im_ROI=Mask_ROI_op(np.array(image)[:,:,0],disk_size,thresh=thresh)
                if len(np.unique(im_ROI))==2:
                    ROI=im_ROI.flatten()
                    data=data.ix[ROI>0]
            if return_ROI:
                return(data,ROI)
            else:
                return(data)
            
    def training_one_image(self,slide,pos_file,list_filters,mask):
        for para in pos_file:
            sample=GetImage(slide,para)
            sample_mask=GetImage(mask,para)
            data=self.image_to_data(make_vigra_image(sample),list_filters,Mask=make_vigra_image(sample_mask))
            data[data['y']>0,'y']=1
            y=data['y']
            names=[el for el in data.columns if el!='y']
            self.fit(data[names],y)
    def training_one_image_cv(self,image,pos_file,list_filters,mask,cv):
        ### Doesn't train the real model.
        skf=KFold(len(pos_file), n_folds=cv, shuffle=True)
        f1_test_mat=np.zeros(shape=(cv,len(self.n_range)))
        ct_skf=0
        for train,test in skf:
            pos_train=pos_file[train]
            pos_test =pos_file[test]
            ct_n_es=0
            for n_es in self.n_range:
                rf=RandomForestClassifier(n_estimators=n_es)
                for para in pos_train:
                    sample=GetImage(image,para)
                    sample_mask=GetImage(mask,para)
                    data=self.image_to_data(make_vigra_image(sample),list_filters,Mask=make_vigra_image(sample_mask))
                    data.ix[data['y']>0,'y']=1
                    y=data['y']
                    names=[el for el in data.columns if el!='y']
                    rf.fit(data[names],y)
                f1_test=np.array([0]*len(pos_test))
                ct_pos_test=0
                for para in pos_test:
                    sample=GetImage(image,para)
                    sample_mask=GetImage(mask,para)
                    data=self.image_to_data(make_vigra_image(sample),list_filters,Mask=make_vigra_image(sample_mask))
                    data.ix[data['y']>0,'y']=1
                    y=data['y']
                    names=[el for el in data.columns if el!='y']
                    y_pred=rf.predict(data[names])
                    f1_test[ct_pos_test]=f1_score(y,y_pred,pos_label=1, average='binary')
                    ct_pos_test+=1
                f1_test_mat[ct_skf,ct_n_es]=np.mean(f1_test)
                ct_n_es+=1
            ct_skf+=1
        return(f1_test_mat)
    
    def image_training_set(self,slide,pos_file,list_filters,disk_size,percentage,mask=None):
        
        for para in pos_file:
            sample=GetImage(slide,para)
            if mask is not None:
                sample_mask=GetImage(mask,para)
                data_temp=self.image_to_data(make_vigra_image(sample),list_filters,disk_size=20,Mask=make_vigra_image(sample_mask))
                data_temp.ix[data_temp['y']>0,'y']=1
            else:
                data_temp=self.image_to_data(make_vigra_image(sample),list_filters,disk_size=20,Mask=None)
                data_temp['y']=0
            y=data_temp['y']
            sub_index=Stratefied_subsampling(y,percentage)
            data_temp=data_temp.ix[sub_index,data_temp.columns]
            if 'data' not in locals():
                data=data_temp.copy()
            else:
                data=data.append(data_temp,ignore_index=True)

        return(data)
def predict_mask_image(slide,pos_file,list_filters,disk_size):
    for para in pos_file:
        sample=GetImage(slide,para)
        