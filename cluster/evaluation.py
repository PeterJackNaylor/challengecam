# -*- coding: cp1252 -*-
"""
Description: 
Challenge CAMELYON16.
Tools for evaluation.

Authors:  Vaïa Machairas, Etienne Decencière, Peter Naylor, Thomas Walter.

Creation date: 2016-02-24
"""
import os
import pdb
import numpy as np
import sklearn as sk
import smilPython as sp
import useful_functions as uf

class my_metrics(object):
    """
    Enables to evaluate the prediction for a given database.
    """
    
    def __init__(self,  db_server):
        """
        Parameter:
        db_server: segmentation data base server (see segm_db_access.py)
        """
        self._db_server = db_server
    
    def get_Precision_Recall_F_score(self,  subset='test'):
        """
        Enables to compute the F-score for a given database.
        """
        if subset=='train':
            my_subset = self._db_server.res_train()
        elif subset=='test':
            my_subset = self._db_server.res_test()
        elif subset=='val':
            my_subset = self._db_server.res_val()
        elif subset=='otsu_train':
            my_subset = self._db_server.otsu_res_train()
        elif subset=='otsu_test':
            my_subset = self._db_server.otsu_res_test()
        elif subset=='otsu_val':
            my_subset = self._db_server.otsu_res_val()
        else:
            raise TypeError('Choose the subset among train, test or val.')
        list_precision=[]
        list_recall=[]
        list_f_scores = []
        for  gt_im_list, original_image, _ ,  _ in my_subset:
            
            y_pred = np.ravel(original_image.getNumArray())
            y_pred = y_pred/np.max(y_pred)
            image_GT = gt_im_list[0] # for now, we suppose that we only have a single GT per image
            y_true = np.ravel(image_GT.getNumArray())
            y_true = y_true/np.max(y_true)
            precision_image,  recall_image,  f_score_image,  _ = sk.metrics.precision_recall_fscore_support(y_true, y_pred, beta=1.0, labels=None, pos_label=None, average=None, warn_for=('precision', 'recall', 'f-score'), sample_weight=None)
            list_precision += [precision_image[1]]
            list_recall += [recall_image[1]]
            list_f_scores += [f_score_image[1]]
        F_score = np.mean(list_f_scores)
        Precision = np.mean(list_precision)
        Recall = np.mean(list_recall)
        
        return Precision,  Recall,  F_score

    def get_Jaccard_index(self, subset = 'test'):
        """
        Enables to compute the Jaccard index for a given database.
        """
        if subset=='train':
            my_subset = self._db_server.res_train()
        elif subset=='test':
            my_subset = self._db_server.res_test()
        elif subset=='val':
            my_subset = self._db_server.res_val()
        elif subset=='otsu_train':
            my_subset = self._db_server.otsu_res_train()
        elif subset=='otsu_test':
            my_subset = self._db_server.otsu_res_test()
        elif subset=='otsu_val':
            my_subset = self._db_server.otsu_res_val()
        else:
            raise TypeError('Choose the subset among train, test or val.')
        index_list = []
        for  gt_im_list, original_image, _ ,  _ in my_subset:
            
            copie8 = sp.Image(gt_im_list[0],  "UINT8")
            sp.test(gt_im_list[0]>0,  255,  0,  copie8)
            copie8orig = sp.Image(original_image,  "UINT8")
            sp.test(original_image>0,  255,  0,  copie8orig)
            im_union = sp.Image(copie8)
            im_inter = sp.Image(copie8)
            sp.sup(copie8orig, copie8, im_union)
            meas_union = sp.vol(im_union)
            im_inter = sp.Image(copie8)
            sp.inf(copie8orig, copie8,  im_inter)
            meas_inter = sp.vol(im_inter)
            index_list += [meas_inter/float(meas_union)]
        return np.mean(index_list)


    def overall_pixel_accuracy(self,  subset='test'):
        """
        Enables to compute the overall pixel accuracy for a given database.
        """
        if subset=='train':
            my_subset = self._db_server.res_train()
        elif subset=='test':
            my_subset = self._db_server.res_test()
        elif subset=='val':
            my_subset = self._db_server.res_val()
        elif subset=='otsu_train':
            my_subset = self._db_server.otsu_res_train()
        elif subset=='otsu_test':
            my_subset = self._db_server.otsu_res_test()
        elif subset=='otsu_val':
            my_subset = self._db_server.otsu_res_val()
        else:
            raise TypeError('Choose the subset among train, test or val.')
        index_list = []
        for  gt_im_list, original_image, _,  _ in my_subset:
            copie8 = sp.Image(gt_im_list[0],  "UINT8")
            sp.test(gt_im_list[0]>0,  255,  0,  copie8)
            copie8orig = sp.Image(original_image,  "UINT8")
            sp.test(original_image>0,  255,  0,  copie8orig)
            im_eq = sp.Image(copie8orig)
            sp.compare(copie8orig,  "==", copie8,  1,  0,  im_eq)
            meas_vol = sp.vol(im_eq)
            nb_pix = im_eq.getSize()[0] * im_eq.getSize()[1]
            index_list += [meas_vol/float(nb_pix)]
        return np.mean(index_list)
            
    def visualization_TFPN(self, subset='test'):
        """
        Enables to compute the predicted image with different colors for TP, FP, TN and FN.
        """
        if subset=='train':
            my_subset = self._db_server.res_train()
            save_folder_dir = os.path.join(self._db_server.get_input_dir(), "resultats",  "TFPN",  "train") 
        elif subset=='test':
            my_subset = self._db_server.res_test()
            save_folder_dir = os.path.join(self._db_server.get_input_dir(), "resultats",  "TFPN",  "test") 
        elif subset=='val':
            my_subset = self._db_server.res_val()
            save_folder_dir = os.path.join(self._db_server.get_input_dir(), "resultats",  "TFPN",  "val") 
        elif subset=='otsu_train':
            my_subset = self._db_server.otsu_res_train()
        elif subset=='otsu_test':
            my_subset = self._db_server.otsu_res_test()
        elif subset=='otsu_val':
            my_subset = self._db_server.otsu_res_val()
        else:
            raise TypeError('Choose the subset among train, test or val.')
        for  gt_im_list, original_image, name in my_subset:
            copie8 = sp.Image(gt_im_list[0],  "UINT8")
            sp.test(gt_im_list[0]>0,  255,  0,  copie8)
            copie8orig = sp.Image(original_image,  "UINT8")
            sp.test(original_image>0,  255,  0,  copie8orig)
            imout = uf.visu_TFPN(copie8orig,  copie8)
            sp.write(imout, os.path.join(save_folder_dir,name))
        return
        
    def computation_TFPN(self, subset='test'):
        """
        Enables to compute the predicted image with different colors for TP, FP, TN and FN.
        """
        if subset=='train':
            my_subset = self._db_server.res_train()
            save_folder_dir = os.path.join(self._db_server.get_input_dir(), "resultats",  "TFPN",  "train") 
        elif subset=='test':
            my_subset = self._db_server.res_test()
            save_folder_dir = os.path.join(self._db_server.get_input_dir(), "resultats",  "TFPN",  "test") 
        elif subset=='val':
            my_subset = self._db_server.res_val()
            save_folder_dir = os.path.join(self._db_server.get_input_dir(), "resultats",  "TFPN",  "val") 
        elif subset=='otsu_train':
            my_subset = self._db_server.otsu_res_train()
        elif subset=='otsu_test':
            my_subset = self._db_server.otsu_res_test()
        elif subset=='otsu_val':
            my_subset = self._db_server.otsu_res_val()
        else:
            raise TypeError('Choose the subset among train, test or val.')
        
        dico={'tp_pre':0, 'fp_pre':0, 'tp_rec':0,  'fp_rec':0,  'acc':0,  'im_size':0,  'inter':0,  'union':0}
        for  gt_im_list, original_image, name in my_subset:
            copie8 = sp.Image(gt_im_list[0],  "UINT8")
            sp.test(gt_im_list[0]>0,  255,  0,  copie8)
            copie8orig = sp.Image(original_image,  "UINT8")
            sp.test(original_image>0,  255,  0,  copie8orig)
            ## accuracy:
            imacc = sp.Image(copie8)
            sp.compare(copie8, "==",  copie8orig,  1, 0, imacc)
            dico['acc'] += sp.vol(imacc)
            dico['im_size'] += copie8.getSize()[0] * copie8.getSize()[1]
            ## Jaccard index:
            iminf = sp.Image(copie8)
            sp.inf(copie8,  copie8orig,  iminf)
            dico['inter'] += sp.vol(iminf) / float(255)
            imsup = sp.Image(copie8)
            sp.sup(copie8,  copie8orig,  imsup)
            dico['union'] += sp.vol(imsup)/ float(255)
            ## for precision and recall:
            im_dil = sp.Image(copie8)
            sp.dilate(copie8, im_dil, 2)
            tp_pre, fp_pre = uf.compute_TFPN(copie8orig, im_dil)
            dico['tp_pre'] += tp_pre
            dico['fp_pre'] += fp_pre
            sp.dilate(copie8orig, im_dil, 2)
            tp_rec, fp_rec = uf.compute_TFPN(copie8, im_dil)
            dico['tp_rec'] += tp_rec
            dico['fp_rec'] += fp_rec
            ##
            
        return dico
    

    def number_of_connected_components(self,  subset='test'):
        """
        Enables to compute the average number of connected components in the subset image base.
        """
        if subset=='train':
            my_subset = self._db_server.res_train()
            save_folder_dir = os.path.join(self._db_server.get_input_dir(), "resultats",  "labels",  "train") 
        elif subset=='test':
            save_folder_dir = os.path.join(self._db_server.get_input_dir(), "resultats",  "labels",  "test")
            my_subset = self._db_server.res_test()
        elif subset=='val':
            my_subset = self._db_server.res_val()
            save_folder_dir = os.path.join(self._db_server.get_input_dir(), "resultats",  "labels",  "val")
        elif subset=='otsu_train':
            my_subset = self._db_server.otsu_res_train()
        elif subset=='otsu_test':
            my_subset = self._db_server.otsu_res_test()
        elif subset=='otsu_val':
            my_subset = self._db_server.otsu_res_val()
        else:
            raise TypeError('Choose the subset among train, test or val.')
        number_list = []
        for  gt_im_list, original_image, name in my_subset: ###probleme avec original_image, it is a fucking list!!!
            pdb.set_trace()            
            copie8orig = sp.Image(original_image,  "UINT8")
            sp.test(original_image>0,  255,  0,  copie8orig)
            imlabel = sp.Image(copie8orig,  'UINT16')
            sp.label(copie8orig,  imlabel)
            sp.write(imlabel, os.path.join(save_folder_dir,name))
            number = sp.maxVal(imlabel)
            number_list += [number]
        return round(np.mean(number_list), 2)
