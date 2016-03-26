# -*- coding: cp1252 -*-
"""
Description: 
Challenge CAMELYON16.
This file contains the class PixelClassification 
which is used to learn a segmentation model by pixel from a set of reference segmentations, and apply it to other images.

Authors:  Vaïa Machairas, Etienne Decencière, Peter Naylor, Thomas Walter.

Creation date: 2016-02-24
"""
import time
import pdb
import os
import numpy as np
from debug_mode import debug_mess
import cPickle as pickle
import random
import smilPython as sp
import useful_functions as uf
##--------------------------------------
##--------------------------------------
class LearnSegmentation(object):
    """
    Base class for PixelClassification and SuperpixelClassification.
    """

    def __init__(self, db_server, output_folder, features_list, nb_samples=None):
        """
        Parameters:
        db_server: segmentation data base server (see segm_db_access.py)
        output_folder (string): path to resulting segmentations/classifications.
        features_list (list): a list of instances of feature classes. The feature class must offer:
            * a __call__ method which takes as input a single SMIL image, and computes a matrix (see note in features_for_classification.py)
            * a name attribute which gives the official name for the feature.
        nb_samples: total number of samples to be used. If equal to None, than all uc are taken into account (which might be *a lot*
           when they are pixels)
        """
        self._db_server = db_server
       # if os.path.isdir(output_folder) is False:
       #     raise OSError("Could  find output directory. Stopping process.")
        self._output_folder = output_folder
        self._features_list = features_list
        self._nb_samples = nb_samples
        self.classifier = None


    def save_model(self, filename):
        file_stream = open(filename, "wb")
        pickle.dump(self.classifier, file_stream)
        file_stream.close()


    def load_model(self, filename):
        file_stream = open(filename, "r")
        self.classifier = pickle.load(file_stream)
        file_stream.close()


    def transform_image_uc(self, original_image):
        """Compute classification units (uc) image."""
        raise NotImplementedError("transfor_image_uc() should be implemented in derived classes")


    def get_X_per_image_with_save_3_original(self,  original_image,  original_image_name, folder_sauv_path,  image_sauv_path):
        """
        morceau de code en commun pour sauvegarder les features déjà calculés et gagner du temps.
        deuxième version: on sauve:
        - les vecteurs en une matrice dont la ligne correspond à un feature. (format .npy)
        - un dictionnaire qui à chaque nom de feature (string) associe le numéro de la ligne dans la matrice précédente. (format .pickle) 
        """
        orig_name = '.'.join(original_image_name.split('.')[:-1])
        ##
        if os.path.isdir(folder_sauv_path) is not True:
            try:
                os.mkdir(folder_sauv_path)
            except OSError:
                print "Could not make save folder dir"
                raise 
        if os.path.isdir(image_sauv_path) is not True:
            try:
                os.mkdir(image_sauv_path)
            except OSError:
                print "Could not make save uc folder dir"
                raise 
        ##
        image_sauv_name_pickle = os.path.join(image_sauv_path ,  orig_name + ".pickle")
        image_sauv_name_npy = os.path.join(image_sauv_path ,  orig_name + ".npy")
        if os.path.isfile(image_sauv_name_pickle) is not True:
            try:
                dic_image = {}
                fichier = open(image_sauv_name_pickle, 'w')
                pickle.dump(dic_image, fichier)
                fichier.close()
            except OSError:
                print "Could not make image save pickle file "
                raise
        im_pickle = open(image_sauv_name_pickle,  'r')
        dico_image = pickle.load(im_pickle)
        im_pickle.close()
        sauv_dico = False
        try:
            matrix_npy = np.load(image_sauv_name_npy)
        except:
            matrix_npy = None
        ##
        X = None
        for feature in self._features_list:
            if original_image.getTypeAsString()=="UINT8" or original_image.getTypeAsString()=="UINT16":
                feature._channels_list = [0]
            list_feat_per_channel = feature.get_name()
            #print feature.get_name()
            try:
                X_temp = None
                for i in range(len(list_feat_per_channel)):
                    column = dico_image[list_feat_per_channel[i]]
                    x = matrix_npy[column]
                    X_temp = uf.my_concatenation(X_temp, x)
                X = uf.my_concatenation(X,  X_temp)
            except:
                sauv_dico = True
                x = feature.__call__(original_image)
                 #original_image.getSize() = (384,704,3) and x.shape=(811008,)  (==(384*703*3,))
                X = uf.my_concatenation(X, x)
                save_length = len(dico_image)
                for j in range(len(list_feat_per_channel)):
                    dico_image[list_feat_per_channel[j]] = save_length + j
                    if len(list_feat_per_channel)==1:
                        matrix_npy = uf.my_concatenation(matrix_npy,  x)
                    else:
                        matrix_npy = uf.my_concatenation(matrix_npy,  x[j, :])
        #print "taille X",  X.shape
        pdb.set_trace()
        if sauv_dico == True:
            pdb.set_trace()
            print "save new dict"
            im_pickle = open(image_sauv_name_pickle,  'w')
            pickle.dump(dico_image,  im_pickle)
            im_pickle.close()
            print "save new matrix"
            np.save(image_sauv_name_npy,  matrix_npy)
        #####   à enlever ensuite: pour etude temporaire des features si image ndg
##        for i in range(X.shape[0]):
##            im_feature = sp.Image(original_image)
##            im_feature_arr = np.transpose(im_feature.getNumArray())
##            im_arr = np.transpose(np.reshape(X[i, :],  (original_image.getSize()[1],  original_image.getSize()[0])))
##            for k in range(im_arr.shape[0]):
##                for j in range(im_arr.shape[1]):
##                    im_feature_arr[k, j] = im_arr[k, j]
##            im_feature.save('images/visu_im_'+original_image_name+'_cy_'+str(i+1)+'.png')
        #####

        return X.transpose()


    def get_X_per_image_with_save_3(self,  original_image,  original_image_name, folder_sauv_path,  image_sauv_path):
        """
        morceau de code en commun pour sauvegarder les features déjà calculés et gagner du temps.
        deuxième version: on sauve:
        - les vecteurs en une matrice dont la ligne correspond à un feature. (format .npy)
        - un dictionnaire qui à chaque nom de feature (string) associe le numéro de la ligne dans la matrice précédente. (format .pickle) 
        """
        orig_name = '.'.join(original_image_name.split('.')[:-1])
        ## Pour trouver la taille de X:
        nb_samples = original_image.getSize()[0] * original_image.getSize()[1]
        lili = []
        for feature in self._features_list:
            if original_image.getTypeAsString()=="UINT8" or original_image.getTypeAsString()=="UINT16":
                feature._channels_list = [0]
            lili += feature.get_name()
        nb_features = len(lili)
        X = np.zeros((nb_features,  nb_samples))
        #index_lign_free = 0
        #matrix_npy = np.zeros((nb_features,  nb_samples))
        #index_lign_free_matrix_npy = 0
        index = 0

	##
        dico_image = {}
        #matrix_npy = None
        start_time = time.time()
	init_time = start_time
        for feature in self._features_list:
            if original_image.getTypeAsString()=="UINT8" or original_image.getTypeAsString()=="UINT16":
                feature._channels_list = [0]
            list_feat_per_channel = feature.get_name()
            #print feature.get_name()
            sauv_dico = True
            x = feature.__call__(original_image)
            feat_time = time.time() - start_time
            start_time = time.time()
            #original_image.getSize() = (384,704,3) and x.shape=(811008,)  (==(384*703*3,))
            #X = uf.my_concatenation(X, x)
            X[index:(index+x.shape[0])] = x[:,:]
            index += x.shape[0]
            save_length = len(dico_image)
            for j in range(len(list_feat_per_channel)):
                dico_image[list_feat_per_channel[j]] = save_length + j
            #    if len(list_feat_per_channel)==1:
            #        matrix_npy = uf.my_concatenation(matrix_npy,  x)
            #    else:
            #        matrix_npy = uf.my_concatenation(matrix_npy,  x[j, :])
            update_time = time.time() - start_time
            start_time = time.time()
            #print '%s : calc = %i seconds\tupdate = %i seconds' % (feature.get_name(), int(feat_time), int(update_time))
        #print "taille X",  X.shape
	final_diff_time = time.time() - init_time
	print 'time elapsed (feature calculation): %02i:%02i' % ((final_diff_time/60),  (final_diff_time%60))
        return X.transpose(),dico_image


    def get_Y_per_image(self, image_uc_lab, image_GT):
        """Compute the vector of labels of the image."""
        raise NotImplementedError("get_Y_per_image() should de defined in derived classes.")


    def subsample(self, X, Y, code):
        labels = np.unique(Y)
        nb_labels = len(labels)
        #nb_samples_per_image_per_label = self._nb_samples / self._db_server.nb_im(code) / nb_labels
        # I modified this, because we need the number of samples to be determined before calling this function
        # in particular it should not depend on the number of labels (this is to avoid np.concatenate).
        # I am aware of the bias this generates. 
        nb_samples_per_image_per_label = self._nb_samples / nb_labels
        X_sample = None
        for label in labels:
            mask_index = [i for i in np.arange(len(Y)) if Y[i]==label]
            np.random.seed(42)
            mask_random = np.int32(np.floor(np.random.random(nb_samples_per_image_per_label)*len(mask_index)))
            X_tmp = X[mask_index][mask_random]
            Y_tmp = Y[mask_index][mask_random]
            if X_sample is None:
                X_sample = X_tmp
                Y_sample = Y_tmp
            else:
                if len(X_sample.shape)==1:
                    X_sample = X_sample[:, np.newaxis]
                if len(X_tmp.shape)==1:
                    X_tmp = X_tmp[:, np.newaxis]
                X_sample = np.vstack((X_sample, X_tmp))
                Y_sample = np.vstack((Y_sample, Y_tmp))
                
        return X_sample, Y_sample

    def deal_with_missing_values_2(self,  X):
        """
        Meme idée que deal_with_missing_values mais avec une autre stratégie de remplacement des valeurs:
        lorsqu'il y a une valuer manquante dans la ligne i, on cherche quelles sont les cinque autres lignes (uc) 
        qui sont les plus proches (au sens des moindres carrés sur les autres features) puis on remplace la valeur manquante par leur moyenne.
        """
        ##X=np.array([[1,2,3],[4,5,None], [7,8,9],  [4, 4, 4],[None,10,11]])## example for testing
        for i in range(X.shape[0]):
            if sorted(X[i, :])[0] == None:
                dico = {}
                for ii in range(X.shape[0]):
                    if ii != i and sorted(X[ii, :])[0] != None:
                        dico[ii] = uf.SoS(X[i, :],  X[ii, :])
                nb_of_neighbors = min(2,  len(dico))
                new_dico = sorted(dico.items(),  key = uf.my_func_to_sort_dico) ## c'est une liste
                ## maintenant faire la moyenne des valeurs sur les lignes d'indices trouvés pour un feature donné 
                list_indexes = [new_dico[0:nb_of_neighbors][k][0] for k in range(nb_of_neighbors)]
                mat = None
                for index in list_indexes:
                    mat = uf.my_concatenation(mat,  X[index, :])
                for j in range(X.shape[1]):
                    if X[i, j] == None:
                        X[i, j] = np.mean(mat[:, j])
                        print "We had to replace the missing value by the mean of corresponding values in the five most similar samples."
        return X

    def get_X_Y_for_train(self, code, first=None, last=None, N_squares=16, export=False):
        """
        This method enables to compute the vector of labels
        of all classifcation units in all images of the training set.

        Output:
        Y_train (ndarray): vector of size (Mx1) where M is the number of
        classification units in the subset train of images.
        """
        X_train = None
        Y_train = None

        # get the number of samples
        # we loop, as some images might not have the same size (border effects). 
        N = 0
        for original_image, image_GT,  original_image_name in self._db_server.iter_training(code, first, last, N_squares=N_squares):
            print original_image_name
            if not self._nb_samples is None:
                N += self._nb_samples
            else:
                N += original_image.getWidth()*original_image.getHeight()
            if export:
                debug_dir = '/share/data40T_v2/debug_challenge2'
                if not os.path.isdir(debug_dir):
                    os.makedirs(debug_dir)
                sp.write(original_image, os.path.join(debug_dir, 'deconv_%s.png' % original_image_name))
                sp.write(image_GT, os.path.join(debug_dir, 'gt_%s.png' % original_image_name))

        # get the number of features
        complete_feature_list = []
        for feat in self._features_list: complete_feature_list.extend(feat.get_name())
        P = len(complete_feature_list)

        X_train = np.zeros((N, P))
        Y_train = np.zeros((N, 1))        

        i = 0

        for original_image, image_GT,  original_image_name in self._db_server.iter_training(code, first, last, N_squares=N_squares):
            
            X_image, dico= self.get_X_per_image(original_image, original_image_name)
            image_uc_lab = self.transform_image_uc(original_image)
            Y_image = self.get_Y_per_image(image_uc_lab, image_GT)
            print "Name: ", original_image_name
            print "Size: ", original_image.getSize()
            print "dim(X): ", X_image.shape

            if self._nb_samples is not None:
                X_image, Y_image = self.subsample(X_image, Y_image, code)

            print "Size: ", original_image.getSize()
            print "dim(X): ", X_image.shape
            #pdb.set_trace() 
            X_train[i:(i+X_image.shape[0]),:] = X_image[:,:]
            Y_train[i:(i+Y_image.shape[0]),:] = Y_image[:,:]
            i += X_image.shape[0]

            #X_train = uf.my_concatenation(X_train,  X_image)
            #Y_train = uf.my_concatenation(Y_train,  Y_image)
        return X_train, Y_train, dico

    def get_features_names_list(self):
        """
        Enables to get the list of all features computed.
        """
        f_list = []
        for feature in self._features_list:
            f_list += [elem for elem in feature.get_name()]
        return f_list
        

    def set_classifier(self, classifier):
        """
        Sets the classifier used by the class.
        The classifier must be a class instance which offers at least
        a fit and predict methods (as those of sklearn.ensemble, for example).
        """
        self.classifier = classifier


    def fit(self, code="train"):
        """
        Call the class classifier in order to compute a segmentation model.
        Args:
           code (optional) : gives the subbase on which the fit will be computed.
                             Classically, it is train.
        """
        debug_mess("Running fit")
        X_train, Y_train = self.get_X_Y_for_train(code)
        X_train = self.deal_with_missing_values_2(X_train)
        self.classifier.fit(X_train, np.ravel(Y_train))


    def predict(self, new_image,  new_image_filename):
        """
        This method enables to segment an image using the model learned by fit.
        """
        debug_mess("Prediction...")
        X_image = self.get_X_per_image(new_image, new_image_filename)
        X_image = self.deal_with_missing_values_2(X_image)
        Y_image = self.classifier.predict(X_image)

        ## visualization of predicted labels in the image:
        if 0:
            image_uc_lab = self.transform_image_uc(new_image)
            im_pred = self.visu_prediction_on_uc_image(image_uc_lab, np.array(Y_image))
            im_pred.save("/tmp/out.png")
        return Y_image

##--------------------------------------
    def get_S_per_image(self,  image,  name):
        """
        This function enables to compute a vector of size the number of pixels in the image, with every element equal to the name (e.g. of the image).
        To be used for cross-validation only.
        """
        S = np.chararray([image.getSize()[0]*image.getSize()[1], 1], itemsize = 100)
        S[:] = name
        S=np.array(S)
        return S

    def subsample_cross_validation(self, X, Y, S,  code):
        labels = np.unique(Y)
        nb_labels = len(labels)
        nb_samples_per_image_per_label = self._nb_samples / self._db_server.nb_im(code) / nb_labels
        pdb.set_trace()
        X_sample = None
        for label in labels:
            mask_index = [i for i in np.arange(len(Y)) if Y[i]==label]
            np.random.seed(42)
            mask_random = np.int32(np.floor(np.random.random(nb_samples_per_image_per_label)*len(mask_index)))
            X_tmp = X[mask_index][mask_random]
            Y_tmp = Y[mask_index][mask_random]
            S_tmp = S[mask_index][mask_random]
            if X_sample is None:
                X_sample = X_tmp
                Y_sample = Y_tmp
                S_sample = S_tmp
            else:
                if len(X_sample.shape)==1:
                    X_sample = X_sample[:, np.newaxis]
                if len(X_tmp.shape)==1:
                    X_tmp = X_tmp[:, np.newaxis]
                X_sample = np.vstack((X_sample, X_tmp))
                Y_sample = np.vstack((Y_sample, Y_tmp))
                S_sample = np.vstack((S_sample, S_tmp))
                
        return X_sample, Y_sample,  S_sample

    def get_X_Y_S_for_cross_validation(self, code):
        """
        
        """
        X_train = None
        Y_train = None
        S_train = None
        for original_image, image_GT,  original_image_name in self._db_server.iter_training(code):
            X_image = self.get_X_per_image(original_image, original_image_name)
            image_uc_lab = self.transform_image_uc(original_image)
            Y_image = self.get_Y_per_image(image_uc_lab, image_GT)
            print "Name: ", original_image_name
            new_name = original_image_name.split("_")[0]+"_"+original_image_name.split("_")[1]
            S_image = self.get_S_per_image(original_image, new_name)
            if self._nb_samples is not None:
                X_image, Y_image,  S_image = self.subsample_cross_validation(X_image, Y_image, S_image, code)
            X_train = uf.my_concatenation(X_train,  X_image)
            Y_train = uf.my_concatenation(Y_train,  Y_image)
            S_train = uf.my_concatenation(S_train,  S_image)

        return X_train, Y_train,  S_train

    def cross_validation(self, k,  code="train"):
        """
        Call the class classifier in order to compute a segmentation model.
        Args:
           code (optional) : gives the subbase on which the fit will be computed.
                             Classically, it is train.
            k (int): number of folds to split the training data
        """
        debug_mess("Running fit")
        X_train, Y_train,  S_train = self.get_X_Y_S_for_cross_validation(code)
        X_train = self.deal_with_missing_values_2(X_train)
        ## là il faut ajouter la boucle
        list_slides_names = list(np.unique(S_train))
        nber_of_slides = len(list_slides_names)
        nber_of_slides_per_fold = nber_of_slides / k
        performance = 0
        for num_fold in range(k):
            print "Dealing with fold number %i over %i" %(num_fold + 1,  k)
            list_test = [list_slides_names[j] for j in range(nber_of_slides_per_fold * num_fold,  nber_of_slides_per_fold * (num_fold + 1))]
            print list_test
            L = np.uint8(np.zeros(S_train.shape))
            for slide_name in list_test:
                L += np.uint8(S_train==slide_name)
            sel_test = np.where(L==1)[0]
            X_test_fold = X_train[sel_test, :]
            Y_test_fold = Y_train[sel_test, :]
            sel_train = np.where(L==0)[0]
            X_train_fold = X_train[sel_train, :]
            Y_train_fold = Y_train[sel_train, :]
            self.classifier.fit(X_train_fold, np.ravel(Y_train_fold))
            Y_pred_fold = self.classifier.predict(X_test_fold)
            ## comparaison entre y_pred_fold et y_test_fold
            #performance += np.sum(Y_pred_fold - Y_test_fold)
        return performance
##--------------------------------------
##--------------------------------------
class PixelClassification(LearnSegmentation):
    """
    Class used to learn segmentation models by pixel classification,
    and applying them.
    """

    def __init__(self, db_server, output_folder, features_list, nb_samples=None):
        """
        Parameters:
        db_server: segmentation data base server (see segm_db_access.py)
        output_folder (string): path to resulting segmentations/classifications.
        features_list (list): a list of instances of feature classes. The feature class must offer:
            * a __call__ method which takes as input a single SMIL image, and computes a matrix (see note in features_for_classification.py)
            * a name attribute which gives the official name for the feature.
        """
        LearnSegmentation.__init__(self, db_server, output_folder, features_list, nb_samples)


    def transform_image_uc(self, original_image):
        """Compute classification units (uc) image.

        This method enables to transform the original image into an image of same size
        filled with labelled classification units (uc),
        i.e. labelled pixels.

        Input:
        original_image: colour or gray level image.

        Output:
        imout(UINT32): image of the same size as the original_image,
        with every pixel labelled.
        """
        size = original_image.getSize()
        new_array = np.arange(1, size[0]*size[1]+1, dtype=np.uint32).reshape(size[0], size[1])
        imtmp = sp.Image(size[0], size[1])
        imout = sp.Image(imtmp, "UINT16")# pour le moment, normalement mettre uint32 ou faire échantillonnage qd uc = pixels
        imArr = imout.getNumArray()
        for i in range(size[0]):
            for j in range(size[1]):
                imArr[i,j] = new_array[i,j]

        return imout


    def get_X_per_image(self, original_image, original_image_name):
        """
        This method enables to compute the vector of features of all classification units
        and store them in a matrix X.

        Input:
        original_image (UINT8): color or gray image

        Output:
        X: matrix of size MxN where M is the number of pixels in the image
        and N the number of features.
        """
        folder_sauv_path = os.path.join( self._db_server.get_input_dir() ,  "sauvegarde" )
        image_sauv_path = os.path.join(folder_sauv_path,  "uc_pixel")
        X_image, dico = self.get_X_per_image_with_save_3(original_image,  original_image_name,  folder_sauv_path,  image_sauv_path)
        
        return X_image, dico


    def get_Y_per_image(self, image_uc_lab, image_GT):
        """Compute the vector of labels of the pixels in the image.

        Input:
        image_GT (UINT8): image of labelled regions of the scene
        (only UINT8 for now because we only deal with binary classification)

        Output:
        Y: vector of size(Mx1) where M is the number of pixels in the image.
        """
        a = np.ravel(image_GT.getNumArray())
        Y = a.reshape(a.shape[0],1)
        return Y


    def visu_prediction_on_uc_image(self, image_uc_lab,  Y, show_cont=False):
        """
        Enables to visualize the prediction results of classification for every classification unit (uc) in a given image.

        Input:
        image_uc_lab (UINT16): image of labelled uc. (e.g. pixels or superpixels).
        Y (ndarray): predicted (or already known) labels of uc.
        show_cont (optional): boolean we should be set to True when using supperpixels
           if we want to see their contours.

        Output:
        imout (UINT16) : image of uc with classification labels and visualization of contours between uc.
        """
        (size_x, size_y, _) = image_uc_lab.getSize()
        imout = sp.Image(image_uc_lab, "UINT8")
        imout.getNumArray()[:,:] = Y.reshape(size_x, size_y)
        if show_cont:
            print "Warning: PixelClassification.visu_prediction_on_uc_image() does not show contours"

        return imout


##--------------------------------------
##--------------------------------------
