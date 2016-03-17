# -*- coding: cp1252 -*-
"""
Description: useful functions to create the tree view of folders for the leave-one-out procedure.
Author: MoSeS-learning project: Vaïa Machairas, Etienne Decencière, Thomas Walter.
Creation date: 2015-11-03
"""
import pdb
import os
import shutil as sh

##---------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------
def create_folder(path,  new_folder_name, tree = False):
    """
    This function enables to create a new folder with name "new_folder_name" if it does not already exist at the path "path",
    as well as two "daughter" folders": "train" and "test" if "tree" is True.
    Inputs:
    - path : string
    - new_folder_name : string
    - tree: bool
    """
    if os.path.isdir(os.path.join(path,  new_folder_name)) is not True:
        try:
            os.mkdir(os.path.join(path,  new_folder_name))
        except OSError:
            print "Could not create folder %s at path %s"%(new_folder_name,  path)
            raise 
    if tree is True:
        if os.path.isdir(os.path.join(path,  new_folder_name, "train")) is not True:
            try:
                os.mkdir(os.path.join(path,  new_folder_name, "train"))
            except OSError:
                print "Could not create folder %s at path %s"%("train",  os.path.join(path,  new_folder_name))
                raise 
        if os.path.isdir(os.path.join(path,  new_folder_name, "test")) is not True:
            try:
                os.mkdir(os.path.join(path,  new_folder_name, "test"))
            except OSError:
                print "Could not create folder %s at path %s"%("test",  os.path.join(path,  new_folder_name))
                raise 
##---------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------
def create_results_folder(path):
    """
    This function enables to create the folder "results" and its resulting tree of folders at the path "path".
    Input:
    - path: string
    """
    create_folder(path, "resultats",  True)
    create_folder(os.path.join(path, "resultats"), "TFPN",  True)
    create_folder(os.path.join(path, "resultats"), "labels",  True)
    #create_folder(os.path.join(path, "resultats"), "otsu",  True)
##---------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------  
def create_all_subsets_folders(path):
    """
    This function enables to create all images and GT folders for leave_one_out procedure.
    If there is N images in the database, there will be N subfolders corresponding to each split.
    The folder "path" should contain two non-empty folders "images" and "GT".
    Input:
    - path : string
    """
    dico_of_path = {}
    if os.path.isdir(os.path.join(path, "images",  "train")) is True:
        create_results_folder(path)
        dico_of_path["noLOO"] = path
    else:
        #pdb.set_trace()
        list_of_images = os.listdir(os.path.join(path, "images"))
        N = len(list_of_images) ##number of images in the database
        if N<15:
            create_folder(path,  "LOO")
            for i in range(N):
                ## create tree of folders of the split i:
                create_folder(os.path.join(path,  "LOO"), "split"+str(i))
                create_folder(os.path.join(path,  "LOO", "split"+str(i)), "images",  True)
                create_folder(os.path.join(path,  "LOO", "split"+str(i)), "GT", True)
                create_results_folder(os.path.join(path,  "LOO", "split"+str(i)))
                ## copy image i in the folder test:
                sh.copy2(os.path.join(path, "images",  list_of_images[i]), os.path.join(path,  "LOO", "split"+str(i), "images",  "test" ))
                sh.copy2(os.path.join(path, "GT",  list_of_images[i]), os.path.join(path,  "LOO", "split"+str(i), "GT",  "test" ))
                ## copy all other images in the folder train:
                for j in range(N):
                    if j!=i:
                        sh.copy2(os.path.join(path, "images",  list_of_images[j]), os.path.join(path,  "LOO", "split"+str(i), "images",  "train" ))
                        sh.copy2(os.path.join(path, "GT",  list_of_images[j]), os.path.join(path,  "LOO", "split"+str(i), "GT",  "train" ))
                dico_of_path["split"+str(i)] = os.path.join(path,  "LOO",  "split"+str(i))
        else:
            create_folder(path,  "split0")
            create_folder(os.path.join(path,  "split0"),  "images",  True)
            create_folder(os.path.join(path,  "split0"),  "GT",  True)
            create_results_folder(os.path.join(path, "split0"))
            for i in range(int(N/2)+1):
                sh.copy2(os.path.join(path,  "images",  list_of_images[2*i]),  os.path.join(path, "split0", "images",  "train"))
                sh.copy2(os.path.join(path,  "GT",  list_of_images[2*i]),  os.path.join(path, "split0", "GT",  "train"))
            for i in range(int(N/2)):
                sh.copy2(os.path.join(path,  "images",  list_of_images[2*i+1]),  os.path.join(path, "split0", "images",  "test"))
                sh.copy2(os.path.join(path,  "GT",  list_of_images[2*i+1]),  os.path.join(path, "split0", "GT",  "test"))
            dico_of_path["split0"] = os.path.join(path,  "split0")
    return dico_of_path
##---------------------------------------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------------------------------------  
