# -*- coding: cp1252 -*-

"""
python --output /share/data40T_v2/challengecam_results --ProbMapFolder /share/data40T_v2/challengecam_results/probmap/pred_data_set/whole_slide --slideFolder /share/data40T/pnaylor/Cam16/Test --CSVFolder /share/data40T_v2/challengecam_results/ResultatCSV

PREF_ID=Normal_001
PREF=Normal
python /home/naylor/Bureau/challengecam/cluster/fonctions_for_csv.py -f /home/naylor/Bureau/Temp/whole_slide/whole_probmap_$PREF_ID.png -r 2 --disk_size 5 --sigma 5 --plot 0 --subsampling 16 --slide_name /media/naylor/Peter-HD/data/$PREF/$PREF_ID.tif --output /home/naylor/Bureau/Temp/image_results_on_train/Results_$PREF_ID.csv

"""



"""
This file contains functions for generating the .csv file containing the results of prediction for the cahllenge CAMELYON16.
"""
import pdb
import numpy as np
import smilPython as sp
import useful_functions as uf
import skimage
import pandas as pd
from optparse import OptionParser
import matplotlib.pyplot as plt
from skimage.morphology import disk, opening
from scipy.ndimage.filters import gaussian_filter
import openslide as op
##------------------------------------------------------------------------------------------------------------------------------------------------
def preprocessing(image, disk_size=5, sigma=5):

    matrix_npy = np.transpose(image.getNumArray())
    selem = disk(disk_size)
    matrix_npy[:,:] = opening(matrix_npy, selem)
    matrix_npy[:,:] = gaussian_filter(matrix_npy, sigma)

    return image 


def get_max_proba_list_ref_imagette(imagette_smil, subsampling, res, slide_name):
    """
    Enables to compute the list of [conf, x, y] of the imagette smil (reference: imagette).

    Input:
    imagette_smil (UINT8): image smil which corresponds to an imagette of a slide.

    Output:
    list_max_proba (list): list of [conf, x, y] (reference: imagette)
    """
    list_max_proba = []
    im_maxima = sp.Image(imagette_smil)
    se = uf.set_structuring_element('V6',  1)
    sp.maxima(imagette_smil,  im_maxima,  se)
    im_numpy_arr = np.uint8(np.transpose(im_maxima.getNumArray()))
    proba_numpy_arr = np.uint8(np.transpose(imagette_smil.getNumArray()))
    sel = np.where(im_numpy_arr==255)

    slide = op.open_slide(slide_name)
    scale = slide.level_downsamples[res] * subsampling
    
    for i in range(len(sel[0])):
        list_max_proba += [[proba_numpy_arr[sel][i]/float(255), int(sel[1][i] * scale),  int(sel[0][i] * scale)]]
    return list_max_proba
##------------------------------------------------------------------------------------------------------------------------------------------------
def get_max_proba_list_ref_slide(imagette_smil,  coords):
    """
    Enables to compute the list of [conf, x, y] of the imagette smil (reference: slide).

    Input:
    imagette_smil (UINT8): image smil which corresponds to an imagette of a slide.
    coords (list): list of coordinates [x_slide,y_slide] of the left up corner of the imagette in the slide.

    Output:
    list_max_proba (list): list of [conf, x, y] (reference: slide)
    """
    list_max_proba_imagette = get_max_proba_list_ref_imagette(imagette_smil)
    for i in range(len(list_max_proba_imagette)):
        list_max_proba_imagette[i][1] += coords[0]
        list_max_proba_imagette[i][2] += coords[1]
    return list_max_proba_imagette
##------------------------------------------------------------------------------------------------------------------------------------------------

## Test:
if __name__ ==  "__main__":
    parser = OptionParser()
    
    parser.add_option("-f", "--file", dest="file",
                      help="Where to find probability map", metavar="FILE")
    #'/home/naylor/Bureau/Temp/whole_probmap_Test_002.png'
    parser.add_option("-o","--output_name",dest="output",
                      help="Name of the output csv file",metavar="folder")
    parser.add_option("-r","--resolution",dest="res",
                      help="resolution of the resolution")
    parser.add_option("--disk_size",dest="disk_size",
                      help="disk size for the opening")
    parser.add_option("--sigma",dest="sigma",
                      help="parameter for the gaussian smoothing")
    parser.add_option("--plot",dest="plot",default="0",
                      help="Should plot?")
    parser.add_option("--subsampling",dest="sampling",default="16",
                      help="Should plot?")
    parser.add_option("--slide_name",dest="slide_name",
                      help=" slide name, tiff file")

    (options, args) = parser.parse_args()

    print "file name:   |"+options.file
    print "resolution:  |"+options.res
    print "disk_size:   |"+options.disk_size
    print "sigma:       |"+options.sigma
    print "plotting:    |"+options.plot
    print "sampling step|"+options.sampling
    print "slide address|"+options.slide_name
    if not options.output is None:
        print "out csv name |"+options.output
    else:
        print "out csv name |"+"Not given"

    res = int(options.res)
    slide_name = options.slide_name
    samp = int(options.sampling)

    image = sp.Image(options.file,"UINT8")
    image = preprocessing(image, disk_size=int(options.disk_size), sigma=int(options.sigma))

    list_max_proba = get_max_proba_list_ref_imagette(image, samp, res, slide_name)

    data = pd.DataFrame(list_max_proba, columns=('Confidence','X coordinate','Y coordinate'))
    
    if not options.output is None:
        data.to_csv(options.output, index=False,header=False)

    if int(options.plot) == 0:
        npy_matrix = np.transpose(image.getNumArray())

        plt.imshow(npy_matrix,cmap = plt.cm.gray )

        slide = op.open_slide(slide_name)
        scale = slide.level_downsamples[res] * samp

        x = np.array(data['X coordinate'] / scale)
        y = np.array(data['Y coordinate'] / scale)

        colors = np.array(data['Confidence'])

        plt.scatter(x, y, c=colors)
        plt.savefig(options.file.split('.')[0] +"_targets" + '.png')



