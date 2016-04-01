import os, sys, time, re

import numpy as np
import skimage.io

import cPickle as pickle

from optparse import OptionParser

import pdb



class WholeSlideGenerator(object):
    def __init__(self, prob_map_folder, img_orig_folder, output_folder):
        self.prob_map_folder = prob_map_folder
        self.img_orig_folder
        self.output_folder = output_folder
        if not os.path.isdir(self.output_folder):
            print 'make %s' % self.output_folder
            os.makedirs(self.output_folder)

        # the level at which the probability maps have been stored.
        self.resolution_level = 2

        # the additional subsampling that has been applied
        self.subsampling_level = 16
        
    def from_highres_to_lowres(self, x, y, slide_factors):
        new_x = np.int(np.floor(np.float(x) / slide_factors[self.resolution_level] / self.subsampling_level))
        new_y = np.int(np.floor(np.float(y) / slide_factors[self.resolution_level] / self.subsampling_level))        
        return new_x, new_y
    
    def __call__(self, slidename):

        # read information from the original slide
        slide_filename = os.path.join(self.img_orig_folder, '%s.tif' % slidename)
        slide = openslide.open_slide(slide_filename)
        slide_dimensions = slide.level_dimensions
        slide_factors = slide.level_downsamples
        slide.close()
        
        # shape of the slide at resolution level 0
        w0 = slide_dimensions[0][0]
        h0 = slide_dimensions[0][1]

        w2 = slide_dimensions[2][0]
        h2 = slide_dimensions[2][1]
        
        w2_ss = w2 / self.subsampling_level
        h2_ss = h2 / self.subsampling_level
        
        # generate the output image
        img_out = np.zeros((h2_ss, w2_ss))
        counts = np.zeros((h2_ss, w2_ss), dtype=np.float)
        
        crop_folder = os.path.join(self.prob_map_folder, slidename)
        imagenames = self.listdir(crop_folder)
        for i, imagename in enumerate(imagenames):
            
            print 'processing %i / %i : %s' % (i, len(imagenames), imagename)
            
            # retrieve information from filename
            # probmap_Test_002_21016_114392_690_874
            info = os.path.splitext(feature_file)[0].split('_')
            x = int(info[3])
            y = int(info[4])
            width = int(info[5])
            height = int(info[6])

            new_x, new_y = self.from_highres_to_lowres(x,y,slide_factors)
                        
            # read image
            img = skimage.io.imread(os.path.join(crop_folder, imagename))
            img_h, img_w = img.shape 
            
            img_out[new_y:(new_y + img_h),new_x:(new_x + img_w)] += img 
            counts[new_y:(new_y + img_h),new_x:(new_x + img_w)] += 1

        counts[counts==0] = 1.0
        img_out = img_out / counts
        
        out_filename = os.path.join(self.output_folder, 'whole_probmap_%s.png' % slidename)
        print 'writing %s' % out_filename
        skimage.io.imsave(out_filename, img_out)
        return

if __name__ ==  "__main__":

    parser = OptionParser()

    parser.add_option("--orig_folder", dest="orig_folder",
                      help="folder with original slides")
    parser.add_option("--prob_map_folder", dest="prob_map_folder",
                      help="folder with probability maps")
    parser.add_option("--output_folder", dest="output_folder",
                      help="output folder")

    parser.add_option("--slide_name", dest="slide_name",
                      help="name of the slide (without extension)")
    parser.add_option("--slide_number", dest="slide_number",
                      help="number of the slide")
    
    (options, args) = parser.parse_args()

    if (options.orig_folder is None) or (options.output_folder is None) or (options.prob_map_folder is None):
        raise ValueError("probability map folder, original slide folder and output folder need to be specified.")

    WSG = WholeSlideGenerator(options.prob_map_folder, 
                              options.orig_folder, 
                              options.output_folder)
    
    if not options.slide_number is None:
        slide_number = int(options.slide_number)
        all_slides = filter(lambda x: os.path.splitext(x)[-1] == '.tif', 
                            os.listdir(options.orig_folder))
        slides_info = dict(zip(range(len(all_slides)), sorted(all_slides)))
        slide_name = slides_info[slide_number]
    elif not options.slide_name is None:
        slide_name = options.slide_name
    else:
        raise ValueError('either slide number or slide name must be given.')

    WSG(slide_name)
    
    print 'DONE !'
    

