import os, sys, time, re

import numpy as np
import skimage.io

import cPickle as pickle

from optparse import OptionParser

import pdb

class ImagePredictor(object):
    def __init__(self, classifier_name, feature_folder, output_folder, img_orig_folder=None):
        
        self.classifier_name = classifier_name
        self.classifier = self.read_classifier()
        self.feature_folder = feature_folder
        self.output_folder = output_folder
        self.img_orig_folder = img_orig_folder

        self.map_resolution_level = 4
        self.resolution_level = 2
        
    def read_classifier(self):
        fp = open(self.classifier_name, 'r')
        predictor = pickle.load(fp)
        fp.close()
        return predictor
    
    def read_data(self, data_filename):
        X = np.load(data_filename)
        return X 
    
    def __call__(self, slidename, write_crop=False, subsample=None, upper_limit=None, 
                 adapt_size=True):
        
        # features
        slide_folder = os.path.join(self.feature_folder, slidename)
        feature_files = filter(lambda x: os.path.splitext(x)[-1] == '.npy', 
                               os.listdir(slide_folder))

        crop_output_folder = os.path.join(self.output_folder, 'crops', slidename)
        if not os.path.isdir(crop_output_folder):
            os.makedirs(crop_output_folder)

        if upper_limit is None:
            upper_limit = len(feature_files)
 
        for feature_file in feature_files[:upper_limit]:
            start_time = time.time()
            print 'processing %s' % os.path.splitext(os.path.basename(feature_file))[0]
            
            if adapt_size:
                img = self.process_file_small(os.path.join(slide_folder, feature_file), subsample)
            else: 
                img = self.process_file(os.path.join(slide_folder, feature_file), subsample)
                
            img_filename = 'probmap_%s.png' % os.path.splitext(os.path.basename(feature_file))[0]
            skimage.io.imsave(os.path.join(crop_output_folder, img_filename), img)
            
            difftime = time.time() - start_time
            ms = np.int(np.floor((difftime - np.floor(difftime)  )   * 1000))
            difftime = np.int(np.floor(difftime))
            print '\t time elapsed: %02i:%02i:%03i' % ( (difftime / 60), (difftime%60), ms)
            
        # slide 
        # This would be /share/data40T/pnaylor/Cam16/Test/
        #slide_filename = os.path.join(self.img_orig_folder, '%s.tif' % slidename)
        #slide = openslide.open_slide(slide_filename)
        #slide_dimensions = slide.level_dimensions
        #slide_factors = slide.level_downsamples
        #slide.close()
        
        # shape of the slide at resolution level 0
        #w0 = slide_dimensions[0][0]
        #h0 = slide_dimensions[0][1]
        

        return
    
    def process_file(self, data_filename, subsample=None):

        info = os.path.splitext(os.path.basename(data_filename))[0].split('_')
        x = int(info[2])
        y = int(info[3])
        width = int(info[4])
        height = int(info[5])

        X = self.read_data(data_filename)
        N = X.shape[0]

        if subsample is None:
            probs = self.classifier.predict_proba(X)                    
        else:
            indices = np.arange(0, N, step=subsample)
            Xs = X[indices,:]
            Ys = self.classifier.predict_proba(Xs)
            small_indices = np.arange(len(Ys))
            
            # repeats the indices [0, 1, 2] -> [0, 0, 1, 1, 2, 2]
            # this corresponds to an upscaling of small_indices.             
            large_indices = np.repeat(small_indices, subsample)
            probs = Ys[large_indices,1]
            probs = probs[:N]

        img = probs.reshape((width, height))
        
        return img

    def process_file_small(self, data_filename, subsample=None):

        info = os.path.splitext(os.path.basename(data_filename))[0].split('_')
        x = int(info[2])
        y = int(info[3])
        width = int(info[4])
        height = int(info[5])

        X = self.read_data(data_filename)
        if subsample is None:
            probs = self.classifier.predict_proba(X)
            img = probs.reshape((width, height))                  
        else:
            col_indices = np.arange(0, width, step=subsample)
            row_indices = np.arange(0, height, step=subsample)
            new_width = len(col_indices)
            new_height = len(row_indices)
            A = np.zeros((width, height))
            B = np.zeros((width, height))
            A[col_indices,:] = 1
            B[:,row_indices] = 1
            C = A * B
            Cvec = np.ravel(C)
            indices = np.where(Cvec>0)[0]
            
            #N = X.shape[0]
            #indices = np.arange(0, N, step=subsample)
            Xs = X[indices,:]
            Ys = self.classifier.predict_proba(Xs)
            probs = Ys[:,1]
            img = probs.reshape((new_width, new_height))
            
        return img.T


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
    
        
class SlidePredictor(object):
    def __init__(self, classifier_name, feature_folder, output_folder, img_orig_folder=None):
        
        self.classifier_name = self.ip.classifier_name
        self.feature_folder = self.ip.feature_folder
        self.output_folder = self.ip.output_folder
        self.img_orig_folder = self.ip.img_orig_folder

        self.map_resolution_level = 4
        self.resolution_level = 2

        self.ip = ImagePredictor(self.classifier_name, self.feature_folder, 
                                 self.output_folder, self.img_orig_folder)


    def process_slide(self, slidename):
        # features
        slide_feature_folder = os.path.join(self.feature_folder, slidename)
        feature_files = filter(lambda x: os.path.splitext(x)[-1] == '.npy', 
                               os.listdir(slide_feature_folder))

        # for crops
        crop_output_folder = os.path.join(self.output_folder, 'crops', slidename)
        if not os.path.isdir(crop_output_folder):
            os.makedirs(crop_output_folder)
        self.ip.output_folder = crop_output_folder
        
        #for feature_file in feature_files:

        return
    
    def _DEPRECATED_process_slide(self, slidename, write_crops=True):
        
        # features
        slide_feature_folder = os.path.join(self.feature_folder, slidename)
        feature_files = filter(lambda x: os.path.splitext(x)[-1] == '.npy', 
                               os.listdir(slide_feature_folder))

        # output
#         whole_slide_output_folder = os.path.join(self.output_folder, 'whole_slide', slidename)
#         if not os.path.isdir(whole_slide_output_folder):
#             os.makedirs(whole_slide_output_folder)

        crop_output_folder = os.path.join(self.output_folder, 'crops', slidename)
        if not os.path.isdir(crop_output_folder):
            os.makedirs(crop_output_folder)
            
        # slide 
        # This would be /share/data40T/pnaylor/Cam16/Test/
        slide_filename = os.path.join(self.img_orig_folder, '%s.tif' % slidename)
        slide = openslide.open_slide(slide_filename)
        slide_dimensions = slide.level_dimensions
        slide_factors = slide.level_downsamples
        slide.close()
        
        # shape of the slide at resolution level 0
        w0 = slide_dimensions[0][0]
        h0 = slide_dimensions[0][1]
        
        res = []
        for feature_file in feature_files:
            info = os.path.splitext(feature_file)[0].split('_')
            x = info[1]
            y = info[2]
            width = info[3]
            height = info[4]
            
            full_filename = os.path.join(slide_folder, feature_file)
            img = self.process_file(full_filename, width, height)
            
            img_small = skimage.transform.downscale_local_mean(img, np.ones(len(img.shape)) * self.factor)
            res.append({
                        'img': img_small,
                        'x': x,
                        'y': y,
                        'x0': x * slide_factors[self.resolution_level],
                        'y0': y * slide_factors[self.resolution_level],                        
                        })
            
            if write_crops:
                img_name = os.path.join(crop_output_folder, 'prob_map_%s.png' + os.path.splitext(fulle_filename)[0])
                skimage.io.imsave(img_name, img)
        
        # reconstruct the entire image
        width = slide_dimensions[self.map_resolution_level][0]
        height= slide_dimensions[self.map_resolution_level][1]
        img_rec = np.zeros( shape = (width , height) ) 
        coverage = np.zeros( shape = (width , height) ) 

#        for crop in res:
#            # to do: loop over crops. 
#            print res['x']
            
        return

    
    def _deprecated_process_slide(self, slidename, write_crops=False):
        
        # features
        slide_folder = os.path.join(self.feature_folder, slidename)
        feature_files = filter(lambda x: os.path.splitext(x)[-1] == '.npy', 
                               os.listdir(slide_folder))

        # output
        whole_slide_output_folder = os.path.join(self.output_folder, 'whole_slide', slidename)
        if not os.path.isdir(whole_slide_output_folder):
            os.makedirs(whole_slide_output_folder)

        crop_output_folder = os.path.join(self.output_folder, 'crops', slidename)
        if not os.path.isdir(crop_output_folder):
            os.makedirs(crop_output_folder)
            
        # slide 
        # This would be /share/data40T/pnaylor/Cam16/Test/
        slide_filename = os.path.join(self.img_orig_folder, '%s.tif' % slidename)
        slide = openslide.open_slide(slide_filename)
        slide_dimensions = slide.level_dimensions
        slide_factors = slide.level_downsamples
        slide.close()
        
        # shape of the slide at resolution level 0
        w0 = slide_dimensions[0][0]
        h0 = slide_dimensions[0][1]
        
        res = []
        for feature_file in feature_files:
            info = os.path.splitext(feature_file)[0].split('_')
            x = info[1]
            y = info[2]
            width = info[3]
            height = info[4]
            
            full_filename = os.path.join(slide_folder, feature_file)
            img = self.process_file(full_filename, width, height)
            
            img_small = skimage.transform.downscale_local_mean(img, np.ones(len(img.shape)) * self.factor)
            res.append({
                        'img': img_small,
                        'x': x,
                        'y': y,
                        'x0': x * slide_factors[self.resolution_level],
                        'y0': y * slide_factors[self.resolution_level],                        
                        })
            
            if write_crops:
                img_name = os.path.join(crop_output_folder, 'prob_map_%s.png' + os.path.splitext(fulle_filename)[0])
                skimage.io.imsave(img_name, img)
        
        # reconstruct the entire image
        width = slide_dimensions[self.map_resolution_level][0]
        height= slide_dimensions[self.map_resolution_level][1]
        img_rec = np.zeros( shape = (width , height) ) 
        coverage = np.zeros( shape = (width , height) ) 

#        for crop in res:
#            # to do: loop over crops. 
#            print res['x']
            
        return

# def get_X_Y_from_0(slide,x_1,y_1,level):
#     ## Gives you the coordinates for the level 'level' image for a given couple of pixel from resolution 0
# 
#     size_x_0=slide.level_dimensions[level][0]
#     size_y_0=slide.level_dimensions[level][1]
#     size_x_1=float(slide.level_dimensions[0][0])
#     size_y_1=float(slide.level_dimensions[0][1])
#   
#     x_0=x_1*size_x_0/size_x_1
#     y_0=y_1*size_y_0/size_y_1
#   
#     return int(x_0),int(y_0)
# 
# def predict_WSI(slide,training_res,pred_WSI_res,classifier_vaia):
#     if slide is str:
#         slide = openslide.open_slide(slide)
# 
#     ROI_para = ROI(name,ref_level=training_res, disk_size=4, thresh=None, black_spots=None,
#                    number_of_pixels_max=1000000, verbose=False, marge=0.5, method='grid_etienne')
#     WSI_pred=np.zeros(shape=(slide.level_dimensions[pred_WSI_res][0],slide.level_dimensions[pred_WSI_res][1],2))
#     for para in ROI_para:
#         sub_image = slide.read_region((para[0],para[1]),para[4],(para[2],para[3]))
#         ### prediction  ###
# 
#         image_pred 
#         to_insert = change_res_np(image_pred)
#         x0, y0 = get_X_Y_from_0(slide,para[0],para[1],pred_WSI_res) 
#         size_x,size_y = get_size(slide, para[2], para[3], training_res, pred_WSI_res)
#         WSI_pred[x0:(x0+size_x),y0:(y0+size_y),0] += to_insert[0:size_x,0:size_y]  ###we maybe have to invert x and y
#         WSI_pred[x0:(x0+size_x),y0:(y0+size_y),0] += 1
# 
#     zeros = np.where(WSI_pred[:,:,1]==0)
#     WSI_pred[zeros,0] = WSI_pred[zeros,0] / WSI_pred[zeros,1]
# 
#     return(WSI_pred[:,:,0])
# 
# 
# class FullImagePredictor(object):
#     def __init__(self, image_predictor, 
#                  feature_folder):
#         self.image_predictor = image_predictor
#         self.feature_folder = feature_folder
#         
#         # typically for prediction
#         # feature_folder: /share/data40T_v2/challengecam_results/Pred_data_set
#         
#                 
    
if __name__ ==  "__main__":

    parser = OptionParser()

    parser.add_option("--classifier_name", dest="classifier_name",
                      help="classifier pickle file")
    parser.add_option("--feature_folder", dest="feature_folder",
                      help="feature folder")
    parser.add_option("--output_folder", dest="output_folder",
                      help="output folder")
    parser.add_option("--subsample_factor", dest="subsample_factor",
                      help="subsample factor")
    parser.add_option("--upper_limit", dest="upper_limit",
                      help="maximal number of crops to be analyzed (mainly for debugging)")
    parser.add_option("--slide_name", dest="slide_name",
                      help="name of the slide (without extension)")
    parser.add_option("--slide_number", dest="slide_number",
                      help="number of the slide")
    
    (options, args) = parser.parse_args()

    if (options.classifier_name is None) or (options.feature_folder is None) or (options.output_folder is None):
        raise ValueError("classifier name, feature folder and output folder need to be specified.")

    
    if not options.subsample_factor is None:
        subsample_factor = int(options.subsample_factor)
    else:
        subsample_factor = options.subsample_factor
        
    if not options.upper_limit is None:
        upper_limit = int(options.upper_limit)
    else:
        upper_limit = options.upper_limit
        
    IP = ImagePredictor(options.classifier_name,
                        options.feature_folder,
                        options.output_folder)
    
    if not options.slide_number is None:
        slide_number = int(options.slide_number)
        all_slides = filter(lambda x: os.path.isdir(os.path.join(options.feature_folder, x)), 
                        os.listdir(options.feature_folder))
        slides_info = dict(zip(range(len(all_slides)), sorted(all_slides)))
        slide_name = slides_info[slide_number]
    elif not options.slide_name is None:
        slide_name = options.slide_name
    else:
        raise ValueError('either slide number or slide name must be given.')

    IP(slide_name, subsample=subsample_factor,
       upper_limit=upper_limit)
    
    print 'DONE !'
    
    
        
