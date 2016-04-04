from optparse import OptionParser
import os
import glob
import openslide as op
import cPickle as pkl
import pdb
import skimage.io
if __name__ ==  "__main__":

	parser = OptionParser()
	
	parser.add_option("-s", "--source", dest="source",
	                  help="Where to find the tested slides", metavar="FILE")
	parser.add_option("--slide",default=".",dest="slide_folder",
					  help="where to find the slides folder",metavar="folder")
	parser.add_option("-o","--output",default=".",dest="output",
					  help="output folder",metavar="folder")
	parser.add_option("--resolution",default="5",dest="resolution",
					  help="resolution to extract ")

	(options, args) = parser.parse_args()
	print "creating maps for this folder : " + options.source
	print "the slide can be found here   : " + options.slide_folder
	print "the resulting images can be found in:" + options.output
	print "working at resolution         : " + options.resolution


	resolution = int(options.resolution)
	slide_folder = options.slide_folder
	output_folder = options.output
	files = glob.glob(os.path.join(options.source), "*")


	for fn in files:

		slide_name = fn.split('/')[-1]

		pref = slide_name.split('_')[0]

		path_file = os.path.join(slide_folder, pref, slide_name + ".tif")
		slide = op.open_slide(path_file)
		image = slide.read_region((0,0),resolution,slide.level_dimensions[resolution])
		skimage.io.save(image, os.path.join(output_folder, slide_name + "_resolution_"+str(resolution)+"_.png"))

		if pref == "Tumor":
			path_file_GT = os.path.join(slide_folder, pref + "_Mask", slide_name + "_Mask" + ".tif")
			slide_GT = op.open_slide(path_file_GT)
			image_GT = slide_GT.read_region((0,0),resolution,slide_GT.level_dimensions[resolution])
			skimage.io.save(image_GT, os.path.join(output_folder, slide_name + "_GT_resolution_"+str(resolution)+"_.png"))
			



