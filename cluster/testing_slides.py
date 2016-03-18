
import pdb
from optparse import OptionParser
import os
from find_ROI import ROI, GetImage
import openslide

def checking_slide(slide_name):
	try:
		slide = openslide.open_slide(slide_name)
		if 'Tumor' in slide_name:
			slide_name_list = slide_name.split('/')
			mask = "_Mask"
			slide_name_list[0]='/'
			slide_name_list[-2] +=mask
			slide_name_list[-1] = slide_name_list[-1].split('.')[0]+mask+'.tif' 
			
			cm = os.path.join(*slide_name_list)
			list_ROI = ROI(slide_name, ref_level = 0, disk_size = 4, thresh = 220, 
	        	            	     	black_spots = 20, number_of_pixels_max = 1000000,
	            	            	   	method = 'SP_ROI', mask_address = cm,
	                	            	N_squares = 4, verbose = False )
		else: 
			list_ROI = ROI(slide_name, ref_level = 0, disk_size = 4, thresh = 220, 
	        		                    black_spots = 20, number_of_pixels_max = 1000000,
	            		               	method = 'SP_ROI', mask_address = None,
	                		            N_squares = 4, verbose = False )
		for para in list_ROI:
			sample = GetImage(slide,para)
	except:
		return False

	return True 


def file_generator(folder_source):

	Tumor_folder = os.path.join(folder_source, "Tumor")
	Normal_folder = os.path.join(folder_source, "Normal")
	Test_folder = os.path.join(folder_source, "Test")

	for fold in [Tumor_folder,Normal_folder,Test_folder]:
		for im_file in os.listdir(fold):
			yield os.path.join(fold, im_file)


def writing_outputs(name_generator,output_folder):
	f = open( os.path.join(output_folder,'output_checking.txt') , 'a')
	for slide_name in name_generator:
		if not checking_slide(slide_name):
			f.write(slide_name.split('/')[-1] + " is not ok \n")
#		else:
#			f.write(slide_name.split('/')[-1] + " is ok \n")
	f.close()


if __name__ == "__main__":


	parser = OptionParser()
	parser.add_option("-s", "--source", dest="folder_source",
	                  help="Where to find Tumor files", metavar="FILE")
	parser.add_option("-o", "--output", dest="output_folder",
	                  help="Where to find output files", metavar="FILE")

	(options, args) = parser.parse_args()

	name_generator = file_generator(options.folder_source)
	writing_outputs(name_generator, options.output_folder)
