############ This files creates a setting file for 
# machine_learning.sh and machine_learning.py
# it has to be a text file seperated by ' ' and with 6 fields
# field0 is the line
# field1 is Normal or Tumor
# field2 is x
# field3 is y
# field4 is w
# field5 is h
# field6 is res

import os
from find_ROI import ROI
from optparse import OptionParser
import pdb

if __name__ ==  "__main__":

	parser = OptionParser()
	parser.add_option("-s", "--source", dest="folder_source",
	                  help="Where to find Tumor files", metavar="FILE")
	parser.add_option("-r", "--res", dest="resolution",
	                  help="resolution for the prediction", metavar="INT")
	parser.add_option("-m", "--margin", dest="margin",default=0.3,
	                  help="Margin for the overlayer", metavar="FLOAT (0,1)")
	parser.add_option("-o", "--output", dest="output",
	                  help="Output folder", metavar="FILE")
	(options, args) = parser.parse_args()

	image_sauvegarde = os.path.join(options.folder_source,options.output)
	if not os.path.isdir(image_sauvegarde):
		os.mkdir(image_sauvegarde)

	f = open("settings_for_pred_base.txt", "a")
	line = 0
	for prefixe in ["Test"]:#,"Normal","Tumor"]:

		if prefixe == "Tumor":
			n_range = 110
		elif prefixe == "Normal":
			n_range = 160
		else:
			n_range = 120

		for i in range(1,n_range+1):
			slide = prefixe + "_" +(3-len(str(i)))*'0' + str(i) +".tif"
			slide_name = os.path.join(options.folder_source, prefixe, slide)
			save_folder = os.path.join(image_sauvegarde, slide.split('.')[0])
			if not os.path.isdir(save_folder):
				os.mkdir(save_folder)
			ROI_pos=ROI(slide_name, ref_level = int(options.resolution), thresh = 220, 
                                 black_spots = 20, number_of_pixels_max = 1000000,
                                 method='grid_etienne',marge=float(options.margin) )

			for para in ROI_pos:	
				f.write("__" + str(line) + "__ "+prefixe+" "+str(i)+" "+str(para[0])+" "+str(para[1])+" "
						+ str(para[2]) + " " +str(para[3]) + " " + str(para[4]) +"\n")
				line += 1
	f.close()