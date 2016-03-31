import os, sys, time, re
from optparse import OptionParser
import pdb

def read_info_jobs(filename):
    print 'reading %s' % filename
    
    # __0__ Test 1 60136 5096 294 326 2
    fp = open(filename, 'r')
    temp = fp.readlines()
    fp.close()
    
    res = {}
    keys = ['job_id', 'type', 'slide_number', 'x', 'y', 'width', 'height', 'resolution']
    for line in temp:
        entries = [x.strip('\n') for x in line.split(' ')]
        temp_info = dict(zip(keys[2:], [int(x) for x in entries[2:]]))
        temp_info['job_id'] = entries[0]
	temp_info['type'] = entries[1]
        if not temp_info['slide_number'] in res:
            res[temp_info['slide_number']] = []
        filename = '%s_%03i_%i_%i_%i_%i' % (temp_info['type'], temp_info['slide_number'],
                                            temp_info['x'], temp_info['y'], 
                                            temp_info['width'], temp_info['height'])
            
        # Test_001_39216_54032_948_1076.pickle        
        res[temp_info['slide_number']].append((filename, temp_info['job_id']))
        
        #res[temp_info['job_id']] = dict(zip(keys[1:], temp.split('_')[1:] ))
        
    return res

def check_integrity(in_folder, res, export_filename=None):
    slide_folders = filter(lambda x: os.path.isdir(os.path.join(in_folder, x)), os.listdir(in_folder))
    keys = ['type', 'slide_number', 'x', 'y', 'width', 'height']
    found = {}
    for slide_folder in slide_folders:

        images = filter(lambda x: os.path.splitext(x)[-1] == '.npy', 
                        os.listdir(os.path.join(in_folder, slide_folder)))
        slide_number = int(slide_folder.split('_')[-1])
	#pdb.set_trace()
        for image_name in images:
            if not slide_number in found:
                found[slide_number] = []
            
            found[slide_number].append(os.path.splitext(image_name)[0])

    #pdb.set_trace()
    not_processed = {}
    for sn in res.keys():
        image_list = res[sn]
        not_processed[sn] = []
        for image, job_id in image_list:
            if not image in found[sn]:
                not_processed[sn].append(image)
    
    if not export_filename is None:
	i = 1
	fp = open(export_filename, 'w')
        for sn in not_processed.keys():
            for filename in not_processed[sn]:
                 entries = filename.split('_')
		 #Test_001_39216_54032_948_1076
		 #__0__ Test 1 60136 5096 294 326 2
		 dataset = entries[0]
		 x = int(entries[2])
		 y = int(entries[3])
		 width = int(entries[4])
		 height = int(entries[5])
		 tempStr = '__%i__ %s %i %i %i %i %i 2' % (i, dataset, sn, x, y, width, height)
		 fp.write(tempStr + '\n')
		 i += 1 
        fp.close()   
            
    return not_processed


if __name__ ==  "__main__":

    parser = OptionParser()

    parser.add_option("-i", "--input_folder", dest="input_folder",
                      help="input folder")
    parser.add_option("-f", "--filename", dest="filename",
                      help="Filename with info on the submitted jobs")
    parser.add_option("-e", "--export_filename", dest="export_filename",
                      help="Filename for export")

    # /share/data40T_v2/challengecam_results/Pred_data_set
    
    (options, args) = parser.parse_args()
    res = read_info_jobs(options.filename)

    #pdb.set_trace()    
    not_processed = check_integrity(options.input_folder, res, options.export_filename)
    for sn in not_processed:
        print '%s\t%s' % (sn, not_processed[sn])
	print
        
    print 'DONE!'
    
    
