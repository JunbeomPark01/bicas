import os
import cv2

from .fun import readNshow, matching, input_key, tempfile_remove, Do_CopyNDel

def check_label(_keyword, pass_conf, amb_conf, fail_conf):
    #get dir file for paths
    dir_txt = open('./check_labeling/predefined/dir.txt',encoding='UTF8').read().split()
    
    DIR_PATH = dir_txt[9] + _keyword
    AMB_IMG_DIR = DIR_PATH + "/amb/images/"
    AMB_LABEL_DIR = DIR_PATH + "/amb/labels/"
    TEMP_DIR_PATH = dir_txt[11]
    image_array = os.listdir(path=AMB_IMG_DIR)
    label_array = os.listdir(path=AMB_LABEL_DIR)

    tempfile_remove()
    #file extension remove
    label_names = []
    for l in label_array:
        label_names.append(os.path.splitext(l))
    
    #iter 1 image name
    img_num = 0 
    while img_num != len(image_array):
        image = image_array[img_num]
        label = matching(image,label_names) #checking for label is exist
        if not label: #.txt file not found next img load
            continue
        #image read n show
        readNshow(AMB_IMG_DIR+image, AMB_LABEL_DIR+label, _keyword, pass_conf, amb_conf, fail_conf) 
        #input key
        key = input_key(image,label,TEMP_DIR_PATH)
        if key == 1: img_num +=1
        elif key == -1 and img_num > 0: img_num -=1

        cv2.destroyAllWindows()

    Do_CopyNDel(DIR_PATH,AMB_IMG_DIR,AMB_LABEL_DIR)
    return 1
