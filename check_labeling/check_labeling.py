import os
import cv2

from .fun import readNshow, matching, input_key

def check_label(_keyword):
    #사실 없어도됨.
    dir_txt = open('./check_labeling/predefined/dir.txt',encoding='UTF8').read().split()
    #AMB_IMG_DIR = dir_txt[5]
    #AMB_LABEL_DIR = dir_txt[7]
    DIR_PATH = dir_txt[9] + _keyword
    AMB_IMG_DIR = DIR_PATH + "/amb/images/"
    AMB_LABEL_DIR = DIR_PATH + "/amb/labels/"
    image_array = os.listdir(path=AMB_IMG_DIR)
    label_array = os.listdir(path=AMB_LABEL_DIR)
    #print(label_array)
    label_names = []
    for l in label_array:
        label_names.append(os.path.splitext(l))

    for image in image_array:
        label = matching(image,label_names)
        if not label: #.txt file not found 시 다음 img load
            continue

        readNshow(AMB_IMG_DIR+image, AMB_LABEL_DIR+label, _keyword) #image read n show

        while input_key(image,label,_keyword):
            continue

        cv2.destroyAllWindows()
    return 1
