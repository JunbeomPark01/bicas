import cv2
import os
import shutil
import random
from datetime import datetime 

dir_txt = open('./check_labeling/predefined/dir.txt',encoding='UTF8').read().split()

def readNshow(img_path, label_path, _keyword):
    class_name_array = open(f'./data/{_keyword}/classes.txt',encoding='UTF8').read().split()
    #read .txt, img file
    lines = open(label_path).readlines()
    frame = cv2.imread(img_path)
    #plot BBox
    for line in lines:
            val = line.split()
            class_name = class_name_array[int(val[0])] + ":" +val[-1]
            plot_one_box(restore_x(val[1:], frame.shape[:2]), frame, label=class_name)
    #make window
    name = os.path.splitext(img_path)[0]
    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, width=640, height=640)
    cv2.moveWindow(name, x=100, y=100)
    cv2.imshow(name,frame)


def copyNdel(src, dst,only_del=False):
    if not only_del:
        shutil.copy(src,dst)

    if os.path.exists(src):
        os.remove(src)


def matching(img_name, label_array):  #img name과 맞는 
    split_img_name = os.path.splitext(img_name)[0]
    for label_name, tag in label_array:
        print(label_name)
        print(img_name)
        if split_img_name == label_name: return label_name+tag
    
    print(f"Wrong img name : {img_name}\nimg이름과 맞는 label를 찾을 수 없습니다.\t다음 img를 load합니다.")
    return 0


def input_key(image,label,_keyword):
    DIR_PATH = dir_txt[9] + _keyword
    AMB_IMG_DIR = DIR_PATH + "/amb/images/"
    AMB_LABEL_DIR = DIR_PATH + "/amb/labels/"
    LABELED_DIR = DIR_PATH + "/pass/" #pass dir
    EDIT_DIR = DIR_PATH + "/edit/"
    key = cv2.waitKey(0) #input key
    
    if key == ord('s'): #save img, label
        
        copyNdel(AMB_IMG_DIR+image, LABELED_DIR+"images/"+image)
        copyNdel(AMB_LABEL_DIR+label,LABELED_DIR+"labels/"+label)
        print(f"{datetime.now()} | go to labeled folder | file name : {label[:-4]}")
        return False
    
    elif key == ord('d'): #del img, label
        copyNdel(AMB_IMG_DIR+image, LABELED_DIR+'images',only_del=True)
        copyNdel(AMB_LABEL_DIR+label,LABELED_DIR+'labels',only_del=True)
        print(f"{datetime.now()} | delete | file name : {label[:-4]}")
        return False
    
    elif key == ord('e'): #edit img, label
        copyNdel(AMB_IMG_DIR+image, EDIT_DIR+'images/'+image)
        copyNdel(AMB_LABEL_DIR+label,EDIT_DIR+'labels/'+label)
        print(f"{datetime.now()} | go to edit folder | file name : {label[:-4]}")
        return False
    
    else:
        print("Wrong key input\tplease input keys [s,d,e]\n\t- s: go to labeled folder\n\t- d: delete\n\t- e: go to edit folder")
        return True


def restore_x(x,img_shape):
    y_size,x_size = img_shape
    center_x,center_y = float(x[0]) * x_size ,float(x[1]) * y_size 
    width, height = float(x[2]) * x_size, float(x[3]) * y_size 
    
    return[center_x - width/2, center_y - height/2, center_x + width/2, center_y + height/2]

def plot_one_box(x, img, color=None, label=None, line_thickness=2):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 6, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

