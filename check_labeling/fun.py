import cv2
import os
import shutil
import random
from datetime import datetime 

dir_txt = open('./check_labeling/predefined/dir.txt',encoding='UTF8').read().split()

def tempfile_remove():
    TEMP_DIR_PATH = dir_txt[11]
    for f in ['save','del','edit']:
        if os.path.exists(TEMP_DIR_PATH+f'temp_{f}.txt'):
            os.remove(TEMP_DIR_PATH+f'temp_{f}.txt')
        
def readNshow(img_path, label_path, _keyword, pass_conf, amb_conf, fail_conf):
    class_name_array = open(f'./data/{_keyword}/classes.txt',encoding='UTF8').read().split()
    #read .txt, img file
    lines = open(label_path).readlines()
    frame = cv2.imread(img_path)
    #plot BBox
    for line in lines:
            val = line.split()
            conf = val[-1]
            class_name = class_name_array[int(val[0])] + ":" + str(round(float(conf),2))
            if float(conf) > pass_conf:
                color = [0,255,0]
            elif float(conf) > amb_conf:
                color = [0,0,255]
            else:
                color = [255,0,0]
            plot_one_box(restore_x(val[1:], frame.shape[:2]), frame, label=class_name, color=color)
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

def Do_CopyNDel(DIR_PATH,AMB_IMG_DIR,AMB_LABEL_DIR):

    for p in ['save','edit','del']:
        if p == 'save': subdir = 'pass'
        else: subdir = 'edit'

        if not os.path.exists(f'./check_labeling/temp/temp_{p}.txt'):
            continue

        if p == 'del': only_del=True
        else: only_del=False

        f = open(f'./check_labeling/temp/temp_{p}.txt','r')
        for line in f.readlines():
            image,label = line.split()
            copyNdel(AMB_IMG_DIR+image, DIR_PATH+f'/{subdir}/images/'+image, only_del)
            copyNdel(AMB_LABEL_DIR+label, DIR_PATH+f'/{subdir}/labels/'+label, only_del)
        f.close()


def matching(img_name, label_array):  #check for img_name in label_array
    split_img_name = os.path.splitext(img_name)[0]
    for label_name, tag in label_array:
        if split_img_name == label_name: return label_name+tag
    
    print(f"Wrong img name : {img_name}\nimg이름과 맞는 label를 찾을 수 없습니다.\t다음 img를 load합니다.")
    return 0


def overwrite_file(image,label,exist):
    v = f'{image} {label}\n'
    exist_file = open(f'./check_labeling/temp/temp_{exist}.txt','r')
    new_value = exist_file.read().replace(v,'')
    exist_file.close()
    overwrite = open(f'./check_labeling/temp/temp_{exist}.txt','w')
    overwrite.write(new_value)
    overwrite.close()
    print(new_value)

def input_key(image,label,TEMP_DIR_PATH):
    key = cv2.waitKeyEx(0) #input key
    for p in ['save','edit','del']:
        is_exist=None
        if not os.path.exists(f'./check_labeling/temp/temp_{p}.txt'):
            continue

        f = open(f'./check_labeling/temp/temp_{p}.txt')
        values = f.read().split()
        
        if image in values:
            is_exist = p
            f.close()
            break
        f.close()
    if key == ord('s'): #save img, label
        if is_exist:
            overwrite_file(image,label,is_exist)
        temp_save = open(TEMP_DIR_PATH+"temp_save.txt",'a')
        temp_save.write(f'{image} {label}\n')
        print(f"{datetime.now()} | go to labeled folder | file name : {label[:-4]}")
        temp_save.close()
        return 1
    
    elif key == ord('d'): #del img, label
        if is_exist:
            overwrite_file(image,label,is_exist)
        temp_del = open(TEMP_DIR_PATH+"temp_del.txt",'a')
        temp_del.write(f'{image} {label}\n')
        print(f"{datetime.now()} | delete | file name : {label[:-4]}")
        temp_del.close()
        return 1
    
    elif key == ord('e'): #edit img, label
        if is_exist:
            overwrite_file(image,label,is_exist)
        temp_edit = open(TEMP_DIR_PATH+"temp_edit.txt",'a')
        temp_edit.write(f'{image} {label}\n')
        print(f"{datetime.now()} | go to edit folder | file name : {label[:-4]}")    
        temp_edit.close()
        return 1
    
    elif key == 0x270000: return 1 #input right key | show next image
    
    elif key == 0x250000: return -1 #input left key | show prev image
        

    else:
        print("Wrong key input\tplease input keys [s,d,e,LEFT,RIGHT]\n\t- s: go to labeled folder\n\t- d: delete\n\t- e: go to edit folder\n\t- LEFT: show prev img\n\t- RIGHT: show next image")
        return 0


#center_x,center_y,width,height -> x1y1x2y2
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


def remove_sixth_column_from_txt_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r") as file:
                lines = file.readlines()

            modified_lines = []
            for line in lines:
                elements = line.strip().split(" ")
                if len(elements) == 6:  # 6개의 열(5개의 좌표값 + 1개의 confidence)이 있다면 여섯 번째 열을 삭제합니다.
                    modified_line = " ".join(elements[:5]) + "\n"
                    modified_lines.append(modified_line)
                elif len(elements) == 5:  # 5개의 열만 있는 경우 해당 라인을 그대로 유지합니다.
                    modified_lines.append(line)

            with open(filepath, "w") as file:
                file.writelines(modified_lines)