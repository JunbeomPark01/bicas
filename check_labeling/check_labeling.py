import os
import cv2

from .fun import readNshow, matching, input_key

def check_label(_keyword, pass_conf, amb_conf, fail_conf):
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

        readNshow(AMB_IMG_DIR+image, AMB_LABEL_DIR+label, _keyword, pass_conf, amb_conf, fail_conf) #image read n show

        while input_key(image,label,_keyword):
            continue

        cv2.destroyAllWindows()
    return 1

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