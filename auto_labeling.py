import os
import sys
import shutil
from yolov7 import auto_detect as Y7
#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
#from yolov5 import auto_detect as Y5
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from check_labeling import check_labeling as CL
from labelImg import edit_labeling as EL

if __name__ == "__main__":
    
    dataset = input("Input dataset : ")
    source = f"./data/{dataset}/images"

    if os.path.isdir(source):
        level = ["pass", "amb", "fail", "edit"]
        for lev in level:
            os.makedirs(f"./data/{dataset}/{lev}/images", exist_ok = True)
            os.makedirs(f"./data/{dataset}/{lev}/labels", exist_ok = True)
        shutil.copy(f"./data/{dataset}/classes.txt", f"./data/{dataset}/edit/labels/classes.txt")
        weightsY7 = f"./data/{dataset}/Y7_{dataset}.pt"
        name = dataset
        # 이 값 위로는 pass 시키겠다. ex) 0.8 float 입력.
        pass_conf = float(input("Pass filter : "))
        # 이 값 위로는 ambiguous로 검수 시키겠다.
        amb_conf = float(input("Ambiguous filter : "))
        fail_conf = float(input("Fail filter : "))

        conf = Y7.detect(source, weightsY7, name, dataset, pass_conf, amb_conf, fail_conf)
        os.rmdir(f'./data/{dataset}/labels')
        done = CL.check_label(dataset, pass_conf, amb_conf, fail_conf)
        if done:
            EL.edit(dataset)

        # 끝나면 edit folder에 있는것들이 pass로 가기.
        edit_images_folder_path = f'../data/{dataset}/edit/images/'
        edit_images_folder = os.listdir(edit_images_folder_path)
        for edit_images_file in edit_images_folder:
            shutil.move(edit_images_folder_path + edit_images_file, f'../data/{dataset}/pass/images/')

        edit_labels_folder_path = f'../data/{dataset}/edit/labels/'
        edit_labels_folder = os.listdir(edit_labels_folder_path)
        for edit_labels_file in edit_labels_folder:
            shutil.move(edit_labels_folder_path + edit_labels_file, f'../data/{dataset}/pass/labels/')

        shutil.rmtree(f'../data/{dataset}/amb')
        shutil.rmtree(f'../data/{dataset}/edit')
        
        # test folder에 남은 것들은 삭제시키기.
        # fail이 사실상 다음 test 셋을 위한 것.

    else:
        print("No Data")

    