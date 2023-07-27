import time
import os
import sys
import shutil
import argparse
from check_labeling.check_labeling import check_label
from check_labeling.fun import remove_sixth_column_from_txt_files as DelSixthCol
from labelImg import edit_labeling as EL

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='dataset name')
    parser.add_argument('-y7','--y7weights', nargs='+', type=str, help='y7 models.pt path(s)')
    parser.add_argument('-y5','--y5weights', nargs='+', type=str, help='y7 models.pt path(s)')
    parser.add_argument('-p','--pass-conf', type=float, help='pass confidence')
    parser.add_argument('-a','--amb-conf', type=float, help='amb confidence')
    parser.add_argument('-f','--fail-conf', type=float, help='fail confidence')

    opt = parser.parse_args()
    
    y7,y5, dataset, pass_conf, amb_conf, fail_conf = opt.y7weights, opt.y5weights, opt.dataset, opt.pass_conf, opt.amb_conf, opt.fail_conf
    print(opt)
    print('\nensemble weights : ',y7,y5,'\n')

    source = f"./data/{dataset}/images"
    if not os.path.isdir(source):
        print("No Data")
        raise

    level = ["pass", "amb", "fail", "edit"]
    for lev in level:
        os.makedirs(f"./data/{dataset}/{lev}/images", exist_ok = True)
        os.makedirs(f"./data/{dataset}/{lev}/labels", exist_ok = True)
    shutil.copy(f"./data/{dataset}/classes.txt", f"./data/{dataset}/edit/labels/classes.txt")
    

    #이전 yolo detect result 들 삭제
    erase_dir_path = './WBF_module/data/labels'
    erase_dir = os.listdir(erase_dir_path)
    if len(erase_dir) > 0:
        for e_d in erase_dir:
            shutil.rmtree(os.path.join(erase_dir_path,e_d))
    #yolov7 pt weights들 inference
    for w_7 in y7:
            
        os.chdir('./yolov7')
        com = f'python detect.py --weights .{w_7} --source .{source} --project ../WBF_module/data/labels --save-txt --save-conf --nosave'
        os.system(com)
    os.chdir('../')
    #yolov5 pt weights들 inference
    for w_5 in y5:
            
        os.chdir('./yolov5-master')
        com = f'python detect.py --weights .{w_5} --source .{source} --project ../WBF_module/data/labels --save-txt --save-conf --nosave'
        os.system(com)
    os.chdir('../')
    #Do WBF
    os.chdir('./WBF_module')
    os.system(f'python Do_WBF.py --source .{source} --dataset {dataset} --pass-conf {pass_conf} --amb-conf {amb_conf} --fail-conf {fail_conf}')
    os.chdir('../')
    
    # labels 삭제
    if os.path.exists(os.path.join('./data/',dataset,'labels')):
        shutil.rmtree(os.path.join('./data/',dataset,'labels'))

    done = check_label(dataset, pass_conf, amb_conf, fail_conf)
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

    DelSixthCol(f"../data/{dataset}/pass/labels")

