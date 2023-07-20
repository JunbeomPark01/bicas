import numpy as np
import cv2
import random
import os
from ensemble_boxes import weighted_boxes_fusion
from datetime import datetime
import shutil
import time
# import warnings
# warnings.filterwarnings(action='ignore')

#center_x,center_y,width,height -> x1y1x2y2
def restore_for_WBF(x):
    center_x,center_y = float(x[0]) ,float(x[1]) 
    width, height = float(x[2]), float(x[3])
    
    return[center_x - width/2, center_y - height/2, center_x + width/2, center_y + height/2]

#iou compute
def compute_iou(pred_box, gt_box):
  x1 = np.maximum(pred_box[0],gt_box[0])
  y1 = np.maximum(pred_box[1],gt_box[1])
  x2 = np.minimum(pred_box[2],gt_box[2])
  y2 = np.minimum(pred_box[3],gt_box[3])

  intersection = np.maximum(x2-x1,0) * np.maximum(y2-y1,0)
  pred_box_area = (pred_box[2]-pred_box[0]) * (pred_box[3] - pred_box[1])
  gt_box_area = (gt_box[2]-gt_box[0]) * (gt_box[3] - gt_box[1])
  
  union = pred_box_area + gt_box_area - intersection
  iou = intersection/union
  return iou


def make_dir(save_base_dir_path):
#save폴더 생성
    save_path = save_base_dir_path+"result_"+datetime.now().strftime("%Y_%m_%d_%H%M_%S")
    os.mkdir(save_path)
    shutil.copy('./predefined/classes.txt',save_path)
    return save_path

def save_result(boxes,scores,labels,save_name):
    #iou 높은 box들 삭제 
    #사실 상 nms랑 똑같아서 open lib 사용해도 무관
    result_boxes = []
    result_scores = []
    result_labels = []

    for i in range(len(boxes)):
        
        iou_cnt = 0
        for r in result_boxes:
            if compute_iou(r,boxes[i])>0.7:
                iou_cnt+=1
            
        if not iou_cnt:
            result_boxes.append(boxes[i])
            result_scores.append(scores[i])
            result_labels.append(labels[i])

        else:
                for j in range(len(result_boxes)):
                    if compute_iou(result_boxes[j],boxes[i])>0.7 and scores[i]>result_scores[j]:
                        result_boxes[j] = boxes[i]
                        result_scores[j] = scores[i]
                        result_labels[j] = labels[i]

    
    #output txt file 작성
    input_txt = []
    
    for s_i in range(len(result_boxes)):
        x1,y1,x2,y2 = result_boxes[s_i][0],result_boxes[s_i][1],result_boxes[s_i][2],result_boxes[s_i][3]
        width, height= x2-x1, y2-y1
        center_x, center_y = x1+width/2, y1+height/2
        input_txt.append(f'{int(result_labels[s_i])} {center_x} {center_y} {width} {height} {result_scores[s_i]}\n')
    with open(save_name,"w") as file:
        file.writelines(input_txt)


# main func
# param으로 image가 있는 dir path 넣어줘야함
def make_WBF_file(image_dir_path):
    # print start
    print("-"*50)
    print(f"make_WBF_file START")
    start_time = time.time()
    #기본 path들 지정
    predefined = open('./predefined/dir.txt').read().split()  #./predefined/dir.txt에서 설정 변경 가능
    
    ############# TREE #################
    # BASE
    # ├─main.py
    # ├─Do_WBF.py
    # ├─data
    # │  └─labels
    # │     ├─yolov3_detect
    # │     │  └─labels
    # │     │     ├─0.txt
    # │     │     └─1.txt
    # │     ├─yolov5_detect
    # │     │  └─labels
    # │     │     ├─0.txt
    # │     │     └─1.txt
    # │     └─yolov7_detect
    # │        └─labels
    # │           ├─0.txt
    # │           └─1.txt
    # ├─runs
    # │  └─result_2023_07_19_1506_01
    # │     ├─result_0.txt
    # │     └─result_1.txt
    # ├─predefined
    # │  ├─classes.txt
    # │  └─dir.txt
    # └─test_img (테스트를 위함임 실제로는 사용하지 않음)
    #    └─test01.jpg
    ####################################

    #기본 path들 지정
    predefined = open('./predefined/dir.txt').read().split()  #./predefined/dir.txt에서 설정 변경 가능
    label_dir_path = predefined[1]  # 여러 모델의 추론 결과가 모여있는 dir path
    save_base_dir_path = predefined[3]
    txt_path = '{}{}/labels/{}.txt'
    make_dir_path = make_dir(save_base_dir_path)
    
    
    ####### WBF param 정의 #########

    weights = np.zeros(3)
    for w_i in range(len(weights)):
        weights[w_i] = 1
    iou_thr = 0.5
    skip_box_thr = 0.0001

    ###############################


    img_list = os.listdir(image_dir_path)
    label_files = os.listdir(label_dir_path) #./data/labels | label_files = [yolov3_detect,yolov3_detect,yolov3_detect]
    
    #이미지 하나 선택
    for img_name in img_list:
        #WBF 를 위한 list 생성
        boxes_list = []
        scores_list = []
        labels_list = []
        img_name_split = os.path.splitext(img_name)
        save_name = make_dir_path+'/'+img_name_split[0]+'.txt'
        #존재 확인
        is_exist = []
        for model_name in label_files:  #각 모델별 추론한 txt file 존재 여부 확인
            is_exist.append(txt_path.format(label_dir_path,model_name,img_name_split[0]))
        if not any(is_exist):  #모든 모델이 추론 실패시 pass
            continue
        
        #txt 읽어와서 append
        label_file = None
        for model_name in label_files:
            # 모델별로 묶어서 append하기 위한 temp list
            temp_box = []        
            temp_score = []        
            temp_label = []        
            
            if os.path.exists(txt_path.format(label_dir_path,model_name,img_name_split[0])):
                lines = open(txt_path.format(label_dir_path,model_name,img_name_split[0])).readlines()

                for line in lines:
                    val = line.split()
                    temp_label.append(int(val[0]))
                    temp_score.append(float(val[-1]))
                    temp_box.append(restore_for_WBF(val[1:-1]))
            
                boxes_list.append(temp_box)
                scores_list.append(temp_score)
                labels_list.append(temp_label)

            else:
                boxes_list.append([])
                scores_list.append([])
                labels_list.append([])

        #WBF compute
        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)


        #save result.txt in save_name
        save_result(boxes,scores,labels,save_name)

    #print finish
    print(f"make_WBF_file successful FINISH\nsave in {make_dir_path} | total save txt file : {len(os.listdir(make_dir_path))-1}/{len(img_list)}\nelapsed time : {round(time.time() - start_time,2)}s")
    print("-"*50)