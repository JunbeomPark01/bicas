### yolov7 EigenCAM module
### EigenCAM : Takes the first principle component of the 2D Activations
### You can see why the custom yolov7 model recognized it as an object.
### References: https://github.com/jacobgil/pytorch-grad-cam

### !pip install grad-cam
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch    
import cv2
import numpy as np

import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image

import time
import argparse
import sys

COLORS = np.random.uniform(0, 255, size=(80, 3))

#model results -> (boxes, colors, names)
def parse_detections(results):
    detections = results.pandas().xyxy[0]
    detections = detections.to_dict()
    boxes, colors, names = [], [], []

    for i in range(len(detections["xmin"])):
        confidence = detections["confidence"][i]
        if confidence < 0.2:
            continue
        xmin = int(detections["xmin"][i])
        ymin = int(detections["ymin"][i])
        xmax = int(detections["xmax"][i])
        ymax = int(detections["ymax"][i])
        name = detections["name"][i]
        category = int(detections["class"][i])
        color = COLORS[category]

        boxes.append((xmin, ymin, xmax, ymax))
        colors.append(color)
        names.append(name)
    return boxes, colors, names

#draw boxes
def draw_detections(boxes, colors, names, img):
    for box, color, name in zip(boxes, colors, names):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(
            img,
            (xmin, ymin),
            (xmax, ymax),
            color, 
            2)

        cv2.putText(img, name, (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return img




#renormalize cam in bounding boxes
def renormalize_cam_in_bounding_boxes(boxes, colors, names, image_float_np, grayscale_cam):
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    image_with_bounding_boxes = draw_detections(boxes, colors, names, eigencam_image_renormalized)
    return image_with_bounding_boxes



#############################################################################################
parser = argparse.ArgumentParser(description='save Eigen Class Activation Map by yolov7')   
parser.add_argument('--source',default='./images', help='img folder path')
parser.add_argument('-w','--weights', default='./yolov7_test.pt' ,help='yolov7 .pt file path')
parser.add_argument('-y','--yolov7', help='yolov7 folder path')
parser.add_argument('-cn','--class_num', default=1,help='class_num')
parser.add_argument('--save', default='./runs',help='save dir path')
args = parser.parse_args()

source,weights,save,cn,yolov7 = args.source,args.weights,args.save,args.class_num,args.yolov7
#############################################################################################


import os
sys.path.append(yolov7)
from hubconf import custom

#model load
model = custom(path_or_model=weights)
model = torch.hub.load('WongKinYiu/yolov7','custom', weights,verbose=False)
model.eval()
model.cpu()
target_layers = [model.model.model[-2]]
targets = [ClassifierOutputTarget(cn)]

img_list = os.listdir(source)

#run
print("model load complete\n\n#### start CAM ####\n")
for name in img_list:
    start_time = time.time()
    img = cv2.imread(os.path.join(source,name))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    rgb_img = img.copy()
    img = np.float32(img) / 255
    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0)

    
    results = model([rgb_img])
    #model results -> (boxes, colors, names)
    boxes, colors, names = parse_detections(results)
    #draw boxes
    detections = draw_detections(boxes, colors, names, rgb_img.copy())
    #get CAM results
    cam = EigenCAM(model, target_layers,use_cuda=False)
    grayscale_cam = cam(tensor,targets,eigen_smooth=True)[0, :, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    
    #renormalize cam in bounding boxes
    renormalized_cam_image = renormalize_cam_in_bounding_boxes(boxes, colors, names, img, grayscale_cam)
    save_img = np.hstack((detections, cam_image, renormalized_cam_image))

    #img save
    cv2.imwrite(os.path.join(save,name),cv2.cvtColor(save_img,cv2.COLOR_BGR2RGB))
    print(f"save in {os.path.join(save,name)} ({round(time.time()-start_time,3)}s)")

print("\n#### DONE ####")

sys.path.remove(yolov7)
