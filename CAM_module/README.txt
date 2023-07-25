yolov7 EigenCAM module
EigenCAM : Takes the first principle component of the 2D Activations
You can see why the custom yolov7 model recognized it as an object.

you have to install grad-cam
pip install grad-cam

References: https://github.com/jacobgil/pytorch-grad-cam


you can be used this project by using paser. 
here is an example

options:

  -h, --help            show this help message and exit

  --source SOURCE       img folder path

  -w WEIGHTS, --weights WEIGHTS
                        yolov7 .pt file path

  -y YOLOV7, --yolov7 YOLOV7
                        yolov7 folder path

  -cn CLASS_NUM, --class_num CLASS_NUM
                        class_num

  --save SAVE           save dir path

python ./yolov7_CAM.py --source ./images -w ./yolov7_test.pt -y ../yolov7 -cn 10 --save ./runs