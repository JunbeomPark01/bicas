# yolov7 EigenCAM module

  

EigenCAM : Takes the first principle component of the 2D Activations
You can see why the custom yolov7 model recognized it as an object.

you have to install grad-cam

``` shell
pip  install  grad-cam
```

  

you can be used this project by using option.  here is an example

  
### options

|  | |  |
|--|--|--|
| --source		 	| SOURCE 	| image folder path 	|
| -w, --weights	 	| WEIGHTS 	| yolov7 .pt file path 	|
| -cn , --class_num | CLASS_NUM	| classes_num 			|
| --save 			| SAVE		| save dir path 		|

### example
``` shell
cd CAM_module
python  ./yolov7_CAM.py  --source  ./images  -w  ./yolov7_test.pt  -y  ../yolov7  -cn  10  --save  ./runs
```

References: pytorch-grad-cam (https://github.com/jacobgil/pytorch-grad-cam)