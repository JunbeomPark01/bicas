1. auto-labeling.py 실행 (0720 수정)

2. !!! Data 폴더는 따로 넣어줘야 합니다 !!!
  2-1 Data 폴더에 새로 추가할 때 
       (ex. thermal-human 이 아니라 pose 일때)
       .pt file 도 pose로 동일하게 해주셔야 합니다!

3. delete는 자동으로 삭제되지는 않고,
   test폴더에 그대로 남아있습니다.

4. Data 내 이미지 폴더 이름 test ---> images 로 수정
   처음 보는 사람들이 들으면 헷갈릴까봐 수정하였습니다.

############# TREE #################

    ROOT
     ├─auto_labeling.py (EXECUTE HERE)
     ├─yolov7
     ├─yolov5 (COMING SOON)
     ├─labelImg
     ├─check_labeling
     ├─data
     │   └─{dataset_name}
     │         ├─test      
     │         │  ├─images
     │         │  └─labels
     │         │      ├─divided_data1
     │         │      │     ├─yolov7 : model[1]
     │         │      │     └─yolov5 : model[2]
     │         │      └─divided_data2
     │         │            ├─yolov7 : model[3]
     │         │            └─yolov5 : model[4]
     │	       ├─pass
     │         │  ├─images
     │         │  └─labels
     │	       ├─amb          (Check Labeling Folder)
     │         │  ├─images
     │         │  └─labels
     │	       ├─fail
     │         │  ├─images
     │         │  └─labels
     │	       └─edit         (Edit with LabelImg Folder)
     │            ├─images
     │            └─labels
     │         
     └─README

####################################