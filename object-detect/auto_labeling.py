import auto_detect as Detect
import os
import sys
import shutil
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from check_labeling import check_labeling as CL
from labelImg import edit_labeling as EL

if __name__ == "__main__":

    dataset = input("Input dataset : ")
    source = f"./data/{dataset}/test"

    if os.path.isdir(source):
        level = ["pass", "amb", "fail", "edit"]
        for lev in level:
            os.makedirs(f"./data/{dataset}/{lev}/images", exist_ok = True)
            os.makedirs(f"./data/{dataset}/{lev}/labels", exist_ok = True)
        shutil.copy(f"./data/{dataset}/classes.txt", f"./data/{dataset}/edit/labels/classes.txt")
        weights = f"./data/{dataset}/{dataset}.pt"
        name = dataset
        # 이 값 위로는 pass 시키겠다. ex) 0.8 float 입력.
        pass_conf = float(input("Pass filter : "))
        # 이 값 위로는 ambiguous로 검수 시키겠다.
        amb_conf = float(input("Ambiguous filter : "))
        fail_conf = float(input("Fail filter : "))

        conf = Detect.detect(source, weights, name, dataset, pass_conf, amb_conf, fail_conf)
        done = CL.check_label(dataset)
        if done:
            EL.edit(dataset)
            
        # labelImg 가 끝나면 xml file로 변환시키기
        # 아니면 labelImg 에서 바로 저장시켜도 되고ㅇㅇ
    else:
        print("No Data")

    