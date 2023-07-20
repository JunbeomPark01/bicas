# check에서 검수가 되고 'e'키로 edit(임시 폴더 명)에 있는 폴더의 path 가지고
# 자동으로 labelimg을 켜 가지고 있는 폴더의 path를 바로 열고
# 저장 할 곳은 edit.path 와 같은 폴더에 하나 생성 하거나 검수완료한 폴더(받아야함)로 넣기

import subprocess
import os


def edit(_keyword):
#if __name__ == "__main__":
    keyword = _keyword
    labelimg_dir = os.getcwd() + "/labelImg"
    os.chdir(labelimg_dir)
    print(os.getcwd())
    subprocess.call(['python', 'labelImg.py', keyword])
