import subprocess
import os


def edit(_keyword):
#if __name__ == "__main__":
    #_keyword = "thermal-human"
    labelimg_dir = os.getcwd() + "/labelImg"
    os.chdir(labelimg_dir)
    print(os.getcwd())
    subprocess.call(['python', 'labelImg.py', _keyword])
