from ast import arg
from encodings import utf_8
from textwrap import indent
from tokenize import group
from xml.dom import minidom
import os
import time
from datetime import datetime

from os import listdir
import natsort
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import SubElement, ElementTree, dump
from labelingTemplateXml import indent
import pandas as pd
import numpy as np

import glob

dt_now = datetime.now()

#XML띄워쓰기 출력
def _pretty_print(current, parent=None, index=-1, depth=0):
    for i, node in enumerate(current):
        _pretty_print(node, current, i, depth + 1)
    if parent is not None:
        if index == 0:
            parent.text = '\n' + ('\t' * depth)
        else:
            parent[index - 1].tail = '\n' + ('\t' * depth)
        if index == len(parent) - 1:
            current.tail = '\n' + ('\t' * (depth - 1))

#class, x1, y1, x2, y2,  
# coding isCrowd,date
#auto increase //last +1 ==  index,
# each insert worker,  imgsize
#data path

#입력값 받아서 태그에 저장하기
worker = input("작업자 이름을 입력하세요 :")
imgSize = input("이미지 크기를 입력하세요 :")
# train_folder = input("트레인파일 경로를 입력하세요(파이썬 파일기준) :")
train_folder = './rifle/train/'
print(train_folder)
dir_list = os.listdir(train_folder)

#load label data
print(dir_list[1])
file_list = os.listdir(train_folder+"/"+ dir_list[1]+"/")
fpath =natsort.natsorted(file_list)
# print(file_list)
print(fpath)

root = ET.Element("root")

while True:
        for i in range(len(file_list)):
            datasets = ET.SubElement(root,"datasets")

            #txt in split at " " & print for array
            print("-------------------")
            file_name = os.path.splitext(fpath[i])
            fr = open(train_folder+ "/" + dir_list[1] + "/" + fpath[i], "r")
            # print(fr)
            fsplit=[x.split() for x in fr]
            
            # Create each xml 

            header = ET.SubElement(datasets, "header")
            imgInfo = ET.SubElement(datasets, "imgInfo")

            ET.SubElement(header, "nc").text = fsplit[0][0]
            ET.SubElement(header, "date").text = dt_now.strftime('%y%m%d %H%M')
            ET.SubElement(header, "worker").text = str(worker)

            ET.SubElement(imgInfo, "index").text = str(i+1)
            ET.SubElement(imgInfo, "imgSize").text = str(imgSize)


            label = ET.SubElement(imgInfo, "label")
            for j in range(len(fsplit)):
                #points realted
                points = ET.SubElement(label, "points")
                ET.SubElement(points, "class").text = fsplit[j][0]
                ET.SubElement(points, "x1").text = fsplit[j][1]
                ET.SubElement(points, "y1").text = fsplit[j][2]
                ET.SubElement(points, "x2").text = fsplit[j][3]
                ET.SubElement(points, "y2").text = fsplit[j][4]
            ET.SubElement(imgInfo, "isCrowd").text = str(len(fsplit))
        indent(root)
        break

tree = ET.ElementTree(root)

new_dir_path = dt_now.strftime('%y%m%d %H%M')
    
os.makedirs(new_dir_path, exist_ok=True)

if not os.path.exists(new_dir_path):
    os.makedirs(new_dir_path)


tempFile = "./"+ new_dir_path+"/" + dir_list[1] + ".xml"
#print(tempFile)
tree.write(tempFile, encoding="UTF-8", xml_declaration=True) # taking time about 4 seconds(29 xml files)=> 1285 : about 20 mins
