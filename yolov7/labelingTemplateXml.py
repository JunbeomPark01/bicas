# templateXml.py :  dump
from xml.etree.ElementTree import Element, SubElement, ElementTree, dump

root = Element("datasets")

header = SubElement(root, "header")
imgInfo = SubElement(root, "imgInfo")

SubElement(header, "classNum").text = "1"
SubElement(header, "date").text = "20230629"
SubElement(header, "worker").text = "hansolKim"

SubElement(imgInfo, "index").text = "HK-01773-1"
SubElement(imgInfo, "imgSize").text = "0"

label = SubElement(imgInfo, "label")
SubElement(label, "class").text = "0"

points = SubElement(label, "points")
SubElement(points, "x1").text = "0.675355"
SubElement(points, "y1").text = "0.594142"
SubElement(points, "x2").text = "0.298578"
SubElement(points, "y2").text = "0.761506"

imgInfo = SubElement(root, "isCrowd")


def indent(elem, level=0): #자료 출처 https://goo.gl/J8VoDK
    i = "\n" + level*"  "
    
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
            
    return elem
tree = ElementTree(root)
indent(root)
#dump(root)

#tree.write("template001.xml", encoding="utf-8",xml_declaration=True,)