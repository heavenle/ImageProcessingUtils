# -*- coding: utf-8 -*-
__data__ = "2023.2.23"
__author__ = "玉堃"
__description__ = "xml format data convert yolov5 format data."
__function__ = [
                "注释: class:def[类别:函数]",
                "xml2yolo_box:convert_xywh: 将xmin,ymin,xmax,ymax转化为 x, y, w, h",
                "xml2yolo_box:read_xml: 读取pie中xml标签中的xmin,ymin,xmax,ymax以及图像的size",
                "xml2yolo_box:write_label_txt: 将转换好的数据转化为yolo中的txt文件。",
                "xml2yolo_box:foward: ",
                "-:pie_to_yolo: 将xml格式的数据转化为yolo格式的数据。"
                ]

import os
import glob
import os.path as osp
import xml.etree.ElementTree as ET
import argparse
import shutil
import cv2

"""
pie format:
root files:
        |-----train
        |       |-----images
        |       |       |------1.tif
        |       |       |------2.tif
        |       |       |------3.tif
        |       |-----labels
        |               |------1.xml
        |               |------2.xml
        |               |------3.xml
        |-----valid
        |       |-----images
        |       |       |------4.tif
        |       |       |------5.tif
        |       |       |------6.tif
        |       |-----labels
        |               |------4.xml
        |               |------5.xml
        |               |------6.xml
        |------create_pie_dataset.json

xml format:
<?xml version="1.0" encoding="utf-8"?>
<annotation>
	<source>
		<image>optic rs image</image>
		<annotation>Lmars RSDS2016</annotation>
		<flickrid>0</flickrid>
		<database>Lmars Detection Dataset of RS</database>
	</source>
	<object>
		<bndbox>
			<xmin>695</xmin>
			<ymin>527</ymin>
			<ymax>598</ymax>
			<xmax>759</xmax>
		</bndbox>
		<difficult>0</difficult>
		<pose>Left</pose>
		<name>aircraft</name>
		<truncated>1</truncated>
	</object>
	<object>
		<bndbox>
			<xmin>520</xmin>
			<ymin>63</ymin>
			<ymax>128</ymax>
			<xmax>584</xmax>
		</bndbox>
		<difficult>0</difficult>
		<pose>Left</pose>
		<name>aircraft</name>
		<truncated>1</truncated>
	</object>
	<filename>aircraft_293.jpg</filename>
	<segmented>0</segmented>
	<owner>
		<name>Lmars, Wuhan University</name>
		<flickrid>I do not know</flickrid>
	</owner>
	<folder>RSDS2016</folder>
	<size>
		<width>1116</width>
		<depth>3</depth>
		<height>659</height>
	</size>
</annotation>

yolo format:
root files:
        |-----train
        |       |-----images
        |       |       |------1.tif
        |       |       |------2.tif
        |       |       |------3.tif
        |       |-----labels
        |               |------1.txt
        |               |------2.txt
        |               |------3.txt
        |-----valid
        |       |-----images
        |       |       |------4.tif
        |       |       |------5.tif
        |       |       |------6.tif
        |       |-----labels
        |               |------4.txt
        |               |------5.txt
        |               |------6.txt
        |------train.txt
        |------valid.txt

train.txt format:
C:/Users/Administrator/Desktop/plane608/train/images/102_35.jpg
C:/Users/Administrator/Desktop/plane608/train/images/102_36.jpg
C:/Users/Administrator/Desktop/plane608/train/images/102_43.jpg
C:/Users/Administrator/Desktop/plane608/train/images/102_44.jpg
C:/Users/Administrator/Desktop/plane608/train/images/102_48.jpg
C:/Users/Administrator/Desktop/plane608/train/images/102_49.jpg
C:/Users/Administrator/Desktop/plane608/train/images/102_56.jpg
C:/Users/Administrator/Desktop/plane608/train/images/102_57.jpg

txt format:

label|x|y|w|h
8 0.6376518726348877 0.8533891677856446 0.07287430763244629 0.06298615932464599
8 0.7145220756530761 0.945036792755127 0.0589526355266571 0.054665178060531616

"""


class xml2yolo_box():
    """
    xmin, ymin, xmax, ymax  to x, y, w, h

    """

    def __init__(self, opt):
        super(xml2yolo_box, self).__init__()
        self.opt = opt
        self.xml_path = opt.xml_path
        self.yolo_path = opt.yolov5_path
        # self.label = eval(opt.label)
        self.label = opt.label

    def convert_xywh(self, size, box):
        """
        将xmin,ymin,xmax,ymax转化为 x, y, w, h

        :param size: (原图w,原图h)
        :param box: (xmin,ymin,xmax,ymax)
        :return:(x, y, w, h)
        """
        dw = 1. / size[0]  # 1/w
        dh = 1. / size[1]  # 1/h
        x = (box[0] + box[2]) / 2.0  # 物体在图中的中心点x坐标
        y = (box[1] + box[3]) / 2.0  # 物体在图中的中心点y坐标
        w = box[2] - box[0]  # 物体实际像素宽度
        h = box[3] - box[1]  # 物体实际像素高度
        x = x * dw  # 物体中心点x的坐标比(相当于 x/原图w)
        w = w * dw  # 物体宽度的宽度比(相当于 w/原图w)
        y = y * dh  # 物体中心点y的坐标比(相当于 y/原图h)
        h = h * dh  # 物体宽度的宽度比(相当于 h/原图h)
        return (x, y, w, h)  # 返回 相对于原图的物体中心点的x坐标比,y坐标比,宽度比,高度比,取值范围[0-1]

    def read_xml(self, xml_name: str):
        """
        读取pie中xml标签中的xmin,ymin,xmax,ymax以及图像的size

        :param xml_name: pie中的xml标签文件名
        :return:
            result(list): xml文件中的所有xmin,ymin,xmax,ymax。
            size(tuple): xml文件中的图像的width和height。
        """
        xml_et = ET.parse(xml_name)
        root = xml_et.getroot()
        result = []
        for obj in root.iter('object'):
            temp = []
            temp.append(obj.find('name').text)
            for box in obj.find('bndbox'):
                temp.append(float(box.text))
            result.append(temp)

        size = (int(root.find('size').find('width').text), int(root.find('size').find('height').text))
        return result, size

    def write_label_txt(self, xml_path, result, size):
        """
        将转换好的数据转化为yolo中的txt文件。

        :param xml_path:xml文件的路径
        :param result: xml文件中的所有xmin,ymin,xmax,ymax。
        :param size:xml文件中的图像的width和height。
        :return:
        """
        with open(os.path.join(self.yolo_path, os.path.splitext(os.path.basename(xml_path))[0] + '.txt'), 'w') as F:
            for obj in result:
                box = self.convert_xywh(size, obj[1:])
                # (x, y, w, h)
                # img = cv2.imread(os.path.join(osp.join(osp.dirname(self.opt.xml_path), "images"),
                #                               os.path.basename(xml_path).replace('.xml', '.jpg')))
                # pt1 = (int(box[0] * size[0] - box[2] * size[0] / 2), int(box[1] * size[1] - box[3] * size[1] / 2))
                # pt2 = (int(box[0] * size[0] + box[2] * size[0] / 2), int(box[1] * size[1] + box[3] * size[1] / 2))
                # cv2.rectangle(img, pt1, pt2,  (255, 0, 0), thickness=3)
                # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
                # cv2.imshow('img', img)
                # cv2.waitKey(0)
                # box = box.squeeze()
                # box = box.squeeze()
                F.write(str(self.label[obj[0]]) + ' ' + str(box[0])
                        + ' ' + str(box[1]) + ' ' + str(box[2])
                        + ' ' + str(box[3]))
                F.write('\n')

    def foward(self):
        xml_list = glob.glob(os.path.join(self.xml_path, '*.xml'))
        count = 0
        for xml_name in xml_list:
            result, size = self.read_xml(xml_name)
            self.write_label_txt(xml_name, result, size)
            count += 1
            print('{}/{}'.format(count, len(xml_list)))
            with open(os.path.join(self.opt.save_path, self.opt.mode + ".txt"), 'a') as F:
                F.write(os.path.splitext(os.path.basename(xml_name))[0])
                F.write('\n')


def pie_to_yolo(root_path: str, save_path: str, label_dict: dict):
    """
    将pie格式的数据转化为yolo格式的数据。
    
    :param root_path: 包含图像的pie路径
    :param save_path: 保存图像的yolo路径
    :param label_dict: 类别和标签值的字典
    :return: 
    """
    if not osp.exists(save_path):
        os.mkdir(save_path)
    else:
        shutil.rmtree(save_path)
        os.mkdir(save_path)

    mode = ['train', 'valid']
    for m in mode:
        os.mkdir(osp.join(save_path, m))

        images_path = osp.join(root_path, m, "images")
        labels_path = osp.join(root_path, m, "labels")

        images_save_path = osp.join(save_path, m, "images")
        labels_save_path = osp.join(save_path, m, "labels")
        os.mkdir(osp.join(save_path, m, "images"))
        os.mkdir(osp.join(save_path, m, "labels"))

        parse = argparse.ArgumentParser(description=" transform xml_file to txt_file ")
        opt = parse.parse_args()
        opt.xml_path = labels_path
        opt.yolov5_path = labels_save_path
        opt.label = label_dict
        opt.mode = m
        opt.save_path = save_path
        opt.root_path = root_path

        xml2yolo = xml2yolo_box(opt)
        xml2yolo.foward()
        for img_path in glob.glob(osp.join(images_path, "*")):
            shutil.copy(img_path, osp.join(images_save_path, osp.basename(img_path)))


if __name__ == "__main__":
    root_path = r'C:\Users\Lee\Downloads\pie'
    save_path = r"C:\Users\Lee\Downloads\yolov5"
    label_dict = {'Expressway-toll-station': 0, "vehicle": 1, "airplane": 2}
    pie_to_yolo(root_path, save_path, label_dict)
