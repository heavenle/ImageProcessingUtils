# -*- coding: UTF-8 -*-
__data__ = "2022.3.30"
__author__ = "玉堃"
__description__ = "格式转化代码"
__function__ = ["注释: class:def[类别:函数]",
                "-:del_file",
                "-:pie_to_voc",
                "-:voc_to_pie",
                "-:pie_to_coco",
                "-:coco_to_pie"]

"""
PIE segmentation data format 
root files:
        |-----train
        |       |-----images
        |       |       |------1.tif
        |       |       |------2.tif
        |       |       |------3.tif
        |       |-----labels
        |               |------1.tif
        |               |------2.tif
        |               |------3.tif
        |-----valid
        |       |-----images
        |       |       |------4.tif
        |       |       |------5.tif
        |       |       |------6.tif
        |       |-----labels
        |               |------4.tif
        |               |------5.tif
        |               |------6.tif
        |------dataset.json
        
voc segmentation data format 
VOCdevkit/VOC2012:
        |-----ImageSets
        |       |-----Segmentation
        |       |       |------train.txt
        |       |       |------val.txt
        |       |       |------trainval.txt
        |-----JPEGImages
        |       |------1.tif
        |       |------2.tif
        |       |------3.tif
        |-----SegmentationClass
                |------1.tif
                |------2.tif
                |------3.tif
                
coco segmentation data format 
root:
        |-----annotations
        |       |------instances_train2014.json
        |       |------instances_val2014.json
        |
        |-----train2014
        |       |------1.tif
        |       |------2.tif
        |       |------3.tif
        |       
        |-----val2014
                |------1.tif
                |------2.tif
                |------3.tif
"""

import os
import glob
import shutil
import json
import skimage.io as io
import cv2
import numpy as np
import pycocotools.coco as COCO
from osgeo import gdal


def del_file(path):
    """
    删除文件夹中的内容。

    :param path:文件夹的路径
    :return:
    """
    for file in os.listdir(path):
        file_name = path + "\\" + file
        if os.path.isfile(file_name):
            os.remove(file_name)
        else:
            del_file(file_name)


def write_gdal(img, save_path):
    """
    使用gdal保存图片

    :param img: 图片值 np.array
    :param save_path: 保存图片路径
    :return:
    """
    driver = gdal.GetDriverByName("GTiff")
    out_tif = driver.Create(save_path, img.shape[1], img.shape[0], 1)
    out_tif.GetRasterBand(1).WriteArray(img[:, :])


def write_coco_json(pie_root_path, label_list):
    """
    遍历训练集和测试集生成对应的字典,并对pie的数据进行移位操作

    :param pie_root_path: pie数据的根路径
    :param label_list: 保存图像路径的列表
    :return:
    """
    # 初始化json中的参数.
    coco_dict = {"images": [],
                 "categories": [],
                 "annotations": []
                 }
    coco_images_dict = {"height": 0, "width": 0, "id": 0, "file_name": ""}
    coco_categories_dict = {"supercategory": "", "id": 0, "name": ""}
    coco_annotations_dict = {"segmentation": [], "area": 0, "iscrowd": 0, "image_id": 0, "bbox": [],
                             "category_id": 0, "id": 0}
    # 设置categories字典
    with open(os.path.join(pie_root_path, 'dataset.json'), 'r', encoding='utf-8') as F:
        data_set_json = json.load(F)

    for l_index in data_set_json['labels']:
        if l_index['class_name'] == 'background':
            continue
        else:
            coco_categories_dict["supercategory"] = l_index['class_name']
            coco_categories_dict["name"] = l_index['class_name']
            coco_categories_dict["id"] = l_index['class_value'] if l_index['class_value'] != 255 else 1
        coco_dict["categories"].append(coco_categories_dict.copy())

    for train_label_list_index in range(len(label_list)):
        print("finished write json images[{}/{}]".format(train_label_list_index + 1, len(label_list)))
        train_label = io.imread(label_list[train_label_list_index])
        # 设置图像字典
        coco_images_dict["file_name"] = os.path.basename(label_list[train_label_list_index])
        coco_images_dict["height"] = train_label.shape[0]
        coco_images_dict["width"] = train_label.shape[1]
        coco_images_dict["id"] += 1
        coco_dict["images"].append(coco_images_dict.copy())
        label = np.unique(train_label)
        if isinstance(label[0], np.uint8):
            label = [int(value) for value in label]
        # 计算当前标签图中每个类别对应coco_annotations_dict参数
        for l_index in label:
            # 初始化参数
            segmentation_point_list = []
            if l_index == 0:
                continue
            else:
                label_temp = np.zeros((train_label.shape[0], train_label.shape[1]))
                label_temp[train_label == l_index] = 1
                # 计算坐标点
                contours, hierarchy = cv2.findContours(label_temp.astype('uint8'), cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                for contours_index in range(len(contours)):
                    single_area_points = contours[contours_index].squeeze(1)
                    area_points = []
                    for point in single_area_points:
                        area_points.append(int(point[0]))
                        area_points.append(int(point[1]))
                    segmentation_point_list.append(area_points)
                coco_annotations_dict["segmentation"] = segmentation_point_list.copy()
                # 计算总面积
                for i in contours:
                    coco_annotations_dict["area"] += cv2.contourArea(i)
                coco_annotations_dict["image_id"] = coco_images_dict["id"]
                coco_annotations_dict["category_id"] = l_index if l_index != 255 else 1
                coco_annotations_dict["id"] += 1
            coco_dict["annotations"].append(coco_annotations_dict.copy())

    return coco_dict


def pie_to_voc(pie_root_path, voc_root_path):
    """
    pie格式数据转为voc格式数据

    :param pie_root_path: pie数据的根路径
    :param voc_root_path: voc数据的根路径
    :return:
    """
    # 生成voc对应文件夹。
    voc_path = os.path.join(voc_root_path, 'VOCdevkit', 'VOC2012')
    if not os.path.exists(os.path.join(voc_path, 'ImageSets', 'Segmentation')):
        os.makedirs(os.path.join(voc_path, 'ImageSets', 'Segmentation'))
    else:
        del_file(os.path.join(voc_path, 'ImageSets', 'Segmentation'))

    if not os.path.exists(os.path.join(voc_path, 'JPEGImages')):
        os.makedirs(os.path.join(voc_path, 'JPEGImages'))
    else:
        del_file(os.path.join(voc_path, 'JPEGImages'))

    if not os.path.exists(os.path.join(voc_path, 'SegmentationClass')):
        os.makedirs(os.path.join(voc_path, 'SegmentationClass'))
    else:
        del_file(os.path.join(voc_path, 'SegmentationClass'))

    # 将训练数据和验证数写成txt, 同时将所有的数据及文档移动到对应位置。
    train_list = glob.glob(os.path.join(pie_root_path, "train", "images", '*'))
    val_list = glob.glob(os.path.join(pie_root_path, "valid", "images", '*'))

    for i in range(len(train_list)):
        print("finished train images[{}/{}]".format(i + 1, len(train_list)))
        shutil.copy(train_list[i], os.path.join(voc_path, "JPEGImages", os.path.basename(train_list[i])))
        shutil.copy(os.path.join(os.path.dirname(os.path.dirname(train_list[i])), "labels",
                                 os.path.basename(train_list[i])),
                    os.path.join(voc_path, 'SegmentationClass', os.path.basename(train_list[i])))

        file_name = os.path.basename(train_list[i])
        with open(os.path.join(os.path.join(voc_path, "ImageSets", "Segmentation"), "train.txt"), 'a') as F:
            F.write(os.path.splitext(file_name)[0])
            F.write('\n')
        with open(os.path.join(os.path.join(voc_path, "ImageSets", "Segmentation"), "trainval.txt"), 'a') as F:
            F.write(os.path.splitext(file_name)[0])
            F.write('\n')

    for i in range(len(val_list)):
        print("finished valid images[{}/{}]".format(i + 1, len(val_list)))
        shutil.copy(val_list[i], os.path.join(voc_path, "JPEGImages", os.path.basename(val_list[i])))
        shutil.copy(os.path.join(os.path.dirname(os.path.dirname(val_list[i])), "labels",
                                 os.path.basename(val_list[i])),
                    os.path.join(voc_path, "SegmentationClass", os.path.basename(val_list[i])))

        file_name = os.path.basename(val_list[i])
        with open(os.path.join(os.path.join(voc_path, "ImageSets", "Segmentation"), "val.txt"), 'a') as F:
            F.write(os.path.splitext(file_name)[0])
            F.write('\n')
        with open(os.path.join(os.path.join(voc_path, "ImageSets", "Segmentation"), "trainval.txt"), 'a') as F:
            F.write(os.path.splitext(file_name)[0])
            F.write('\n')


def voc_to_pie(pie_root_path, voc_root_path, dataset_json=None):
    """
    voc格式数据转为pie格式数据。

    :param pie_root_path: pie数据路径
    :param voc_root_path: voc数据路径
    :param dataset_json: dataset.json中的内容[格式为字符串或者字典]
    :return:
    """
    # 生成pie对应文件夹。
    if not os.path.exists(os.path.join(pie_root_path, 'train', 'images')):
        os.makedirs(os.path.join(pie_root_path, 'train', 'images'))
    else:
        del_file(os.path.join(pie_root_path, 'train', 'images'))

    if not os.path.exists(os.path.join(pie_root_path, 'valid', 'images')):
        os.makedirs(os.path.join(pie_root_path, 'valid', 'images'))
    else:
        del_file(os.path.join(pie_root_path, 'valid', 'images'))

    if not os.path.exists(os.path.join(pie_root_path, 'train', 'labels')):
        os.makedirs(os.path.join(pie_root_path, 'train', 'labels'))
    else:
        del_file(os.path.join(pie_root_path, 'train', 'labels'))

    if not os.path.exists(os.path.join(pie_root_path, 'valid', 'labels')):
        os.makedirs(os.path.join(pie_root_path, 'valid', 'labels'))
    else:
        del_file(os.path.join(pie_root_path, 'valid', 'labels'))

    # 根据train.txt和val.txt来将图像移动到对应位置。
    with open(os.path.join(voc_root_path, "VOCdevkit", "VOC2012", "ImageSets", "Segmentation", "train.txt"), 'r') as F:
        img_list = F.readlines()
        for img_index in range(len(img_list)):
            print("finished train images[{}/{}]".format(img_index + 1, len(img_list)))

            shutil.copy(os.path.join(os.path.join(voc_root_path, "VOCdevkit", "VOC2012", "JPEGImages"),
                                     img_list[img_index].replace('\n', '') + '.tif'),
                        os.path.join(pie_root_path, 'train', 'images',
                                     img_list[img_index].replace('\n', '') + '.tif'))

            shutil.copy(os.path.join(os.path.join(voc_root_path, "VOCdevkit", "VOC2012", "SegmentationClass"),
                                     img_list[img_index].replace('\n', '') + '.tif'),
                        os.path.join(pie_root_path, 'train', 'labels',
                                     img_list[img_index].replace('\n', '') + '.tif'))

    with open(os.path.join(voc_root_path, "VOCdevkit", "VOC2012", "ImageSets", "Segmentation", "val.txt"), 'r') as F:
        img_list = F.readlines()
        for img_index in range(len(img_list)):
            print("finished valid images[{}/{}]".format(img_index + 1, len(img_list)))
            shutil.copy(os.path.join(os.path.join(voc_root_path, "VOCdevkit", "VOC2012", "JPEGImages"),
                                     img_list[img_index].replace('\n', '') + '.tif'),
                        os.path.join(pie_root_path, 'valid', 'images',
                                     img_list[img_index].replace('\n', '') + '.tif'))

            shutil.copy(os.path.join(os.path.join(voc_root_path, "VOCdevkit", "VOC2012", "SegmentationClass"),
                                     img_list[img_index].replace('\n', '') + '.tif'),
                        os.path.join(pie_root_path, 'valid', 'labels',
                                     img_list[img_index].replace('\n', '') + '.tif'))
    if dataset_json:
        if isinstance(dataset_json, dict):
            with open(os.path.join(pie_root_path, 'dataset.json'), 'w', encoding='utf-8') as F:
                json.dump(dataset_json, F, indent=4)
        elif isinstance(dataset_json, str):
            dataset_json = json.dumps(dataset_json)
            with open(os.path.join(pie_root_path, 'dataset.json'), 'w', encoding='utf-8') as F:
                json.dump(dataset_json, F, indent=4)


def pie_to_coco(pie_root_path, coco_root_path):
    """
    将pie格式数据转化为COCO格式数据

    :param pie_root_path: pie数据根目录
    :param coco_root_path: coco数据根目录
    :return:
    """
    # 生成对应的文件夹
    if not os.path.exists(os.path.join(coco_root_path, 'annotations')):
        os.makedirs(os.path.join(coco_root_path, 'annotations'))
    else:
        del_file(os.path.join(coco_root_path, 'annotations'))

    if not os.path.exists(os.path.join(coco_root_path, 'train2017')):
        os.makedirs(os.path.join(coco_root_path, 'train2017'))
    else:
        del_file(os.path.join(coco_root_path, 'train2017'))

    if not os.path.exists(os.path.join(coco_root_path, 'val2017')):
        os.makedirs(os.path.join(coco_root_path, 'val2017'))
    else:
        del_file(os.path.join(coco_root_path, 'val2017'))

    # 使用write_json遍历训练集和测试集生成对应的字典,并对pie的数据进行移位操作
    train_label_list = glob.glob(os.path.join(pie_root_path, "train", "labels", "*"))
    val_label_list = glob.glob(os.path.join(pie_root_path, "valid", "labels", "*"))

    coco_dict = write_coco_json(pie_root_path, train_label_list)
    with open(os.path.join(coco_root_path, 'annotations', 'instances_train2017.json'), 'w') as F:
        json.dump(coco_dict, F, indent=4)

    # 将数据移动到对应的coco的文件夹下
    for train_label_list_index in range(len(train_label_list)):
        print("finished move train images[{}/{}]".format(train_label_list_index + 1, len(train_label_list)))
        shutil.copy(os.path.join(os.path.dirname(os.path.dirname(train_label_list[train_label_list_index])), "images",
                                 os.path.basename(train_label_list[train_label_list_index])),
                    os.path.join(coco_root_path, 'train2017',
                                 os.path.basename(train_label_list[train_label_list_index])))

    coco_dict = write_coco_json(pie_root_path, val_label_list)
    with open(os.path.join(coco_root_path, 'annotations', 'instances_val2017.json'), 'w') as F:
        json.dump(coco_dict, F, indent=4)

    # 将数据移动到对应的coco的文件夹下
    for val_label_list_index in range(len(val_label_list)):
        print("finished move valid images[{}/{}]".format(val_label_list_index + 1, len(val_label_list)))
        shutil.copy(os.path.join(os.path.dirname(os.path.dirname(val_label_list[val_label_list_index])), "images",
                                 os.path.basename(val_label_list[val_label_list_index])),
                    os.path.join(coco_root_path, 'val2017',
                                 os.path.basename(val_label_list[val_label_list_index])))


def coco_to_pie(pie_root_path, coco_root_path, dataset_json=None):
    """
    完成COCO格式数据到PIE格式数据的转换
    
    :param pie_root_path: pie数据的根目录
    :param coco_root_path: coco数据的根目录
    :param dataset_json: dataset.json中的内容[格式为字符串或者字典]
    :return:
    """
    # 生成pie对应文件夹。
    if not os.path.exists(os.path.join(pie_root_path, 'train', 'images')):
        os.makedirs(os.path.join(pie_root_path, 'train', 'images'))
    else:
        del_file(os.path.join(pie_root_path, 'train', 'images'))

    if not os.path.exists(os.path.join(pie_root_path, 'valid', 'images')):
        os.makedirs(os.path.join(pie_root_path, 'valid', 'images'))
    else:
        del_file(os.path.join(pie_root_path, 'valid', 'images'))

    if not os.path.exists(os.path.join(pie_root_path, 'train', 'labels')):
        os.makedirs(os.path.join(pie_root_path, 'train', 'labels'))
    else:
        del_file(os.path.join(pie_root_path, 'train', 'labels'))

    if not os.path.exists(os.path.join(pie_root_path, 'valid', 'labels')):
        os.makedirs(os.path.join(pie_root_path, 'valid', 'labels'))
    else:
        del_file(os.path.join(pie_root_path, 'valid', 'labels'))

    train_list = glob.glob(os.path.join(coco_root_path, "train2017", "*"))
    val_list = glob.glob(os.path.join(coco_root_path, "val2017", "*"))

    coco = COCO.COCO(os.path.join(coco_root_path, "annotations", "instances_train2017.json"))
    for train_list_index in range(len(train_list)):
        print("finished  train images[{}/{}]".format(train_list_index + 1, len(train_list)))
        img_info = coco.loadImgs(train_list_index + 1)[0]
        mask = np.zeros((img_info["height"], img_info["width"])).astype('int8')
        annIds = coco.getAnnIds(imgIds=img_info['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        for anns_index in anns:
            cat = anns_index["category_id"]
            segmentation = anns_index["segmentation"]
            for seg_point in segmentation:
                pts = [[seg_point[i], seg_point[i + 1]] for i in range(0, len(seg_point), 2)]
                pts = np.array([pts], np.int32)
                cv2.fillPoly(mask, pts, color=cat)
        write_gdal(mask, os.path.join(pie_root_path, 'train', 'labels',
                                      os.path.splitext(os.path.basename(train_list[train_list_index]))[0] + ".tif"))
        shutil.copy(train_list[train_list_index],
                    os.path.join(pie_root_path, 'train', 'images',
                                 os.path.splitext(os.path.basename(train_list[train_list_index]))[0] + ".tif"))

    coco = COCO.COCO(os.path.join(coco_root_path, "annotations", "instances_val2017.json"))
    for val_list_index in range(len(val_list)):
        print("finished valid images[{}/{}]".format(val_list_index + 1, len(val_list)))
        img_info = coco.loadImgs(val_list_index + 1)[0]
        mask = np.zeros((img_info["height"], img_info["width"])).astype('int8')
        annIds = coco.getAnnIds(imgIds=img_info['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        for anns_index in anns:
            cat = anns_index["category_id"]
            segmentation = anns_index["segmentation"]
            for seg_point in segmentation:
                pts = [[seg_point[i], seg_point[i + 1]] for i in range(0, len(seg_point), 2)]
                pts = np.array([pts], np.int32)
                cv2.fillPoly(mask, pts, color=cat)

        write_gdal(mask, os.path.join(pie_root_path, 'valid', 'labels',
                                      os.path.splitext(os.path.basename(val_list[val_list_index]))[0] + ".tif"))
        shutil.copy(val_list[val_list_index],
                    os.path.join(pie_root_path, 'valid', 'images',
                                 os.path.splitext(os.path.basename(val_list[val_list_index]))[0] + ".tif"))

    if dataset_json:
        if isinstance(dataset_json, dict):
            with open(os.path.join(pie_root_path, 'dataset.json'), 'w', encoding='utf-8') as F:
                json.dump(dataset_json, F, indent=4)
        elif isinstance(dataset_json, str):
            dataset_json = json.dumps(dataset_json)
            with open(os.path.join(pie_root_path, 'dataset.json'), 'w', encoding='utf-8') as F:
                json.dump(dataset_json, F, indent=4)


def coco_to_voc():
    pass


def voc_to_coco():
    pass


if __name__ == "__main__":
    pie_path = r"F:\database\Segmentation_XiAn_SingleClass_VOC_Build_GuoWangTong\voc_512\pie"
    voc_path = r"F:\database\Segmentation_XiAn_SingleClass_VOC_Build_GuoWangTong\voc_512\voc"
    coco_path = r"F:\database\oiltank\coco"
    data_set_json = {'dir_index': {'train_valid_rate': '9:1'},
                     'sample': {'width': 1024,
                                'height': 1024,
                                'bands': 3,
                                'resolution': 1,
                                'datatype': 'Byte'},
                     'labels': [{'//': '背景默认如下，必填；其它标签自行添加。在标签影像为单波段时：class_value填像素值，'
                                       '否则采用递增方式进行；class_color填原图对应标注物的rgb值。',
                                 'class_name': 'background',
                                 'class_title': '背景',
                                 'class_color': 'rgb(0, 0, 0)',
                                 'class_value': 0},
                                {'class_name': 'Cultivated_land',
                                 'class_title': '耕地',
                                 'class_color': 'rgb(255, 255, 255)',
                                 'class_value': 255}
                                ]
                     }

    # pie_to_voc(pie_path, voc_path)
    voc_to_pie(pie_path, voc_path, dataset_json=None)
    # coco_to_pie(r"F:\database\convert_test_data\coco_2_pie", coco_path, dataset_json=None)
    # pie_to_coco(pie_path, coco_path)
