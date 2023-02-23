# -*- coding: utf-8 -*-
__data__ = "2023.2.8"
__author__ = "玉堃"
__description__ = "本脚本是随机选取验证集合，并生成数据集。"
__function__ = ["注释: class:def[类别:函数]",
                "-:create_pie_dataset"]

import numpy as np
import os
import sys
import glob
import shutil


def create_pie_dataset(root_path, save_root_path, valid_size=None):
    """
    生成PIE平台数据集合。
    前提是图像名称和标签名称一致。

    :param root_path: 图像根目录。
    :param save_root_path: 保存根目录
    :param valid_size: 验证集数量。默认是总数的0.2
    :return:
    """
    # 获取根路径下的图像文件和标签文件。
    root_file = os.listdir(root_path)
    if 'images' not in root_file:
        print('the file name included images is not \'images\'')
        sys.exit(0)

    imgs_list = glob.glob(os.path.join(root_path, 'images', '*.tif'))

    if valid_size == None:
        valid_size = int(len(imgs_list) * 0.2)
    random_index = np.random.choice(imgs_list, valid_size, replace=False)

    # 生成pie数据文件格式
    if not os.path.exists(save_root_path):
        os.mkdir(save_root_path)
        os.mkdir(os.path.join(save_root_path, 'train'))
        os.mkdir(os.path.join(save_root_path, 'train', 'images'))
        os.mkdir(os.path.join(save_root_path, 'train', 'labels'))
        os.mkdir(os.path.join(save_root_path, 'valid'))
        os.mkdir(os.path.join(save_root_path, 'valid', 'images'))
        os.mkdir(os.path.join(save_root_path, 'valid', 'labels'))
    else:
        print("save path is: ", save_root_path)
        print("save path is exist. we will remove file of path, please reconfirm[1 is true, 0 is false]")
        inp = input()
        if inp:
            shutil.rmtree(save_root_path)
        os.mkdir(save_root_path)
        os.mkdir(os.path.join(save_root_path, 'train'))
        os.mkdir(os.path.join(save_root_path, 'train', 'images'))
        os.mkdir(os.path.join(save_root_path, 'train', 'labels'))
        os.mkdir(os.path.join(save_root_path, 'valid'))
        os.mkdir(os.path.join(save_root_path, 'valid', 'images'))
        os.mkdir(os.path.join(save_root_path, 'valid', 'labels'))

    label_name = ''.join(set(['images']) ^ set(root_file))
    for img_name_path in imgs_list:
        file_name = os.path.basename(img_name_path)
        if img_name_path in random_index:
            shutil.copy(img_name_path, os.path.join(save_root_path, 'valid', 'images', file_name))
            shutil.copy(os.path.join(root_path, label_name, file_name),
                        os.path.join(save_root_path, 'valid', 'labels', file_name))
        else:
            shutil.copy(img_name_path, os.path.join(save_root_path, 'train', 'images', file_name))
            shutil.copy(os.path.join(root_path, label_name, file_name),
                        os.path.join(save_root_path, 'train', 'labels', file_name))


if __name__ == "__main__":
    root_path = r'F:\database\Segmentation_XiAn_SingleClass_VOC_Build_GF_build\build_data_512_25'
    save_path = r'F:\database\Segmentation_XiAn_SingleClass_VOC_Build_GF_build\pie'

    create_pie_dataset(root_path, save_path)
