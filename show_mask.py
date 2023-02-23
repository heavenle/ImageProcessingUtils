# -*- coding: utf-8 -*-
__data__ = "2023.2.23"
__author__ = "玉堃"
__description__ = "实现单波段图像的RGB图像转换。"
__function__ = ["注释: class:def[类别:函数]",
                "-:show_mask_npy: 将mask用RGB显示出来"]

import numpy as np
import glob
import os
import skimage.io as io

colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 255)]

class_value = [0, 1]


def show_mask_npy(mask):
    """
    将mask用RGB显示出来，生成RGB图像将保存根路径的上级路径show_mask下

    :param mask: 包含图像的根路径
    :return:
    """
    if isinstance(mask, np.ndarray):
        one_channel_mask = mask
    else:
        sufix = os.path.splitext(mask)
        if sufix[1] == '.tif' or sufix[1] == '.jpg':
            one_channel_mask = io.imread(mask)
        elif sufix[1] == '.npy':
            one_channel_mask = np.load(mask)
        else:
            print('error suffix, suffix ==', sufix)
            exit(0)

    height, width = one_channel_mask.shape
    img = np.zeros((height, width, 3))
    # one_channel_mask = np.argmax(mask_temp.detach().cpu().numpy(), axis=-1)

    for n_class_value in class_value:
        # linshi shezhi
        img[:, :, 0] += ((one_channel_mask[:, :] == n_class_value) * colors[n_class_value][0]).astype(
            'uint8')
        img[:, :, 1] += ((one_channel_mask[:, :] == n_class_value) * colors[n_class_value][1]).astype(
            'uint8')
        img[:, :, 2] += ((one_channel_mask[:, :] == n_class_value) * colors[n_class_value][2]).astype(
            'uint8')
    if isinstance(mask, str):
        if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(mask)), 'show_mask')):
            os.mkdir(os.path.join(os.path.dirname(os.path.dirname(mask)), 'show_mask'))
        io.imsave(os.path.join(os.path.join(os.path.dirname(os.path.dirname(mask)), 'show_mask',
                                            os.path.splitext(os.path.basename(mask))[0] + '.jpg')), img)
    else:
        return img


if __name__ == '__main__':
    img_path = glob.glob(
        r'F:\database\Segmentation_XiAn_SingleClass_VOC_Build_GF_build\mutil_size_dataset\voc\VOCdevkit\VOC2012\SegmentationClass\*')
    count = 0
    for img_name_path in img_path:
        count += 1
        print('{}/{}'.format(count, len(img_path)))
        show_mask_npy(img_name_path)
