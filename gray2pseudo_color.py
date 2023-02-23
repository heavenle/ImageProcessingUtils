# -*- coding: utf-8 -*-
__date__ = "2023.02.23"
__author__ = "玉堃"
__description__ = "生成伪彩图"
__function__ = ["注释: class:def[类别:函数]",
                "-:get_color_map_list", # 返回用于可视化分割掩码的颜色映射，它可以支持任意数量的类。
                "-:gray2pseudo_color" # 生成伪彩图
                ]

# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import os
import os.path as osp
import sys
import numpy as np
from PIL import Image


def get_color_map_list(num_classes):
    """
    Returns the color map for visualizing the segmentation mask,
    which can support arbitrary number of classes.
    Args:
        num_classes (int): Number of classes.
    Returns:
        (list). The color map.
    """

    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]
    return color_map


def gray2pseudo_color(grt_path, save_path):
    """
    生成伪彩图

    :param grt_path:原始图像路径
    :param save_path:保存图像路径
    :return:
    """
    im = Image.open(grt_path)
    lbl = np.asarray(im)
    lbl_pil = Image.fromarray(lbl.astype(np.uint8), mode='P')
    lbl_pil.putpalette(get_color_map_list(256))
    lbl_pil.save(save_path)
