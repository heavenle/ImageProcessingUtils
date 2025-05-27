# -*- coding: utf-8 -*-
__date__ = "2023.2.14"
__author__ = "玉堃"
__description__ = "本脚本是滑框裁图代码"
__function__ = ["注释: class:def[类别:函数]",
                "-:slide_crop"]

import numpy as np
import os
import glob
import skimage.io as io
from data_process.remote_img_process_class import RemoteImgProcess


def slide_crop(
        img_path_list,
        label_root_path,
        save_path,
        slide_window_size=1024,
        overlap_rate=0.25,
        RemoveBlack=True,
        org_band=True
):
    """
    本函是是将img大图进行滑框取图，然后检测每个小图。最后将小图的结果拼接到大图上。

    ！！！注意: 本函数的前提条件是训练数据名称和标签数据名称必须一致

    :param img_path_list: 所有训练图像的路径列表
    :param label_root_path: 所有标签图像的根路径
    :param save_path: 保存路径的根路径
    :param slide_window_size: 滑窗大小
    :param overlap_rate: 重叠率 -> int:0-1
    :param RemoveBlack: 是否需要去除全黑图， 默认：True
    :param org_band: 是否使用原始波段， 默认 True
    """
    rip = RemoteImgProcess()
    # ------------------------------------------------------------------#
    #                创建保存目录
    # ------------------------------------------------------------------#
    if not os.path.exists(save_path + '/images'):
        os.makedirs(save_path + '/images')

    if not os.path.exists(save_path + '/masks'):
        os.makedirs(save_path + '/masks')
    count = 0
    # ------------------------------------------------------------------#
    #                遍历每个数据
    # ------------------------------------------------------------------#
    for img_path in img_path_list:
        count += 1
        print('{}/{}:[{}]'.format(count, len(img_path_list), os.path.basename(img_path)))
        file_name = '.'.join(os.path.basename(img_path).split('.')[:-1])
        img, im_geotrans, im_proj = rip.read_img(img_path)
        mask, im_geotrans, im_proj = rip.read_img(os.path.join(label_root_path, os.path.basename(img_path)))

        height = img.shape[0]
        width = img.shape[1]
        # 滑框的重叠率
        overlap_pixel = int(slide_window_size * (1 - overlap_rate))
        # ------------------------------------------------------------------#
        #                处理图像各个维度尺寸过小的情况。
        # ------------------------------------------------------------------#
        if height - slide_window_size < 0:  # 判断x是否超边界，为真则表示超边界
            x_idx = [0]
        else:
            x_idx = [x for x in range(0, height, overlap_pixel)]

        if width - slide_window_size < 0:
            y_idx = [0]
        else:
            y_idx = [y for y in range(0, width, overlap_pixel)]
        # ----------------------------------------------------------------------#
        #                判断下x,y的尺寸问题，并且设置裁剪大小，方便后续进行padding。
        # ----------------------------------------------------------------------#
        cut_width = slide_window_size
        cut_height = slide_window_size

        if height - slide_window_size < 0 and width - slide_window_size >= 0:  # x小，y正常
            cut_width = slide_window_size
            cut_height = height
            switch_flag = 1
        elif height - slide_window_size < 0 and width - slide_window_size < 0:  # x小， y小
            cut_width = width
            cut_height = height
            switch_flag = 3
        elif height - slide_window_size >= 0 and width - slide_window_size < 0:  # x正常， y小
            cut_height = slide_window_size
            cut_width = width
            switch_flag = 2
        elif height - slide_window_size >= 0 and width - slide_window_size >= 0:
            switch_flag = 0

        # ----------------------------------------------------------------------#
        #                开始滑框取图，并且保存。
        # ----------------------------------------------------------------------#
        for x_start in x_idx:
            if x_start + cut_height > height:
                x_start = height - cut_height
            for y_start in y_idx:
                if y_start + cut_width > width:
                    y_start = width - cut_width
                croped_img = img[x_start:x_start + cut_height, y_start:y_start + cut_width]
                temp = np.zeros((slide_window_size, slide_window_size, 3), dtype=np.uint8)
                if switch_flag == 1:
                    # temp = np.zeros((croped_img.shape[0], cut_height, croped_img.shape[2]), dtype=np.uint8) #此为遥感图像
                    temp[0:cut_height, 0:croped_img.shape[1], :] = croped_img
                    croped_img = temp
                elif switch_flag == 2:
                    # temp = np.zeros((cut_size, croped_img.shape[1], croped_img.shape[2]), dtype=np.uint8)
                    temp[0:croped_img.shape[0], 0:cut_width, :] = croped_img
                    croped_img = temp
                elif switch_flag == 3:
                    temp[0:cut_height, 0:cut_width, :] = croped_img
                    croped_img = temp

                croped_mask = mask[x_start:x_start + cut_height, y_start:y_start + cut_width]
                # ----------------------------------------------------------------------#
                #                依据switch_flag的设置，进行padding。
                # ----------------------------------------------------------------------#
                mask_temp = np.zeros((slide_window_size, slide_window_size), dtype=np.uint8)
                if switch_flag == 1:
                    # temp = np.zeros((croped_img.shape[0], cut_height, croped_img.shape[2]), dtype=np.uint8) #此为遥感图像
                    temp[0:cut_height, 0:croped_mask.shape[1]] = croped_mask
                    croped_mask = mask_temp
                elif switch_flag == 2:
                    # temp = np.zeros((cut_size, croped_img.shape[1], croped_img.shape[2]), dtype=np.uint8)
                    temp[0:croped_mask.shape[0], 0:cut_width] = croped_mask
                    croped_mask = mask_temp
                elif switch_flag == 3:
                    temp[0:cut_height, 0:cut_width] = croped_mask
                    croped_mask = mask_temp

                if RemoveBlack:
                    if max(croped_mask.reshape(-1)) == 0:
                        continue

                if croped_img.shape[2] > 3 and org_band:
                    np.save(save_path + '/images/' +
                            img_path.split('.')[0].split('\\')[-1] + '_' + img_path.split('.')[0].split('\\')[-2] + '_'
                            + '0' * (6 - len(str(count))) + str(count) + '.npy', croped_img)
                elif croped_img.shape[2] > 3:
                    io.imsave(save_path + '/images/' +
                              file_name + '_' + str(x_start) + '_' + str(y_start) + '.tif',
                              croped_img[:, :, 0:3].astype(np.uint8))
                else:
                    io.imsave(save_path + '/images/' +
                              file_name + '_' + str(x_start) + '_' + str(y_start) + '.tif', croped_img.astype(np.uint8))
                io.imsave(save_path + '/masks/' +
                          file_name + '_' + str(x_start) + '_' + str(y_start) + '.tif', croped_mask.astype(np.uint8))


if __name__ == "__main__":
    # 初始化参数
    slide_window_size = 512
    overlap_rate = 0.25
    label_root_path = r'F:\database\Segmentation_XiAn_SingleClass_VOC_Build_GF_build\liaoning\labels'
    img_root_path = r'F:\database\Segmentation_XiAn_SingleClass_VOC_Build_GF_build\liaoning\images'
    save_path = r'F:\database\Segmentation_XiAn_SingleClass_VOC_Build_GF_build\liaoning\build_data_' + str(
        slide_window_size) + '_' + str(overlap_rate).split('.')[1]

    slide_crop(glob.glob(os.path.join(img_root_path, '*.tif')),
               label_root_path,
               save_path,
               slide_window_size=slide_window_size,
               overlap_rate=overlap_rate,
               RemoveBlack=True,
               org_band=False)
