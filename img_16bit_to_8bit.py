# -*- coding: utf-8 -*-
__data__ = "2023.2.8"
__author__ = "玉堃"
__description__ = "实现图像16位转8位"
__function__ = ["注释: class:def[类别:函数]",
                "-:remote_img_16bit_to_8bit"]

import numpy as np
from common.remote_img_process_class import RemoteImgProcess
import os.path as osp
import logging
import argparse


def remote_img_16bit_to_8bit(img_path, low_per_raw=0.01, high_per_raw=0.99):
    """
    实现遥感图像16位转8位。

    :param img_path:遥感图像的地址。
    :param low_per_raw:截断的最低百分比数。
    :param high_per_raw:截断的最高百分比数。
    :return:
    """
    logger = logging.getLogger(name='16 to 8')
    logger.setLevel(logging.DEBUG)
    # 创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # 给logger添加handler
    logger.addHandler(ch)
    logger.info('[Start] start processing.')
    remoteImgProcess = RemoteImgProcess()
    img, im_geotrans, im_proj = remoteImgProcess.read_img(img_path)
    height, width, bands = img.shape
    compress_data = np.zeros((height, width, bands), dtype="uint8")

    for i in range(bands):
        logger.info('[Processing] The {}st band be processed!'.format(i))
        # 得到百分比对应的值，得到0代表的黑边的占比
        cnt_array = np.where(img[:, :, i], 0, 1)
        num0 = np.sum(cnt_array)
        kk = num0 / (height * width)  # 得到0的比例

        # 获取截断的百分比。比非零的最小值稍大一点，比最大值稍小一点。
        low_per = low_per_raw + kk - low_per_raw * kk
        low_per = low_per * 100
        high_per = (1 - high_per_raw) * (1 - kk)
        high_per = 100 - high_per * 100
        # high_per = 100
        # 通过np.percentile取相对的百分比截断数。
        cutmin = np.percentile(img[:, :, i], low_per)
        cutmax = np.percentile(img[:, :, i], high_per)

        data_band = img[:, :, i]
        # 进行截断
        data_band[data_band < cutmin] = cutmin
        data_band[data_band > cutmax] = cutmax
        compress_data[:, :, i] = np.around((data_band[:, :] - cutmin) * 255 / (cutmax - cutmin))
    logger.info(
        '[Write] band processing completed!, the compress_data shape is ({}, {}, {})'.format(compress_data.shape[0],
                                                                                             compress_data.shape[1],
                                                                                             compress_data.shape[2]))

    remoteImgProcess.write_img(compress_data, im_geotrans, im_proj,
                               save_path=
                               osp.join("./", osp.splitext(osp.basename(img_path))[0] + "_8bit" +
                                        osp.splitext(osp.basename(img_path))[1]))
    logger.info("[Finished] Image conversion finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path",
        default=None,
        help="image path"
    )
    opt = parser.parse_args()

    # img_path = r"./GF2_PMS1_E113.1_N34.6_20190312_L1A0003878994.tif"
    remote_img_16bit_to_8bit(opt.img_path)
