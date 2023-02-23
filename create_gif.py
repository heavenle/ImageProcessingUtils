# -*- coding: utf-8 -*-
__date__ = "2023.02.23"
__author__ = "玉堃"
__description__ = "生成git图"
__function__ = ["注释: class:def[类别:函数]",
                "-:create_gif",  # 生成git图
                ]

import imageio
import glob


def create_gif(imageList, outputPath):
    """
    生成GIF
    :param imageList: 图像列表->list
    :param outputPath: 输出路径->str
    """
    gifImagelist = []
    for iImagePath in imageList:
        gifImagelist.append(imageio.imread(iImagePath))
    imageio.mimsave(outputPath, gifImagelist, fps=30)


if __name__ == "__main__":
    imageList = glob.glob(r"F:\img_utils\build_postprocess\maker\*.jpg")
    outputPath = r"F:\img_utils\build_postprocess\maker.gif"
    create_gif(imageList, outputPath)
