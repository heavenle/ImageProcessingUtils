# -*- coding: utf-8 -*-
__data__ = "2023.2.8"
__author__ = "玉堃"
__description__ = "遥感图像处理公共函数汇总"
__function__ = ["注释: class:def[类别:函数]",
                "RemoteImgProcess:read_img", "RemoteImgProcess:write_img", "RemoteImgProcess:tif2shp",
                "RemoteImgProcess:shp2geojson", "RemoteImgProcess:array2tif", "RemoteImgProcess:calculate_mean_std"]

import os
from osgeo import gdal, osr, ogr
import numpy as np
import skimage.io as io
import cv2


class RemoteImgProcess:
    def __init__(self):
        super(RemoteImgProcess, self).__init__()

    @staticmethod
    def read_img(img_path: str, point_pixel: list = None):
        """
        读取遥感图像

        :param img_path:遥感图像的地址。
        :param point_pixel:遥感图像的指定位置坐标，[center_x, center_y, width, height]。
        :return:
            img:图像数据 np.array
            im_geotrans: 图像的地理坐标
            im_proj: 图像的投影坐标
        """
        dataset = gdal.Open(img_path)
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        if point_pixel:
            img = dataset.ReadAsArray(point_pixel[0], point_pixel[1], point_pixel[2], point_pixel[3])
        else:
            img = dataset.ReadAsArray(0, 0, width, height)
        if len(img.shape) == 3:
            img = img.transpose(1, 2, 0)
        im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
        im_proj = dataset.GetProjection()  # 地图投影信息
        del dataset
        return img, im_geotrans, im_proj

    @staticmethod
    def write_img(img, im_geotrans, im_proj, save_path="./default.tif"):
        """
        遥感图像保存。

        :param img:图像数据
        :param im_geotrans:图像的地理坐标
        :param im_proj:图像的投影坐标
        :param save_path: 图像的保存路径，默认是"./default.tif"
        :return:
        """
        # 判断栅格数据的数据类型
        if 'int8' in img.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in img.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        if len(img.shape) == 3:
            im_height, im_width, im_bands = img.shape
        else:
            (im_height, im_width), im_bands = img.shape, 1
        print("Write image shape is ({}, {}, {})".format(img.shape[0], img.shape[1], img.shape[2]))
        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(save_path, im_width, im_height, im_bands, datatype)
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(img)  # 写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i + 1).WriteArray(img[:, :, i])
        del dataset

    @staticmethod
    def tif2shp(tif_path, shp_path):
        """
        实现将tif图像转化为shp图像。

        :param tif_path:tif图像路径。
        :param shp_path:shp图像路径。
        :return:
        """
        print("convert tif to shp........")
        dataset = gdal.Open(tif_path)
        bands = dataset.GetRasterBand(1)
        bands.SetNoDataValue(0)
        maskband = bands.GetMaskBand()
        prj = osr.SpatialReference()
        prj.ImportFromWkt(dataset.GetProjection())

        output_shp = shp_path[:-4] + ".shp"
        drv = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(output_shp):
            drv.DeleteDataSource(output_shp)
        Polygon = drv.CreateDataSource(output_shp)
        Poly_layer = Polygon.CreateLayer(shp_path[:-4], srs=prj, geom_type=ogr.wkbMultiPolygon)

        newField = ogr.FieldDefn('value', ogr.OFTReal)
        Poly_layer.CreateField(newField)
        gdal.FPolygonize(bands, maskband, Poly_layer, 0)
        Polygon.SyncToDisk()
        del dataset
        print("convert to shp Success........")

    @staticmethod
    def shp2geojson(shp_path, output_path):
        print("convert shp to geojson........")
        # 打开矢量图层
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
        gdal.SetConfigOption("SHAPE_ENCODING", "GBK")
        shp_ds = ogr.Open(shp_path)
        shp_lyr = shp_ds.GetLayer()

        # 创建结果Geojson
        baseName = os.path.basename(output_path)
        out_driver = ogr.GetDriverByName('GeoJSON')
        out_ds = out_driver.CreateDataSource(output_path)
        if out_ds.GetLayer(baseName):
            out_ds.DeleteLayer(baseName)
        out_lyr = out_ds.CreateLayer(baseName, shp_lyr.GetSpatialRef())
        out_lyr.CreateFields(shp_lyr.schema)
        out_feat = ogr.Feature(out_lyr.GetLayerDefn())

        # 生成结果文件
        for feature in shp_lyr:
            out_feat.SetGeometry(feature.geometry())
            for j in range(feature.GetFieldCount()):
                out_feat.SetField(j, feature.GetField(j))
            out_lyr.CreateFeature(out_feat)

        del out_ds
        del shp_ds
        print("convert to geojson Success........")

    @staticmethod
    def array2tif(img_path: str, save_path: str, rasterOrigin: list, pixel_size: list):
        """
        给图像添加遥感坐标。

        :param img_path: img图片的路径。
        :param save_path: 保存输出tif图片的路径。
        :param rasterOrigin: 栅格数据左下角的经纬度。[经度，维度]
        :param pixel_size: 像元大小。[xsize, ysize]
        :return:
        """
        print("convert array to tif........")
        img_data = io.imread(img_path)
        img_data = cv2.flip(img_data, 0)
        if 'int8' in img_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in img_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        width = img_data.shape[1]  # 矩阵列数
        height = img_data.shape[0]  # 矩阵行数
        bands = img_data.shape[2]

        originX = rasterOrigin[0]  # 起始像元经度
        originY = rasterOrigin[1]  # 起始像元纬度

        driver = gdal.GetDriverByName('GTiff')
        outRaster = driver.Create(save_path, width, height, bands, datatype)
        # 3 is num of bands
        # 括号中两个0表示起始像元的行列号从(0,0)开始
        outRaster.SetGeoTransform((originX, pixel_size[0], 0, originY, 0, pixel_size[1]))
        # 获取数据集第一个波段，是从1开始，不是从0开始
        for i in range(1, bands + 1):
            outband = outRaster.GetRasterBand(i)
            outband.WriteArray(img_data[:, :, i - 1])
        outRasterSRS = osr.SpatialReference()
        # 代码4326表示WGS84坐标
        outRasterSRS.ImportFromEPSG(4326)
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
        outband.FlushCache()
        print("convert to tif Success........")

    @staticmethod
    def calculate_mean_std(images_list: list):
        """
        计算图像的均值和方差.

        :param img_list:图像路径列表。["xx/xx.tif","xx/xx.tif",...]
        :return:
            mean 均值
            std 方差
        """
        print("calculate mean and std........")
        R_channel = 0
        G_channel = 0
        B_channel = 0

        for idx in range(len(images_list)):

            print("imread img [idx:[{}/{}], pathname:{}]\r".format(idx + 1, len(images_list),
                                                                   images_list[idx].split('/')[-1]))
            if images_list[idx] == '.ipynb_checkpoints':
                continue
            # img = cv2.imread(images_list[idx])
            dataset = gdal.Open(images_list[idx])
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            img = dataset.ReadAsArray(0, 0, width, height).transpose(1, 2, 0)

            R_channel = R_channel + np.sum(img[:, :, 0])
            G_channel = G_channel + np.sum(img[:, :, 1])
            B_channel = B_channel + np.sum(img[:, :, 2])

        num = len(images_list) * 1024 * 1024  # 这里（512,512）是每幅图片的大小，所有图片尺寸都一样
        R_mean = R_channel / num
        G_mean = G_channel / num
        B_mean = B_channel / num
        print('#' * 10 + "obtained mean value" + '#' * 10)
        R_channel = 0
        G_channel = 0
        B_channel = 0

        for idx in range(len(images_list)):
            dataset = gdal.Open(images_list[idx])
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            img = dataset.ReadAsArray(0, 0, width, height).transpose(1, 2, 0)

            R_channel = R_channel + np.sum((img[:, :, 0] - R_mean) ** 2)
            G_channel = G_channel + np.sum((img[:, :, 1] - G_mean) ** 2)
            B_channel = B_channel + np.sum((img[:, :, 2] - B_mean) ** 2)
            print("imread img [idx:[{}/{}], pathname:{}]\r".format(idx + 1, len(images_list),
                                                                   images_list[idx].split('/')[-1]))

        R_var = np.sqrt(R_channel / num)
        G_var = np.sqrt(G_channel / num)
        B_var = np.sqrt(B_channel / num)
        print('#' * 10 + "obtained std value" + '#' * 10)
        mean = np.array([R_mean, G_mean, B_mean], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([R_var, G_var, B_var], dtype=np.float32).reshape(1, 1, 3)
        print('mean:{}, std:{}'.format(mean, std))
        return mean, std


if __name__ == "__main__":
    rip = RemoteImgProcess()
    import glob

    i = glob.glob(r"F:\BaiduNetdiskDownload\WLD\train\images\*.jpg")
    rip.calculate_mean_std(i)
