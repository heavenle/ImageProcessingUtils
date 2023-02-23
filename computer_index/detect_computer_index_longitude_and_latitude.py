# -*- coding: utf-8 -*-
__date__ = "2023.02.23"
__author__ = "玉堃"
__description__ = "通过经纬度坐标来计算目标检测中的指标【p, r, f1】"
__function__ = ["注释: class:def[类别:函数]",
                ]

from shapely.geometry import Polygon, MultiPoint  # 多边形
import shapely
import numpy as np
from osgeo import gdal, ogr
import os
import glob


# import aiUtils
def norm(n):
    return round(n * 100, 2)


def shp_to_txt(input, ispred=True, class_name='FiledName', class_prob='score'):
    mb_v = ogr.Open(input)
    mb_l = mb_v.GetLayer()
    if class_name == "":
        class_name = 'FiledName'

    out = []
    for i in mb_l:
        mb_lv = i.geometry().GetGeometryRef(0)
        # try:
        #     txtfile = str(i[class_name])
        # except:
        txtfile = "铁塔"

        for j in range(mb_lv.GetPointCount() - 1):
            point = mb_lv.GetPoint(j)
            x, y, _ = point
            txtfile += ' ' + str(x) + ' ' + str(y)
        if ispred:
            txtfile += " " + str(0.9)
        out.append(txtfile)
    return out


def bbox_iou_eval(box1, box2):
    '''
    利用python的库函数实现非矩形的IoU计算
    :param box1: list,检测框的四个坐标[x1,y1,x2,y2,x3,y3,x4,y4]
    :param box2: lsit,检测框的四个坐标[x1,y1,x2,y2,x3,y3,x4,y4]
    :return: IoU
    '''

    rec_box1 = [min(box1[::2]), min(box1[1::2]), max(box1[::2]), max(box1[1::2])]
    rec_box2 = [min(box2[::2]), min(box2[1::2]), max(box2[::2]), max(box2[1::2])]

    if rec_box1[0] > rec_box2[2]:
        return 0  # boxA is right of boxB
    if rec_box2[0] > rec_box1[2]:
        return 0  # boxA is left of boxB
    if rec_box1[3] < rec_box2[1]:
        return 0  # boxA is above boxB
    if rec_box1[1] > rec_box2[3]:
        return 0  # boxA is below boxB

    box1 = np.array(box1).reshape(4, 2)  # 四边形二维坐标表示
    # python四边形对象，会自动计算四个点，并将四个点重新排列成
    # 左上，左下，右下，右上，左上（没错左上排了两遍）
    poly1 = Polygon(box1).convex_hull
    box2 = np.array(box2).reshape(4, 2)
    poly2 = Polygon(box2).convex_hull

    try:

        inter_area = poly1.intersection(poly2).area  # 相交面积
        iou = float(inter_area) / (poly1.area + poly2.area - inter_area)
    except shapely.geos.TopologicalError:
        print('shapely.geos.TopologicalError occured, iou set to 0')
        iou = 0

    return iou


class Metric_obj:
    def __init__(self, pred_path, target_path, class_field):
        self.class_field = class_field
        self.label_list = ['铁塔']
        self.pred_list = self.read_txt(pred_path)
        self.target_list = self.read_txt(target_path, False)

    def read_txt(self, path, ispred=True):
        list_txt = []
        list_sum = []
        for i in range(len(path)):
            listt = {}
            if path[i].split('.')[-1] == 'shp':
                list_txt = shp_to_txt(path[i], ispred, self.class_field)

            for line in list_txt:
                if ispred and float(line.split(' ')[-1]) < 0.5:
                    continue
                else:
                    if line.split(' ')[0] == "None":
                        default_class = self.label_list[0]
                        for cs in self.label_list:
                            if str(cs) != 'background' and str(cs) != '背景' and str(cs) != '0':
                                default_class = str(cs)
                                break
                        line = line.replace("None", default_class)
                    if line.split(' ')[0] in listt:
                        listt[line.split(' ')[0]].append([float(i) for i in line.split(' ')[1:]])
                    else:
                        listt[line.split(' ')[0]] = [[float(i) for i in line.split(' ')[1:]]]
            list_sum.append(listt)
            # if path[i].split('.')[-1] == 'shp':
            #     os.remove(output)
        return list_sum

    def metric(self, trush=0.1):
        # self.tp=0
        f1 = 0
        rcal = 0
        pre = 0
        mat = []

        for x in self.label_list:
            tp = 0
            sum_tar = 0
            sum_pre = 0
            for i in range(len(self.pred_list)):
                # for x in self.pred_list[i]:
                try:
                    tp += self.count_tp(i, x, trush)
                    sum_tar += len(self.target_list[i][x])
                    sum_pre += len(self.pred_list[i][x])
                except:
                    tp += 0
                    sum_tar += 0
                    sum_pre += 0

            if sum_tar == 0:
                rcal1 = 0
            else:
                rcal1 = tp / sum_tar
            if sum_pre == 0:
                pre1 = 0
            else:
                pre1 = tp / sum_pre
            rcal += rcal1
            pre += pre1
            if (pre1 + rcal1) == 0:
                f1 += 0
            else:
                f1 += 2 * (pre1 * rcal1) / (pre1 + rcal1)
            mat.append({str(x): {"TP": tp, "FN": sum_tar - tp, "FP": sum_pre - tp}})
        # recal = rcal / len(self.label_list)
        # precesion = pre / len(self.label_list)
        # f1c = f1 / len(self.label_list)
        recal = rcal
        precesion = pre
        f1c = f1
        acc = {"recall": norm(recal), "precision": norm(precesion), "f1": norm(f1c),
               "Matrix": mat}
        return acc

    def count_tp(self, ind, x, trush):
        tp = 0

        for i in self.target_list[ind][x]:
            for j in self.pred_list[ind][x]:
                if j[-1] > 0:
                    iou = bbox_iou_eval(i, j[:8])
                    if iou > trush:
                        tp += 1
        return tp

    def clear_gdal(self):
        gdal.VSICurlClearCache()


if __name__ == '__main__':
    pred_path = [
        r'G:\国网通模型验收\铁塔模型\pred\2_tieta1_pred.shp',
        r'G:\国网通模型验收\铁塔模型\pred\1_tieta2_pred.shp',
    ]
    target_path = [
        r'G:\国网通模型验收\铁塔模型\labels\tieta1_label.shp',
        r'G:\国网通模型验收\铁塔模型\labels\tieta2_label.shp',
    ]
    for pathIndex in range(len(pred_path)):
        print("==={}===".format(os.path.basename(pred_path[pathIndex])))
        a = Metric_obj([pred_path[pathIndex]], [target_path[pathIndex]], 'FiledName')
        print("==>mIou: {}".format(a.metric()))
