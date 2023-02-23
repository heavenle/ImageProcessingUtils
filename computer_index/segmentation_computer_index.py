# -*- coding: utf-8 -*-
__date__ = "2023.2.21"
__author__ = "玉堃"
__description__ = "本脚本是计算语义分割模型的指标，包括iou，miou，acc"
__function__ = ["注释: class:def[类别:函数]",
                "-:eval_miou",
                "-:intersect_and_union",
                "-:total_intersect_and_union",
                "-:eval_metrics",
                ]

from osgeo import gdal
import numpy as np
import os
import mmcv


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray): Prediction segmentation map.
        label (ndarray): Ground truth segmentation map.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)

    if isinstance(label, str):
        label = mmcv.imread(label, flag='unchanged', backend='pillow')
    # modify if custom classes
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        # avoid using underflow conversion
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes,), dtype=np.float32)
    total_area_union = np.zeros((num_classes,), dtype=np.float32)
    total_area_pred_label = np.zeros((num_classes,), dtype=np.float32)
    total_area_label = np.zeros((num_classes,), dtype=np.float32)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                ignore_index, label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, \
           total_area_pred_label, total_area_label


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False):
    """
    Calculate evaluation metrics
    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.
     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category evalution metrics, shape (num_classes, ).
    """

    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))
    total_area_intersect, total_area_union, total_area_pred_label, \
    total_area_label = total_intersect_and_union(results, gt_seg_maps,
                                                 num_classes, ignore_index,
                                                 label_map,
                                                 reduce_zero_label)
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    acc = total_area_intersect / total_area_label
    ret_metrics = [all_acc, acc]
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            ret_metrics.append(iou)
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                    total_area_pred_label + total_area_label)
            ret_metrics.append(dice)
    if nan_to_num is not None:
        ret_metrics = [
            np.nan_to_num(metric, nan=nan_to_num) for metric in ret_metrics
        ]
    return ret_metrics


def eval_miou(results, gt_seg_maps, num_classes):
    """
    计算语义分割模型的mIou指标。

    :param output_img:模型解译图像
    :param label: 原始标签图像
    :param num_classes: 类别数量
    :return:
    """
    all_acc, acc, iou = eval_metrics(
        results,
        gt_seg_maps,
        num_classes,
        ignore_index=255,
        metrics='mIoU',
        nan_to_num=-1)
    print("==>mIou: {:.2f}%".format(iou.sum() / num_classes * 100))


def read(path):
    dataset = gdal.Open(path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    img = dataset.ReadAsArray(0, 0, width, height)
    return img


if __name__ == "__main__":
    pre_path = [
        r"G:\国网通模型验收\铁路模型\pred\2_tielu1_pred_1.tif",
        r"G:\国网通模型验收\铁路模型\pred\1_tielu2_pred_1.tif",
        r"G:\国网通模型验收\大棚模型\pred\2_dapeng1_pred.tif",
        r"G:\国网通模型验收\大棚模型\pred\1_dapeng2_pred.tif",
        r"G:\国网通模型验收\建筑物模型\pred\1_shicha2_1_pred.tif",
        r"G:\国网通模型验收\建筑物模型\pred\2_doufan2L_pred.tif",
        r"G:\国网通模型验收\道路模型\pred\1_BJ4_pred.tif",
        r"G:\国网通模型验收\道路模型\pred\2_BJ11_pred.tif",

    ]
    label_path = [
        r"G:\国网通模型验收\铁路模型\labels\tielu1_label.tif",
        r"G:\国网通模型验收\铁路模型\labels\tielu2_label.tif",
        r"G:\国网通模型验收\大棚模型\labels\dapeng1_label.tif",
        r"G:\国网通模型验收\大棚模型\labels\dapeng2_label.tif",
        r"G:\国网通模型验收\建筑物模型\labels\shicha2_1_label.tif",
        r"G:\国网通模型验收\建筑物模型\labels\doufan2L_label.tif",
        r"G:\国网通模型验收\道路模型\labels\BJ4_label.tif",
        r"G:\国网通模型验收\道路模型\labels\BJ11_label.tif",
    ]

    for pathIndex in range(len(pre_path)):
        print("==={}===".format(os.path.basename(pre_path[pathIndex])))
        pre = read(pre_path[pathIndex])
        label = read(label_path[pathIndex])

        if 255 in np.unique(pre):
            pre = (pre / 255).astype(np.uint8)

        if 255 in np.unique(label):
            label = (label / 255).astype(np.uint8)

        eval_miou(pre, label, 2)
