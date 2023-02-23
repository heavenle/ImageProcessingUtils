# -*- coding: utf-8 -*-
__date__ = "2023.2.21"
__author__ = "玉堃"
__description__ = "本脚本是计算目标检测模型的指标，包括tp, fp, p, r, f1, ap, ap_class"
__function__ = ["注释: class:def[类别:函数]",
                "-:plot_pr_curve",  # 画p-r曲线的函数
                "-:plot_mc_curve",  # 画置信度下指标曲线的函数
                "-:process_batch",  # 生成tp矩阵，行代表预测框，列代表Iou阈值的数量
                "-:box_iou",  # 计算iou值。
                "-:compute_ap",  # 计算ap值。
                "-:ap_per_class",  # main函数，计算整体的p，r，ap， f1
                ]

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    # iou = (labels_num, 300), 存放每个真实标签和每个预测样本的IOU阈值。
    iou = box_iou(labels[:, 1:], detections[:, :4])
    print("iou matix is : ", iou)
    # 选出单类别的下的大于iou阈值的索引
    x = torch.where(
        (iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match

    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                            1).detach().cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list) 形状为（所有预测框的数量，）
        precision: The precision curve (list) 形状为（所有预测框的数量，）
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    # 在开头加个0，在结尾加个1。形状为（所有预测框的数量+2，）
    mrec = np.concatenate(([0.0], recall, [1.0]))
    # 由于recall是从小到大的，那么precision则是从大到小的。因此是开头加1，结尾加0.形状为（所有预测框的数量+2，）
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    # Compute the precision envelope
    # np.maximum.accumulate
    # 计算数组（或数组的特定轴）的累积最大值
    # 举例：
    # import numpy as np
    # d = np.array([2, 0, 3, -4, -2, 7, 9])
    # c = np.maximum.accumulate(d)
    # print(c)  # array([2, 2, 3, 3, 3, 7, 9])
    # np.flip 数据反转，默认是axis=0，按行反转。
    # 该语句的意思是，将mpre反转为从小到大，在计算累计最大值，再反转为从大到小。我的理解可能是让mpre更平滑。
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        # 计算前0-1，共计101个点的面积。即化成101个矩形。
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        # np.trapz用于计算曲线下的面积，注意此时用的是梯形法。
        # 参考博客https://blog.csdn.net/qq_38253797/article/details/119706121
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve
    return ap, mpre, mrec


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='./', names=(), eps=1e-16):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10). 里面都是bool值。形状为（所有框的数量 X iou阈值的数量）
        conf:  Objectness value from 0-1 (nparray). 里面都是置信度值。在0-1之间。形状为（所有框的数量，）
        pred_cls:  Predicted object classes (nparray). 里面都是预测框所预测的类别。形状为（所有框的数量，）
        target_cls:  True object classes (nparray). 里面都是真实框的所有类别。形状为（真实框的数量，），
        plot:  Plot precision-recall curve at mAP@0.5 是否绘制在IoU阈值为0.5的P-R曲线。
        save_dir:  Plot save directory 图片保存路径。
        names：为对应值和类别名，例如：（0：‘oiltanl’）
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    # 将置信度按照从大到小排序的索引。（argsort原本是从小到大排序，将conf变为-conf后即可得到，从大到小排序的索引）
    i = np.argsort(-conf)
    # 按照从大到小的索引，将tp，conf，pred_cls进行排序。（tp，conf，pred_cls每一行都是对应关系，对应一个预测框）
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    # 获取整体唯一的真实类别，并统计每个类别的数量（return_counts=Ture，则返回每个类别的数量，存储在nt中。）
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    # 全部真实类别的数量。
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    # px,py分别是绘制图像的横坐标值和纵坐标值。yolo作者通过将插值的方式（np.interp）获取recall从0-1的置信度阈值下的具体值。
    # 该阈值应该是可以自己设置的。
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    # 初始化ap，p，r。
    # 其中ap的形状为（真实类别数，iou阈值数：0.5，0.55...），
    # p的形状为（真实类别的数量，1000：这是之前设置的置信度阈值数量）
    # r的形状为（真实类别的数量，1000：这是之前设置的置信度阈值数量）
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))

    # 开始计算每个类别下的指标
    # ci表示索引，c表示具体真实指标。
    for ci, c in enumerate(unique_classes):

        # 将pred_cls中等于当前类别的索引提取出来。
        i = pred_cls == c

        # n_l是当前类别下真实框的总数量
        n_l = nt[ci]  # number of labels
        # n_p是当前类别下预测框的总数量
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            # 这个是重点。
            # tp是（所有框的数量 X iou阈值的数量），并且从大到小排序了。在tp中True，表示TP，False表示FP。
            # cumsum(0)表示逐行累加。即第二行表示第一行和第二行的和。第三行表示第一行、第二行第三行的累加。
            # 在进行相加时，true表示1，false表示0。经过cumsum（0）后，每一行都表示之前所有行的tp数量。
            # 该方法相当于西瓜书中，逐个把每个样本都作为正例，来计算P,R。
            # fpc， tpc的结果形状为（所有框的数量 X iou阈值的数量）。
            # 其中fpc的每一行表示在当前行之前的所有预测框都为正例的情况下，fp的数量。每一列表示在该IoU阈值下，不同置信度阈值下的fp数量。
            # 其中tpc的每一行表示在当前行之前的所有预测框都为正例的情况下，tp的数量。每一列表示在该IoU阈值下，不同置信度阈值下的tp数量。
            # fpc = (1 - tp[i]).cumsum(0)
            # 此处我单独跑的时候报错了，因为torch改版后不支持对bool直接进行‘-’操作，因此改为‘~‘反转操作。一样的结果
            fpc = (~tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            # R= TP/GT， n_l是所有真实框的数量，eps是怕分母为零。形状为（所有框的数量 X iou阈值的数量）
            recall = tpc / (n_l + eps)  # recall curve
            # 这个方法非常厉害，通过在曲线（-conf[i]， recall[:, 0]）下，以插值的方法计算在横坐标为-px时，对应的recall值。
            # 这个方法通过插值可以随时修改对应的置信度阈值。
            # 其中recall第二维取0，表示后续P-R曲线,f1指标是基于IoU阈值为0.5的前提下计算的。
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            # 原理同上。
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            # 开始计算不同iou阈值下的AP值。tp.shape[1]表示IoU阈值数量。
            for j in range(tp.shape[1]):
                # 通过compute_ap计算当前IoU阈值下，ap的值，其形状为真实类别的数量，1000：这是之前设置的置信度阈值数量
                # TODO:
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                # 如果要保存图片的话,则添加py的值.J=0表示仅在IoU阈值等于0.5时保存.
                if plot and j == 0:
                    # py表示在mrec和mpre对应曲线上,插值出px对应的py值.
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    # 计算f1值,其形状为(真实类别的数量，1000)。
    f1 = 2 * p * r / (p + r + eps)
    # names原本为（标签值：标签名）
    # 该语句判断标签值是否在unique_classes中
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    # 该语句按照顺序给类别名编号。
    names = {i: v for i, v in enumerate(names)}  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    # 其中f1为(真实类别的数量，1000)，mean(0)表示按第一维数求平均,结果为(1, 1000)。argmax则是取出不同置信度下的最大值的索引。
    i = f1.mean(0).argmax()  # max F1 index
    # 根据最优置信度阈值下的结果。
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    # 反求出预测的tp和fp。
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype('int32')


if __name__ == "__main__":
    # (xmin, ymin, xmax, ymax, conf, label)
    pred = torch.tensor([[6., 4., 192., 257., 0.97, 0.],
                         [9., 147., 129., 293., 0.71, 0.],
                         [229., 8., 309., 111., 0.26, 0.],
                         [201., 142., 285., 290., 0.41, 0.],
                         [319., 104., 450., 274., 0.98, 0.],
                         [345., 134., 459., 297., 0.88, 0.]])
    # (label, xmin, ymin, xmax, ymax)
    label = torch.tensor([[0., 15., 11., 213., 282.],
                          [0., 208., 30., 332., 282.],
                          [0., 312., 117., 437., 285.]])
    iouv = torch.tensor([[0.5]])
    tp = process_batch(pred, label, iouv)
    print("tp is : ", tp)
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(tp,
                                                  pred[:, 4],
                                                  pred[:, 5],
                                                  label[:, 0],
                                                  plot=False,
                                                  save_dir='./',
                                                  names={0: 'dog'})
    print("tp={}, fp={}, p={}, r={}, f1={}, ap={}, ap_class={} ".format(tp, fp, p, r, f1, ap, ap_class))
