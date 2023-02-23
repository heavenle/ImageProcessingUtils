# -*- coding: utf-8 -*-
__date__ = "2023.02.23"
__author__ = "玉堃"
__description__ = "CSDN中《Detection：目标检测常用评价指标的学习总结（IoU、TP、FP、TN、FN、Precision、Recall、F1-score、P-R曲线、AP、mAP、 ROC曲线、TPR、FPR和AUC）》文章代码。"
__function__ = ["注释: class:def[类别:函数]",
                ]

import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, f1_score, recall_score, \
    precision_recall_curve

# 创建数据
y_true = np.array(["cat"] * 15 + ["dog"] * 15 + ["bird"] * 20)
y_pred = np.array(["cat"] * 10 + ["dog"] * 4 + ["bird"] * 1 +
                  ["cat"] * 2 + ["dog"] * 12 + ["bird"] * 1 +
                  ["cat"] * 2 + ["dog"] * 1 + ["bird"] * 17)
# 生成混淆矩阵
cm = confusion_matrix(y_true, y_pred, labels=["cat", "dog", "bird"])

# # 创建数据，用数字也是可以。
# y_true = np.array([0]*15 + [1]*15 + [2]*20)
# y_pred = np.array([0]*10 + [1]*4 + [2]*1 +
#                   [0]*2 + [1]*12 + [2]*1 +
#                   [0]*2 + [1]*1 + [2]*17)
# # 生成混淆矩阵
# cm = confusion_matrix(y_true, y_pred)

# 给混淆矩阵添加索引。
conf_matrix = pd.DataFrame(cm, index=['Cat', 'Dog', 'Pig'], columns=['Cat', 'Dog', 'Pig'])

# 显示混淆矩阵。
fig, ax = plt.subplots(figsize=(4.5, 3.5))
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 19}, cmap="Blues")
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('confusion.jpg', bbox_inches='tight')
plt.show()

# 计算指标
# 宏平均方法
P_Macro_average = precision_score(y_true, y_pred, average="macro")
R_Macro_average = recall_score(y_true, y_pred, average="macro")
F1_Macro_average = f1_score(y_true, y_pred, average="macro")
print("Base Macro_average method, the precision is {}, recall is {}, and f1_score is {}"
      .format(P_Macro_average, R_Macro_average, F1_Macro_average))

# 加权平均方法
P_Weighted_average = precision_score(y_true, y_pred, average="weighted")
R_Weighted_average = recall_score(y_true, y_pred, average="weighted")
F1_Weighted_average = f1_score(y_true, y_pred, average="weighted")
print("Base Weighted_average method, the precision is {}, recall is {}, and f1_score is {}"
      .format(P_Weighted_average, R_Weighted_average, F1_Weighted_average))

# 微平均方法
P_Micro_average = precision_score(y_true, y_pred, average="micro")
R_Micro_average = recall_score(y_true, y_pred, average="micro")
F1_Micro_average = f1_score(y_true, y_pred, average="micro")
print("Base Micro_average method, the precision is {}, recall is {}, and f1_score is {}"
      .format(P_Micro_average, R_Micro_average, F1_Micro_average))

# 准确率
acc = accuracy_score(y_true, y_pred)
print("Base accuracy method, the accuracy is {}"
      .format(acc))
