# -*- coding: UTF-8 -*-
__data__ = "2022.3.30"
__author__ = "玉堃"
__description__ = "笔试题1"


"""
(c++、python ，其他语言都可以)
问题描述：
小i有N个员工，每个员工可以有多个邮箱地址，有些员工名字(名称)是一样的，为了方便账户管理，小i想整理这些员工的邮箱。
如果两个员工都有共同的邮箱地址，两个员工必定属于同一个人则合并账户，如果两个员工名字一样，邮箱地址不一样，则这两个账户不合并。
小i提供员工账户列表staff,staff[i]是员工账户，表示为字符串列表，第一个字符串staff[i][0]表示用户名称，其余字符串都表示该账户的邮箱地址。
合并员工账户后，按以下格式返回账户：每个账户的第一个元素是名称，其余元素是按字符 ASCII 顺序排列的邮箱地址。账户本身可以以任意顺序返回。

输入例子:
输入：
staff=[["wang","wangshuai@mail.com","wang00@mail.com"],
       ["wang", "xiaoiwang@mail.com"],
       ["wang", "wangshuai@mail.com","wangAllen@mail.com"],
       ["cheng", "chengcoco@mail.com"]]
输出：
[["cheng","chengcoco@mail.com"]],
 ["wang","xiaoiwan@mail.com"]，
 ['wang',"wangshuai@mail.com","wang00@mail.com"，"wangAllen@mail.com"]]
解释：
第一个和第三个 wang 是同一个人，因为他们有共同的邮箱地址"wangshuai@mail.com"。
第二个 wang 和 cheng是不同的人，因为他们的邮箱地址没有被其他帐户使用。可以以任何顺序返回这些列表。
"""

import numpy as np


def main(staff):
    userName = np.array([x[0] for x in staff])
    userNameUnique = np.unique(userName)
    output = []
    for unique_name in userNameUnique:
        # 获取唯一名字的索引。
        index_unique_name = [index for (index, value) in enumerate(list(userName)) if value == unique_name]
        temp_output = staff[index_unique_name[0]]

        if len(index_unique_name) == 1:
            output = output + [temp_output]
            continue
        output = output + recursion_loop(index_unique_name, temp_output)

    return output


def recursion_loop(index_unique_name, temp_output):
    """
    递归处理重复人名的内容
    :param index_unique_name: 重复任命的索引
    :param temp_output: 临时输出变量
    :return:
    """
    temp_index = []
    index_unique_name = index_unique_name[1:]
    for i in index_unique_name:
        set_repeat = set(temp_output[1:]) & set(staff[i][1:])
        if len(set_repeat) != 0:
            temp_output = [temp_output[0]] + list(set(temp_output[1:]) | set(staff[i][1:]))
            temp_index.append(i)
        else:
            continue
    for i in temp_index:
        index_unique_name.remove(i)

    if len(index_unique_name) == 0:
        return [temp_output]
    else:
        return [temp_output]+(recursion_loop(index_unique_name, staff[index_unique_name[0]]))


if __name__ == "__main__":
    staff = [["wang", "wangshuai@mail.com", "wang00@mail.com"],
             ["wang", "xiaoiwang@mail.com"],
             ["wang", "wangshuai@mail.com", "wangAllen@mail.com"],
             ["cheng", "chengcoco@mail.com"],
             ["cheng", "chengcheng@mail.com"],
             ["cheng", "chengcheng@mail.com", "chengpengyu@mail.com"],
             ["mei", "meiyuxuan@mail.com"],
             ["mei", "hongpumei@mail.com"],
             ["lee", "htlee@mail.com"],
             ["lee", "qq@mail.com"],
             ["lee", "linfenliyi@mail.com"],
             ["lee", "linfenliyi@mail.com", "hf_liyi@mail.com"],
             ["lee", "linfenliyi@mail.com", "hf_liyi@mail.com", "hongpulee@mail.com"],
             ["lee", "linfenliyi@mail.com", "hf_liyi@mail.com", "hongpulee@mail.com", "163@mail.com"]]

    output = main(staff)
    print(output)
