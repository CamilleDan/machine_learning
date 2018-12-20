# -*- coding: utf-8 -*-
"""
@author: Yangdan
"""

from math import log


# 计算香农熵
def calshannonEnt(dataSet):
    num_sample = len(dataSet)  # 计算样本数量
    labels = {}  # 用于存放分类标签的字典
    for data in dataSet:  # 从数据集中取出标签
        label = data[-1]
        if label not in labels.keys():  # 将标签存入字典
            labels[label] = 1
        labels[label] += 1
    shannonEnt = 0.0  # 初始化信息熵
    for value in labels.values():  # 计算各类占比
        prop = value / float(num_sample)
        shannonEnt -= prop * log(prop, 2)  # 计算信息熵
    return shannonEnt


# 根据属性值划分数据集
# dataset表示数据集；axis表示属性；value表示属性值
def splitDataSet(dataSet, axis, value):
    subDataset = []  # 存放子数据集
    for data in dataSet:  # 根据属性值划分数据集
        if data[axis] == value:
            sub1 = data[:axis]
            sub1.extend(data[axis + 1:])
            subDataset.append(sub1)
    return subDataset


def bestAttributeSlip(dataSet):  # 根据属性值划分数据集
    num_sample = len(dataSet)
    numAttr = len(dataSet[0]) - 1
    shannonEnt = -calshannonEnt(dataSet)
    attrShannonEnt = 0
    bestInfoGain = 0.0
    bestAttr = -1
    for i in range(numAttr):  # 遍历属性
        featList = [example[i] for example in dataSet]
        labelVal = set(featList)  # 收集属性值
        for value in labelVal:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(num_sample)
            attrShannonEnt += prob * calshannonEnt(subDataSet)  # 计算当前属性信息熵
        infoGain = shannonEnt - attrShannonEnt  # 计算当前属性信息增益
        if bestInfoGain < infoGain:
            bestInfoGain = infoGain  # 最大信息增益
            bestAttr = i  # 最优属性划分
    return bestInfoGain, bestAttr


# 创建决策树
def createTree(dataSet,attrs):
    class_list = [example[-1] for example in dataSet]
    # 情况1：样本属于同一类别
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    #
    attr=[]
    num_attr=len(dataSet[0])-1
    bestInfoGain,bestAttr=bestAttributeSlip(dataSet)
    best_attr=attrs[bestAttr]
    mytree={best_attr:{}}
    del(attrs[bestAttr])
    attr_values=[data[bestAttr] for data in dataSet]
    uniqe_value=set(attr_values)
    for value in uniqe_value:
        sub_attrs=attrs[:]
        # 递归
        mytree[best_attr][value]=createTree(splitDataSet(dataSet,bestAttr,value),sub_attrs)
    return mytree

# 使用决策树进行分类
# 参数说明：决策树， 标签， 待分类数据
def classify(input_tree, feature_labels, test_vec):
    first_str = input_tree.keys()[0]
    second_dict = input_tree[first_str]
    # 得到第特征的索引，用于后续根据此特征的分类任务
    feature_index = feature_labels.index(first_str)
    for key in second_dict.keys():
        if test_vec[feature_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                classLabel = classify(second_dict[key], feature_labels, test_vec)
            # 达到叶子节点，返回递归调用，得到分类
            else:
                classLabel = second_dict[key]


# 决策树的存储
# 决策树的构造是一个很耗时的过程，因此需要将构造好的树保存起来以备后用
# 使用pickle序列化对象
def storeTree(input_tree, filename):
    import pickle
    fw = open(filename, "w")
    pickle.dump(input_tree, fw)
    fw.close()


# 读取文件中的决策树
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


