# utf-8
import numpy as np
import utils.const as const
from utils.const import is_continuous
from utils.discretization import Discretization
from utils.tree_node import Node
import math


class NBTree:
    adultSet = []
    lamda = 1
    attrUnder50KDict = []
    attrAbove50KDict = []
    totalUnder50K = 0
    totalAbove50K = 0
    above_total = 0
    above_suc = 0
    under_total = 0
    under_suc = 0

    def __init__(self, dataSet, lamda, disc=None):
        self.adultSet = dataSet
        self.lamda = lamda
        self.select_node = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        self.disc = disc
        self.__calAttr__() # 数据处理

    # 数据处理
    def __calAttr__(self):
        for arr in self.adultSet:
            if arr[const.IF_OVER_50K] == '<=50K':
                self.attrUnder50KDict.append(arr)
                self.totalUnder50K += float(arr[const.FNLWGT])
            elif arr[const.IF_OVER_50K] == '>50K':
                self.attrAbove50KDict.append(arr)
                self.totalAbove50K += float(arr[const.FNLWGT])
        if self.disc is None:
            continuous = [const.AGE, const.CAPITAL_GAIN, const.CAPITAL_LOSS, const.HOURS_PER_WEEK]
            self.disc = dict()
            group = [10, 30, 30, 15]
            threshold = [0, 0, 0, 0]
            iso = [[], [0,-1], [0,-1], []]
            index = 0
            for con in continuous:
                sourceData, sourceData = {}, {}
                for arr in self.adultSet:
                    if arr[con] not in sourceData:
                        sourceData[arr[con]] = float(arr[const.FNLWGT])
                    else:
                        sourceData[arr[con]] += float(arr[const.FNLWGT])
                dis = Discretization(group=group[index], threshold=threshold[index], isolated=iso[index])
                data = dis.loadData(sourceData)
                dis.train(data)
                dis.set_data()
                self.disc[con] = dis
                index += 1

    def utility(self, x, adultSet):
        u = 0
        # 采用5-fold CV
        list = []
        length = len(adultSet)
        for i in range(5):
            list.append(adultSet[math.floor(i/5*length):math.floor((i+1)/5*length)])
        u += self.cross_validation(x, list[0]+list[1]+list[2]+list[3], list[4])
        u += self.cross_validation(x, list[0]+list[1]+list[2]+list[4], list[3])
        u += self.cross_validation(x, list[0]+list[1]+list[3]+list[4], list[2])
        u += self.cross_validation(x, list[0]+list[2]+list[3]+list[4], list[1])
        u += self.cross_validation(x, list[1]+list[2]+list[3]+list[4], list[0])
        return u/5

    def cross_validation(self, x, train_set, test_set):
        total_under, total_above, attr_under, attr_above, succeed = 0, 0, {}, {}, 0
        for data in train_set:
            t = self.util_classifier(x, data)
            if data[const.IF_OVER_50K] == '<=50K':
                total_under += float(data[const.FNLWGT])
                if t not in attr_under:
                    attr_under[t] = float(data[const.FNLWGT])
                else:
                    attr_under[t] += float(data[const.FNLWGT])
            else:
                total_above += float(data[const.FNLWGT])
                if t not in attr_above:
                    attr_above[t] = float(data[const.FNLWGT])
                else:
                    attr_above[t] += float(data[const.FNLWGT])
        pro_under = total_under / (total_under+total_above)
        pro_above = total_above / (total_above+total_under)
        for data in test_set:
            rate_under, rate_above = pro_under, pro_above
            t = self.util_classifier(x, data)
            if t in attr_under:
                rate_under *= attr_under[t] / total_under
            elif total_under != 0:
                rate_under *= self.lamda / total_under
            if t in attr_above:
                rate_above *= attr_above[t] / total_above
            elif total_above != 0:
                rate_above *= self.lamda / total_above
            if rate_above > rate_under and data[const.IF_OVER_50K] == '>50K':
                succeed+=1
            elif rate_above < rate_under and data[const.IF_OVER_50K] == '<=50K':
                succeed+=1
        return succeed / len(test_set)

    def util_classifier(self, x, data):
        if is_continuous(x):
            return self.disc[x].classification(float(data[x]))
        else:
            return data[x]

    def create_node(self, node):
        index, biggest = 0, 0
        for i in range(0, const.IF_OVER_50K):
            if self.select_node[i] == 0:
                tmp = self.utility(i, node.dataset)
                # print(i,' ',tmp)
                if tmp > biggest:
                    biggest = tmp
                    index = i
        self.select_node[index] = 1
        # 相对误差大于5% 且 实例需要超过30
        if biggest - node.utility > 0.05:
            if is_continuous(index):
                tmp_dic = self.disc[index].__dict__()
                for key in tmp_dic.keys():
                    child_set = []
                    for data in node.dataset:
                        if self.disc[index].classification(float(data[index])) == key:
                            child_set.append(data)
                    if len(child_set) > 30:
                        child_node = Node(index, key, biggest, child_set, self.disc, self.lamda)
                        node.next_node_list(child_node)
                        self.create_node(child_node)
                    else:
                        child_node = Node(index, key, -1, child_set, self.disc, self.lamda)
                        child_node.recall_bayes()
                        node.next_node_list(child_node)
            else:
                tmp_dic = {}
                for data in node.dataset:
                    if data[index] not in tmp_dic:
                        tmp_dic[data[index]] = []
                        tmp_dic[data[index]].append(data)
                    else:
                        tmp_dic[data[index]].append(data)
                for key, value in tmp_dic.items():
                    if len(value) > 30:
                        child_node = Node(index, key, biggest, value, self.disc, self.lamda)
                        node.next_node_list(child_node)
                        self.create_node(child_node)
                    else:
                        child_node = Node(index, key, -1, value, self.disc, self.lamda)
                        child_node.recall_bayes()
                        node.next_node_list(child_node)
        else:
            node.recall_bayes()

    # 判别
    def distinguish(self, test_set, single=False):
        if single:
            return self.util_dist_single_sample(self.root, test_set, single)
        else:
            self.create_tree()
            self.util_dist(self.root, test_set)
            # print(self.under_total, self.under_suc, self.above_total, self.above_suc)
            # print('under rate: ', self.under_suc / self.under_total, '  above rate: ',
            #       self.above_suc / self.above_total, ' total:'
            #       , (self.under_suc + self.above_suc) / (self.under_total + self.above_total))

    # 递归调用
    def util_dist(self, node, test_set):
        if node.is_recall():
            above_total, above_suc, under_total, under_suc = node.test_result(test_set)
            self.add_stat(above_total,above_suc,under_total,under_suc)
        else:
            for n in node.node_list:
                i, v, child_set = n.key, n.value, []
                if is_continuous(i):
                    for data in test_set:
                        if self.disc[i].classification(float(data[i])) == v:
                            child_set.append(data)
                else:
                    for data in test_set:
                        if data[i] == v:
                            child_set.append(data)
                self.util_dist(n, child_set)

    # 构造NBTree
    def create_tree(self):
        self.root = Node(-1, -1, -1, self.adultSet, self.disc, self.lamda)
        self.create_node(self.root)
        # print("NBTree构建完成")

    # 测试单个样本
    def util_dist_single_sample(self, node, test_data, single=False):
        if node.is_recall():
            return node.test_single(test_data, single)
        else:
            for n in node.node_list:
                i, v = n.key, n.value
                if is_continuous(i) and self.disc[i].classification(float(test_data[i])) == v:
                    return self.util_dist_single_sample(n, test_data, single)
                elif test_data[i] == v:
                    return self.util_dist_single_sample(n, test_data, single)
            return False

    # 增加
    def add_stat(self, above_total, above_suc, under_total, under_suc):
        self.above_suc += above_suc
        self.above_total += above_total
        self.under_suc += under_suc
        self.under_total += under_total








