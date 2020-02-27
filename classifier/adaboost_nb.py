from classifier.naive_bayes3 import NaiveBayes3
from classifier.NBTree import NBTree
import utils.const as const
from utils.discretization import Discretization
import codecs
from utils.adult_data import Adult
import numpy as np


class Adaboost_Naive_Bayes:
    above_total = 0
    above_suc = 0
    under_total = 0
    under_suc = 0
    adultSet = []
    length = 0
    nb_weak = []
    weak_alpha = []

    def __init__(self, lamda, m):
        self.lamda = lamda
        self.m = m         # 迭代次数
        self.__read_sample__()
        self.__calAttr__() # 数据处理

    # 数据处理
    def __calAttr__(self):
        continuous = [const.AGE, const.CAPITAL_GAIN, const.CAPITAL_LOSS, const.HOURS_PER_WEEK]
        self.disc = dict()
        group = [10, 30, 30, 15]
        threshold = [0, 0, 0, 0]
        index = 0
        for con in continuous:
            sourceData, sourceData = {}, {}
            for arr in self.adultSet:
                if arr[con] not in sourceData:
                    sourceData[arr[con]] = float(arr[const.FNLWGT])
                else:
                    sourceData[arr[con]] += float(arr[const.FNLWGT])
            dis = Discretization(group=group[index], threshold=threshold[index])
            data = dis.loadData(sourceData)
            dis.train(data)
            dis.set_data()
            self.disc[con] = dis
            index += 1

    # 读取初始样本集
    def __read_sample__(self):
        # 读入数据
        handleclass = Adult()
        f = codecs.open('adult-dataset/adult.data', 'r')
        line = f.readline()
        adult = []
        while line:
            bool, ret = handleclass.dataInit_ZIP(str(line))
            if bool:
                adult.append(ret)
            line = f.readline()
        # 降权
        self.length = len(adult)
        for data in adult:
            data[const.FNLWGT] = float(data[const.FNLWGT])/self.length
        self.adultSet = adult

    # ADABOOST模型主体： 使用朴素贝叶斯
    def adaboost(self):
        for i in range(0, self.m):
            suc_list, fail_list = [], []
            nb = NaiveBayes3(self.adultSet, self.lamda, self.disc)
            err = self.cal_err(nb, suc_list, fail_list)
            alpha = self.alpha(err)
            self.nb_weak.append(nb)
            self.weak_alpha.append(alpha)
            for data in suc_list:
                data[const.FNLWGT] *= np.exp(-alpha)
            for data in fail_list:
                data[const.FNLWGT] *= np.exp(alpha)
            self.adultSet = suc_list + fail_list

    # ADABOOST模型主体： 使用NBTree
    def adaboost_nbtree(self):
        for i in range(0, self.m):
            suc_list, fail_list = [], []
            nb = NBTree(self.adultSet, self.lamda, self.disc)
            nb.create_tree()
            suc, total = 0, 0
            for data in self.adultSet:
                total += 1
                if nb.util_dist_single_sample(nb.root, data):
                    suc += 1
                    suc_list.append(data)
                else:
                    fail_list.append(data)
            err = 1-(suc/total)
            alpha = self.alpha(err)
            self.nb_weak.append(nb)
            self.weak_alpha.append(alpha)
            for data in suc_list:
                data[const.FNLWGT] *= np.exp(-alpha)
            for data in fail_list:
                data[const.FNLWGT] *= np.exp(alpha)
            self.adultSet = suc_list + fail_list

    # 计算错误值
    def cal_err(self, nb, suc_list, fail_list):
        suc, total = 0, 0
        for data in self.adultSet:
            total += 1
            if nb.distinguish(data, under_info='<=50K', above_info='>50K'):
                suc += 1
                suc_list.append(data)
            else:
                fail_list.append(data)
        return 1-(suc/total)

    # 分类器系数
    def alpha(self,error):
        return 0.5*np.log((1-error)/error)

    # 测试样本
    def test_sample(self):
        handleclass = Adult()
        f = codecs.open('adult-dataset/adult.test', 'r')
        line = f.readline()
        test_set = []
        while line:
            bool, ret = handleclass.dataInit_ZIP(str(line))
            if bool:
                test_set.append(ret)
            line = f.readline()
        for test in test_set:
            if test[const.IF_OVER_50K] == '<=50K.':
                self.under_total += 1
                sign = 0
                for i in range(0, self.m):
                    if self.nb_weak[i].distinguish(test, single=True):
                        sign += self.weak_alpha[i]
                    else:
                        sign -= self.weak_alpha[i]
                if self.sgn(sign) == 1:
                    self.under_suc += 1
            else:
                self.above_total += 1
                sign = 0
                for i in range(0, self.m):
                    if self.nb_weak[i].distinguish(test, single=True):
                        sign += self.weak_alpha[i]
                    else:
                        sign -= self.weak_alpha[i]
                if self.sgn(sign) == 1:
                    self.above_suc += 1
        print(self.under_total, self.under_suc, self.above_total, self.above_suc)
        print('under rate: ', self.under_suc / self.under_total, '  above rate: ',
              self.above_suc / self.above_total, ' total:'
              , (self.under_suc + self.above_suc) / (self.under_total + self.above_total))

    # sgn函数
    def sgn(self, sign):
        if sign == 0:
            return 0
        elif sign > 0:
            return 1
        else:
            return -1