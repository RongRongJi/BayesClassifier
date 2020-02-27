# utf-8
import numpy as np
import utils.const as const
from utils.const import is_continuous


# 朴素贝叶斯： 连续型变量采用离散化变成离散型变量
class NaiveBayes3:
    adultSet = []
    lamda = 1
    attrUnder50KDict = []
    attrAbove50KDict = []
    disc = []
    totalUnder50K = 0
    totalAbove50K = 0
    above_total = 0
    above_suc = 0
    under_total = 0
    under_suc = 0

    def __init__(self, dataSet, lamda, disc):
        self.adultSet = dataSet
        self.lamda = lamda
        self.disc = disc
        self.attrAbove50KDict, self.attrUnder50KDict = [], []
        for i in range(0,const.IF_OVER_50K):
            dic = {}
            self.attrUnder50KDict.append(dic)
            dic = {}
            self.attrAbove50KDict.append(dic)
        self.__calAttr__()  # 数据处理
        self.__prior_probability__()

    # 统计所有属性
    def __calAttr__(self):
        for arr in self.adultSet:
            if arr[const.IF_OVER_50K] == '<=50K':
                self.totalUnder50K += float(arr[const.FNLWGT])
                for i in range(0, const.IF_OVER_50K):
                    if is_continuous(i):
                        num = self.disc[i].classification(float(arr[i]))
                        if num not in self.attrUnder50KDict[i]:
                            self.attrUnder50KDict[i][num] = float(arr[const.FNLWGT])
                        else:
                            self.attrUnder50KDict[i][num] += float(arr[const.FNLWGT])
                    else:
                        if arr[i] not in self.attrUnder50KDict[i]:
                            self.attrUnder50KDict[i][arr[i]] = float(arr[const.FNLWGT])
                        else:
                            self.attrUnder50KDict[i][arr[i]] += float(arr[const.FNLWGT])
            elif arr[const.IF_OVER_50K] == '>50K':
                self.totalAbove50K += float(arr[const.FNLWGT])
                for i in range(0, const.IF_OVER_50K):
                    if is_continuous(i):
                        num = self.disc[i].classification(float(arr[i]))
                        if num not in self.attrAbove50KDict[i]:
                            self.attrAbove50KDict[i][num] = float(arr[const.FNLWGT])
                        else:
                            self.attrAbove50KDict[i][num] += float(arr[const.FNLWGT])
                    else:
                        if arr[i] not in self.attrAbove50KDict[i]:
                            self.attrAbove50KDict[i][arr[i]] = float(arr[const.FNLWGT])
                        else:
                            self.attrAbove50KDict[i][arr[i]] += float(arr[const.FNLWGT])

    # 先验概率
    def __prior_probability__(self):
        total = self.totalAbove50K + self.totalUnder50K
        # 拉普拉斯修正
        self.priorAbove50K = (self.totalAbove50K+self.lamda) / (total+2*self.lamda)
        self.priorUnder50K = (self.totalUnder50K+self.lamda) / (total+2*self.lamda)
        # 测试输出
        # print(self.priorAbove50K)
        # print(self.priorUnder50K)

    # 判别
    def distinguish(self,data, under_info='<=50K.', above_info='>50K.'):
        if self.totalUnder50K != 0:
            under50Krate = 1
        else:
            under50Krate = 0
        for i in range(0,const.IF_OVER_50K):
            if i == const.FNLWGT:
                continue
            if is_continuous(i):
                tmp = self.disc[i].classification(float(data[i]))
                if tmp in self.attrUnder50KDict[i]:
                    under50Krate *= (self.attrUnder50KDict[i][tmp] + self.lamda) / (self.totalUnder50K +self.lamda)
                else:
                    under50Krate *= self.lamda / (self.totalUnder50K + self.lamda)
            else:
                # 拉普拉斯修正
                if data[i] in self.attrUnder50KDict[i]:
                    under50Krate *= (self.attrUnder50KDict[i][data[i]] + self.lamda) / (self.totalUnder50K +self.lamda)
                else:
                    under50Krate *= self.lamda / (self.totalUnder50K + self.lamda)
            # print('under50Krate:',under50Krate)
        if self.totalAbove50K != 0:
            above50Krate = 1
        else:
            above50Krate = 0
        for i in range(0, const.IF_OVER_50K):
            if i == const.FNLWGT:
                continue
            if is_continuous(i):
                tmp = self.disc[i].classification(float(data[i]))
                if tmp in self.attrAbove50KDict[i]:
                    above50Krate *= (self.attrAbove50KDict[i][tmp] + self.lamda) / (self.totalAbove50K + self.lamda)
                else:
                    above50Krate *= self.lamda / (self.totalAbove50K + self.lamda)
            else:
                # 拉普拉斯修正
                if data[i] in self.attrAbove50KDict[i]:
                    above50Krate *= (self.attrAbove50KDict[i][data[i]] + self.lamda) / (self.totalAbove50K + self.lamda)
                else:
                    above50Krate *= self.lamda / (self.totalAbove50K + self.lamda)
            # print('above50Krate:', above50Krate)
        above50Krate *= self.priorAbove50K
        under50Krate *= self.priorUnder50K
        if above50Krate > under50Krate:
            if data[const.IF_OVER_50K] == above_info:
                return True
            else:
                return False
        elif above50Krate < under50Krate:
            if data[const.IF_OVER_50K] == under_info:
                return True
            else:
                return False
        else:
            return False


