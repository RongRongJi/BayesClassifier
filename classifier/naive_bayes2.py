# utf-8
import numpy as np
import utils.const as const

# 朴素贝叶斯： 连续型变量用零膨胀模型分离零后，采用高斯模型
class NaiveBayes2:
    adultSet = []
    attrUnder50KDict = []
    attrAbove50KDict = []
    laplaceCorrect = []
    totalUnder50K = 0
    totalAbove50K = 0
    priorUnder50K = 0
    priorAbove50K = 0
    lamda = 1

    def __init__(self, dataSet, lamda):
        self.adultSet = dataSet
        self.lamda = lamda
        for i in range(0,const.IF_OVER_50K):
            dic = {}
            self.attrUnder50KDict.append(dic)
            dic = {}
            self.attrAbove50KDict.append(dic)
            dic = {}
            self.laplaceCorrect.append(dic)
        self.__calAttr__()
        self.__prior_probability__()

    # 统计所有属性
    def __calAttr__(self):
        for arr in self.adultSet:
            if arr[const.IF_OVER_50K] == '<=50K':
                self.totalUnder50K += float(arr[const.FNLWGT])
                for i in range(0,const.IF_OVER_50K):
                    if arr[i] not in self.attrUnder50KDict[i]:
                        self.attrUnder50KDict[i][arr[i]] = float(arr[const.FNLWGT])
                    else:
                        self.attrUnder50KDict[i][arr[i]] += float(arr[const.FNLWGT])
                    if arr[i] not in self.laplaceCorrect[i]:
                        self.laplaceCorrect[i][arr[i]] = float(arr[const.FNLWGT])
                    else:
                        self.laplaceCorrect[i][arr[i]] += float(arr[const.FNLWGT])
            elif arr[const.IF_OVER_50K] == '>50K':
                self.totalAbove50K += float(arr[const.FNLWGT])
                for i in range(0, const.IF_OVER_50K):
                    if arr[i] not in self.attrAbove50KDict[i]:
                        self.attrAbove50KDict[i][arr[i]] = float(arr[const.FNLWGT])
                    else:
                        self.attrAbove50KDict[i][arr[i]] += float(arr[const.FNLWGT])
                    if arr[i] not in self.laplaceCorrect[i]:
                        self.laplaceCorrect[i][arr[i]] = float(arr[const.FNLWGT])
                    else:
                        self.laplaceCorrect[i][arr[i]] += float(arr[const.FNLWGT])
        # 测试输出
        # print(self.attrAbove50KDict)
        # print(self.attrUnder50KDict)

    # 先验概率
    def __prior_probability__(self):
        total = self.totalAbove50K + self.totalUnder50K
        # 拉普拉斯修正
        self.priorAbove50K = (self.totalAbove50K+self.lamda) / (total+2*self.lamda)
        self.priorUnder50K = (self.totalUnder50K+self.lamda) / (total+2*self.lamda)

    # 判别
    def distinguish(self,data):
        under50Krate = 1
        for i in range(0,const.IF_OVER_50K):
            if i == const.FNLWGT:
                continue
            if i == const.AGE or i == const.HOURS_PER_WEEK:
                average = 0
                for key, value in self.attrUnder50KDict[i].items():
                    average = average + float(key) * value
                average = average / self.totalUnder50K
                variance = 0
                for key, value in self.attrUnder50KDict[i].items():
                    variance = variance + value * np.power(float(key) - average, 2)
                variance = variance / (self.totalUnder50K-1)
                under50Krate *= 1 / (np.sqrt(2*np.pi*variance)) * np.exp(-(np.power(float(data[i])-average,2))/(2*variance))
                # print('u=',average,'  xigma2=',variance,'  pi=',math.pi,'  rate=',under50Krate)
            elif i == const.CAPITAL_GAIN or i == const.CAPITAL_LOSS:
                construct_zero = self.attrUnder50KDict[i]['-1'] / self.totalUnder50K
                # 判断是否是结构0
                if data[i] == '-1':
                    under50Krate *= construct_zero
                else:
                    average, total = 0, 0
                    for key, value in self.attrUnder50KDict[i].items():
                        if key != '-1':
                            average = average + float(key) * value
                            total += value
                    average = average / total
                    variance = 0
                    for key, value in self.attrUnder50KDict[i].items():
                        if key != '-1':
                            variance = variance + value * np.power(float(key) - average, 2)
                    variance = variance / (total - 1)
                    under50Krate *= 1 / (np.sqrt(2 * np.pi * variance)) * np.exp(
                        -(np.power(float(data[i]) - average, 2)) / (2 * variance))
            else:
                # 拉普拉斯修正
                if data[i] in self.attrUnder50KDict[i]:
                    under50Krate *= (self.attrUnder50KDict[i][data[i]]+self.lamda) / (self.totalUnder50K+len(self.laplaceCorrect[i])*self.lamda)
                else:
                    under50Krate *= self.lamda / (self.totalUnder50K+len(self.laplaceCorrect[i])*self.lamda)
            # print('under50Krate:',under50Krate)
        above50Krate = 1
        for i in range(0, const.IF_OVER_50K):
            if i == const.FNLWGT:
                continue
            if i == const.AGE or i == const.HOURS_PER_WEEK:
                average = 0
                for key, value in self.attrAbove50KDict[i].items():
                    average = average + float(key) * value
                average = average / self.totalAbove50K
                variance = 0
                for key, value in self.attrAbove50KDict[i].items():
                    variance = variance + value * np.power(float(key) - average, 2)
                variance = variance / (self.totalAbove50K-1)
                above50Krate *= 1 / (np.sqrt(2 * np.pi * variance)) * \
                                np.power(np.e, -np.power(float(data[i])-average, 2) / (2 * variance))
            elif i == const.CAPITAL_GAIN or i == const.CAPITAL_LOSS:
                construct_zero = self.attrAbove50KDict[i]['-1'] / self.totalAbove50K
                # 判断是否是结构0
                if data[i] == '-1':
                    above50Krate *= construct_zero
                else:
                    average, total = 0, 0
                    for key, value in self.attrAbove50KDict[i].items():
                        if key != '-1':
                            average = average + float(key) * value
                            total += value
                    average = average / total
                    variance = 0
                    for key, value in self.attrAbove50KDict[i].items():
                        if key != '-1':
                            variance = variance + value * np.power(float(key) - average, 2)
                    variance = variance / (total - 1)
                    above50Krate *= 1 / (np.sqrt(2 * np.pi * variance)) * np.exp(
                        -(np.power(float(data[i]) - average, 2)) / (2 * variance))
            else:
                # 拉普拉斯修正
                if data[i] in self.attrAbove50KDict[i]:
                    above50Krate *= (self.attrAbove50KDict[i][data[i]]+self.lamda) / (self.totalAbove50K+len(self.laplaceCorrect[i])*self.lamda)
                else:
                    above50Krate *= self.lamda / (self.totalAbove50K+len(self.laplaceCorrect[i])*self.lamda)
            # print('above50Krate:', above50Krate)
        above50Krate *= self.priorAbove50K
        under50Krate *= self.priorUnder50K
        # print(above50Krate,' ',under50Krate)
        if above50Krate > under50Krate:
            if data[const.IF_OVER_50K] == '>50K.':
                return True
            else:
                return False
        elif above50Krate < under50Krate:
            if data[const.IF_OVER_50K] == '<=50K.':
                return True
            else:
                return False
        else:
            return False


