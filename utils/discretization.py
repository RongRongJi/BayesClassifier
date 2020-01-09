import numpy as np


class Discretization:
    def __init__(self,group, threshold):
        self.maxGroup = group
        self.minInfoThreshold = threshold
        self.result = dict()
        self.result.setdefault(0, {})

    def loadData(self,sourceData):
        # data = [[1,3],[2,1],[10,4],[240,1],[300,3],[900,5],[5,10]]
        # sourceData = {'1':3,'2':1,'10':4,'240':1,'300':3,'900':5,'5':10}
        data = []
        for key, value in sourceData.items():
            data.append([float(key),value])

        return data

    # 计算信息熵
    def calEntropy(self,data):
        shannoEnt = 0.0
        totalData = 0
        for d in data:
            totalData += d[1]
        for d in data:
            prob = d[1]/totalData
            shannoEnt -= prob * np.log2(prob)
        return shannoEnt

    # 分割数据集
    def splitData(self,data):
        # 排序
        sortData = sorted(data,key=lambda d:d[0])
        ent_minus = np.inf
        index = -1
        last_ent1 = -1
        last_ent2 = -1
        for i in range(0,len(data)-1):
            data1 = sortData[:i+1]
            data2 = sortData[i+1:]
            ent1 = self.calEntropy(data1)
            ent2 = self.calEntropy(data2)
            if ent1 > ent2:
                tmp = ent1 - ent2
            else:
                tmp = ent2 - ent1
            # tmp = ent1 * len(data1) / len(data) + ent2 * len(data1) / len(data)
            if ent_minus > tmp:
                ent_minus = tmp
                index = i
                last_ent1 = ent1
                last_ent2 = ent2
        dic1 = dict()
        dic2 = dict()
        dic1["entropy"] = last_ent1
        dic1["data"] = sortData[:index+1]
        dic2["entropy"] = last_ent2
        dic2["data"] = sortData[index+1:]
        return dic1, dic2, ent_minus

    def train(self,data):
        group = 1
        # 对0值额外处理
        is_zero = False
        for d in data:
            if d[0] == 0:
                is_zero = True
                break
        if is_zero:
            sortData = sorted(data, key=lambda d: d[0])
            self.result[0]["entropy"] = 0
            self.result[0]["data"] = sortData[:1]
            self.result[1] = {}
            self.result[1]["entropy"] = np.inf
            self.result[1]["data"] = sortData[1:]
            currentKey = 1
        else:
            self.result[0]["entropy"] = np.inf
            self.result[0]["data"] = data
            currentKey = 0
        dic1, dic2, ent = self.splitData(self.result[currentKey]["data"])
        while group < self.maxGroup and ent > self.minInfoThreshold:
            self.result[currentKey] = dic1
            newKey = len(self.result)
            self.result[newKey] = dic2
            # 寻找目前最大熵
            index = -1
            max_ent = -1
            for i in range(0,len(self.result)):
                if self.result[i]["entropy"] > max_ent:
                    max_ent = self.result[i]["entropy"]
                    index = i
            currentKey = index
            group += 1
            dic1, dic2, ent = self.splitData(self.result[currentKey]["data"])

    # 数据处理
    def set_data(self):
        for i in range(0,len(self.result)):
            data = self.result[i]["data"]
            total = 0
            sum = 0
            for d in data:
                total += d[1]
                sum += d[0]*d[1]
            sum = sum / total
            self.result[i]["average"] = sum

    # 分类
    def classification(self,num):
        index = -1
        tmp = np.inf
        for i in range(0,len(self.result)):
            aver = self.result[i]["average"]
            if aver > num:
                t = aver - num
            else:
                t = num - aver
            if tmp > t:
                tmp = t
                index = i
        return index

    def __len__(self):
        return len(self.result)

    def __dict__(self):
        dic = {}
        for i in range(0,len(self.result)):
            dic[i] = 0
            for d in self.result[i]["data"]:
                dic[i] += d[1]
        return dic



