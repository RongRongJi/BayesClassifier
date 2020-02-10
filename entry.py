# utf-8
from utils.adult_data import Adult
from classifier.naive_bayes import NaiveBayes
from classifier.naive_bayes2 import NaiveBayes2
from classifier.TAN_bayes import TanBayes
from classifier.TAN2 import TanBayes2
import codecs
import numpy as np


class Entry:

    handleclass = Adult()
    adult_data = None

    def read_file(self,flag=False):
        # 读入数据
        f = codecs.open('adult-dataset/adult.data', 'r')
        line = f.readline()
        adult = []
        if flag:
            while line:
                bool, ret = self.handleclass.dataInit_ZIP(str(line))
                if bool == True:
                    adult.append(ret)
                line = f.readline()
        else:
            while line:
                bool, ret = self.handleclass.dataInit(str(line))
                if bool == True:
                    adult.append(ret)
                line = f.readline()
        self.adult_data = adult

    def eva_naive_bayes(self, lamda):
        # 生成分类器
        adultCreator = NaiveBayes(self.adult_data, lamda)
        # 测试数据
        f = codecs.open('adult-dataset/adult.test', 'r')
        line = f.readline()
        total = 0
        success = 0
        while line:
            bool, ret = self.handleclass.dataInit(str(line))
            if bool == True:
                total += 1
                # 测试数据
                if adultCreator.distinguish(ret) == True:
                    success += 1
                print(success, ' / ', total)
            line = f.readline()
        print('success rate: ', success / total)

    def eva_naive_bayes2(self, lamda):
        # 生成分类器
        adultCreator = NaiveBayes2(self.adult_data, lamda)
        # 测试数据
        f = codecs.open('adult-dataset/adult.test', 'r')
        line = f.readline()
        total = 0
        success = 0
        while line:
            bool, ret = self.handleclass.dataInit_ZIP(str(line))
            if bool == True:
                total += 1
                # 测试数据
                if adultCreator.distinguish(ret) == True:
                    success += 1
                print(success, ' / ', total)
            line = f.readline()
        print('success rate: ', success / total)

    def eva_tan_bayes(self, lamda):
        # 生成分类器
        adultCreator = TanBayes(self.adult_data, lamda)
        # 测试数据
        index = 0
        while index < 13:
            f = codecs.open('adult-dataset/adult.test', 'r')
            line = f.readline()
            total = 0
            success = 0
            while line:
                bool, ret = self.handleclass.dataInit(str(line))
                if bool == True:
                    total += 1
                    # 测试数据
                    if adultCreator.distinguish(ret) == True:
                        success += 1
                    # print(success, ' / ', total)
                line = f.readline()
            index += 1
            adultCreator.create_mswt(adultCreator.pheromone_arr,index)
            print('index: ', index, 'success rate: ', success / total)

    def eva_tan_bayes2(self, lamda):
        # 生成分类器
        adultCreator = TanBayes2(self.adult_data, lamda)
        # 测试数据
        index = 0
        while index < 13:
            f = codecs.open('adult-dataset/adult.test', 'r')
            line = f.readline()
            total = 0
            success = 0
            while line:
                bool, ret = self.handleclass.dataInit(str(line))
                if bool == True:
                    total += 1
                    # 测试数据
                    if adultCreator.distinguish(ret) == True:
                        success += 1
                    # print(success, ' / ', total)
                line = f.readline()
            index += 1
            adultCreator.create_mswt(adultCreator.pheromone_arr,index)
            print('index: ', index, 'success rate: ', success / total)


