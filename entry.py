# utf-8
from utils.adult_data import Adult
from classifier.naive_bayes import NaiveBayes
from classifier.naive_bayes2 import NaiveBayes2
from classifier.TAN2 import TanBayes2
from classifier.TAN import TanBayes
from classifier.NBTree import NBTree
from classifier.adaboost_nb import Adaboost_Naive_Bayes
import codecs
import numpy as np
import utils.const as const


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

    def eva_naive_bayes_gau(self, lamda):
        # 生成分类器
        adultCreator = NaiveBayes(self.adult_data, lamda)
        # 测试数据
        f = codecs.open('adult-dataset/adult.test', 'r')
        line = f.readline()
        above_total, above_suc, under_total, under_suc = 0, 0, 0, 0
        while line:
            bool, ret = self.handleclass.dataInit(str(line))
            if bool == True:
                if ret[const.IF_OVER_50K] == '<=50K.':
                    under_total += 1
                    # 测试数据
                    if adultCreator.distinguish(ret) == True:
                        under_suc += 1
                    print('<=50: ', under_suc, ' / ', under_total)
                else:
                    above_total += 1
                    if adultCreator.distinguish(ret) == True:
                        above_suc += 1
                    print('>50: ', above_suc, ' / ', above_total)
            line = f.readline()
        print('under rate: ', under_suc / under_total, '  above rate: ', above_suc / above_total,' total rate:', (under_suc+above_suc)/(under_total+above_total))

    def eva_naive_bayes_zero_inflation(self, lamda):
        # 生成分类器
        adultCreator = NaiveBayes2(self.adult_data, lamda)
        # 测试数据
        f = codecs.open('adult-dataset/adult.test', 'r')
        line = f.readline()
        above_total, above_suc, under_total, under_suc = 0, 0, 0, 0
        while line:
            bool, ret = self.handleclass.dataInit(str(line))
            if bool == True:
                if ret[const.IF_OVER_50K] == '<=50K.':
                    under_total += 1
                    # 测试数据
                    if adultCreator.distinguish(ret) == True:
                        under_suc += 1
                    print('<=50: ', under_suc, ' / ', under_total)
                else:
                    above_total += 1
                    if adultCreator.distinguish(ret) == True:
                        above_suc += 1
                    print('>50: ', above_suc, ' / ', above_total)
            line = f.readline()
        print('under rate: ', under_suc / under_total, '  above rate: ', above_suc / above_total, ' total rate:', (under_suc + above_suc) / (under_total + above_total))

    def eva_tan_bayes_discretization(self, lamda):
        # 生成分类器
        adultCreator = TanBayes(self.adult_data, lamda)
        # 测试数据
        fw = codecs.open('adult-dataset/result.txt','w')
        index = 0
        while index < 12:
            for t in adultCreator.parent_label_list:
                fw.write(str(t))
                fw.write(" ")
            fw.write('\r\n')
            f = codecs.open('adult-dataset/adult.test', 'r')
            line = f.readline()
            above_total, above_suc, under_total, under_suc = 0, 0, 0, 0
            while line:
                bool, ret = self.handleclass.dataInit_ZIP(str(line))
                if bool == True:
                    if ret[const.IF_OVER_50K] == '<=50K.':
                        under_total += 1
                        # 测试数据
                        if adultCreator.distinguish(ret) == True:
                            under_suc += 1
                        print('<=50: ', under_suc, ' / ', under_total)
                    else:
                        above_total += 1
                        if adultCreator.distinguish(ret) == True:
                            above_suc += 1
                        print('>50: ', above_suc, ' / ', above_total)
                line = f.readline()
            index += 1
            adultCreator.create_mswt(adultCreator.pheromone_arr,index)
            fw.write(str(under_suc / under_total))
            fw.write(' ')
            fw.write(str(above_suc / above_total))
            fw.write(' ')
            fw.write(str((under_suc+above_suc)/(under_total+above_total)))
            fw.write('\r\n')
            print('index: ', index, 'under rate: ', under_suc / under_total, '  above rate: ', above_suc / above_total)
        fw.close()

    def eva_tan_bayes_zero_inflation(self, lamda):
        # 生成分类器
        adultCreator = TanBayes2(self.adult_data, lamda)
        # 测试数据
        fw = codecs.open('adult-dataset/result.txt','w')
        index = 0
        while index < 12:
            for t in adultCreator.parent_label_list:
                fw.write(str(t))
                fw.write(" ")
            fw.write('\r\n')
            f = codecs.open('adult-dataset/adult.test', 'r')
            line = f.readline()
            above_total, above_suc, under_total, under_suc = 0, 0, 0, 0
            while line:
                bool, ret = self.handleclass.dataInit_ZIP(str(line))
                if bool == True:
                    if ret[const.IF_OVER_50K] == '<=50K.':
                        under_total += 1
                        # 测试数据
                        if adultCreator.distinguish(ret) == True:
                            under_suc += 1
                        print('<=50: ', under_suc, ' / ', under_total)
                    else:
                        above_total += 1
                        if adultCreator.distinguish(ret) == True:
                            above_suc += 1
                        print('>50: ', above_suc, ' / ', above_total)
                line = f.readline()
            index += 1
            adultCreator.create_mswt(adultCreator.pheromone_arr,index)
            fw.write(str(under_suc / under_total))
            fw.write(' ')
            fw.write(str(above_suc / above_total))
            fw.write(' ')
            fw.write(str((under_suc+above_suc)/(under_total+above_total)))
            fw.write('\r\n')
            print('index: ', index, 'under rate: ', under_suc / under_total, '  above rate: ', above_suc / above_total)
        fw.close()

    def network_bayes(self, lamda):
        # 生成分类器
        adultCreator = TanBayes(self.adult_data, lamda)
        # 测试数据
        adultCreator.parent_label_list = [-1,-1,-1,-1,3,7,-1,9,-1,-1,11,-1,-1,-1]
        f = codecs.open('adult-dataset/adult.test', 'r')
        line = f.readline()
        above_total, above_suc, under_total, under_suc = 0, 0, 0, 0
        while line:
            bool, ret = self.handleclass.dataInit_ZIP(str(line))
            if bool == True:
                if ret[const.IF_OVER_50K] == '<=50K.':
                    under_total += 1
                    # 测试数据
                    if adultCreator.distinguish(ret) == True:
                        under_suc += 1
                    print('<=50: ', under_suc, ' / ', under_total)
                else:
                    above_total += 1
                    if adultCreator.distinguish(ret) == True:
                        above_suc += 1
                    print('>50: ', above_suc, ' / ', above_total)
            line = f.readline()
        print('under rate: ', under_suc / under_total, '  above rate: ', above_suc / above_total)
        print((under_suc+above_suc)/(under_total+above_total))

    def eva_NBTree(self, lamda):
        # 生成分类器
        adultCreator = NBTree(self.adult_data, lamda)
        # 测试数据
        f = codecs.open('adult-dataset/adult.test', 'r')
        line = f.readline()
        test_set = []
        while line:
            bool, ret = self.handleclass.dataInit_ZIP(str(line))
            if bool == True:
                test_set.append(ret)
            line = f.readline()
        adultCreator.distinguish(test_set)

    def eva_adaboost_nb(self, lamda, m):
        adultCreator = Adaboost_Naive_Bayes(lamda, m)
        adultCreator.adaboost()
        adultCreator.test_sample()

    def eva_adaboost_nbtree(self, lamda, m):
        adultCreator = Adaboost_Naive_Bayes(lamda, m)
        adultCreator.adaboost_nbtree()
        adultCreator.test_sample()