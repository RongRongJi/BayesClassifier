# utf-8
from utils.adult_data import Adult
from classifier.naive_bayes import NaiveBayes
from classifier.naive_bayes2 import NaiveBayes2
from classifier.TAN2 import TanBayes2
from classifier.TAN import TanBayes
from classifier.NBTree import NBTree
from classifier.adaboost_nb import Adaboost_Naive_Bayes
from classifier.adaboost_nbtree import Adaboost_NBTree
import codecs
import numpy as np
import utils.const as const


class Entry:

    handleclass = Adult()

    def read_file(self,path,flag=False):
        # 读入数据
        f = codecs.open(path, 'r')
        line = f.readline()
        adult = []
        if flag:
            while line:
                bool, ret = self.handleclass.dataInit_ZIP(str(line))
                if bool:
                    adult.append(ret)
                line = f.readline()
        else:
            while line:
                bool, ret = self.handleclass.dataInit(str(line))
                if bool:
                    adult.append(ret)
                line = f.readline()
        self.adult_data = adult
        return adult

    def test_result(self, test_data, classifier):
        above_total, above_suc, under_total, under_suc = 0, 0, 0, 0
        for data in test_data:
            self.handleclass.printInfo(data)
            if data[const.IF_OVER_50K] == '<=50K.':
                under_total += float(data[const.FNLWGT])
                if classifier.distinguish(data):
                    print('\033[0;32m',' Correct','\033[0m')
                    under_suc += float(data[const.FNLWGT])
                else:
                    print('\033[0;31m', ' Wrong', '\033[0m')
            else:
                above_total += float(data[const.FNLWGT])
                if classifier.distinguish(data):
                    print('\033[0;32m', ' Correct', '\033[0m')
                    above_suc += float(data[const.FNLWGT])
                else:
                    print('\033[0;31m', ' Wrong', '\033[0m')
        return above_total, above_suc, under_total, under_suc

    def eva_naive_bayes_gau(self, lamda, flag=False):
        adult_data = self.read_file('adult-dataset/adult.data',flag)
        # 生成分类器
        adultCreator = NaiveBayes(adult_data, lamda)
        # 测试数据
        test_data = self.read_file('adult-dataset/adult.test', flag)
        above_total, above_suc, under_total, under_suc = self.test_result(test_data, adultCreator)
        print('under rate: ', under_suc / under_total, '  above rate: ', above_suc / above_total,' total rate:', (under_suc+above_suc)/(under_total+above_total))

    def eva_naive_bayes_zero_inflation(self, lamda, flag=True):
        adult_data = self.read_file('adult-dataset/adult.data', flag)
        # 生成分类器
        adultCreator = NaiveBayes2(adult_data, lamda)
        # 测试数据
        test_data = self.read_file('adult-dataset/adult.test', flag)
        above_total, above_suc, under_total, under_suc = self.test_result(test_data, adultCreator)
        print('under rate: ', under_suc / under_total, '  above rate: ', above_suc / above_total, ' total rate:',
              (under_suc + above_suc) / (under_total + above_total))

    def eva_tan_bayes_discretization(self, lamda, flag=True):
        adult_data = self.read_file('adult-dataset/adult.data', flag)
        # 生成分类器
        adultCreator = TanBayes(adult_data, lamda)
        # 测试数据
        test_data = self.read_file('adult-dataset/adult.test', flag)
        fw = codecs.open('adult-dataset/result.txt','w')
        index = 3
        cur_score = 0
        # 设定m值
        adultCreator.create_mswt(adultCreator.pheromone_arr, index)
        for t in adultCreator.parent_label_list:
            fw.write(str(t))
            fw.write(" ")
        fw.write('\r\n')
        above_total, above_suc, under_total, under_suc = self.test_result(test_data, adultCreator)
        cur_score = np.sqrt((above_suc/above_total)*(under_suc/under_total))
        string = str(under_suc / under_total) + " " + str(above_suc / above_total) + " " + str((under_suc+above_suc)/(under_total+above_total)) +\
                 " " + str(cur_score) + "\r\n"
        fw.write(string)
        print('index: ', index, 'under rate: ', under_suc / under_total, '  above rate: ', above_suc / above_total)
        # 祖先节点替换
        for i in range(0, const.IF_OVER_50K):
            cur_p = adultCreator.parent_label_list[i]
            anc = adultCreator.find_ancestors(i)
            for j in anc:
                adultCreator.parent_label_list[i] = j
                above_total, above_suc, under_total, under_suc = self.test_result(test_data, adultCreator)
                gmean = np.sqrt((above_suc/above_total)*(under_suc/under_total))
                if gmean > cur_score:
                    cur_score = gmean
                    cur_p = j
                    for t in adultCreator.parent_label_list:
                        fw.write(str(t))
                        fw.write(" ")
                    fw.write('\r\n')
                    string = str(under_suc / under_total) + " " + str(above_suc / above_total) + " " + str(
                        (under_suc + above_suc) / (under_total + above_total)) + \
                             " " + str(cur_score) + "\r\n"
                    fw.write(string)
                    print('index: ', index, 'under rate: ', under_suc / under_total, '  above rate: ',
                          above_suc / above_total)
            adultCreator.parent_label_list[i] = cur_p
        fw.close()

    def eva_tan_bayes_zero_inflation(self, lamda, flag=True):
        adult_data = self.read_file('adult-dataset/adult.data', flag)
        # 生成分类器
        adultCreator = TanBayes2(adult_data, lamda)
        # 测试数据
        test_data = self.read_file('adult-dataset/adult.test', flag)
        fw = codecs.open('adult-dataset/result.txt','w')
        index = 0
        while index < 12:
            for t in adultCreator.parent_label_list:
                fw.write(str(t))
                fw.write(" ")
            fw.write('\r\n')
            above_total, above_suc, under_total, under_suc = self.test_result(test_data, adultCreator)
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

    def network_bayes(self, lamda, flag=True, network=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]):
        adult_data = self.read_file('adult-dataset/adult.data', flag)
        # 生成分类器
        adultCreator = TanBayes2(adult_data, lamda)
        # 测试数据
        test_data = self.read_file('adult-dataset/adult.test', flag)
        adultCreator.parent_label_list = network
        above_total, above_suc, under_total, under_suc = self.test_result(test_data, adultCreator)
        print('under rate: ', under_suc / under_total, '  above rate: ', above_suc / above_total)
        print((under_suc+above_suc)/(under_total+above_total))

    def eva_NBTree(self, lamda, flag=True):
        adult_data = self.read_file('adult-dataset/adult.data', flag)
        # 生成分类器
        adultCreator = NBTree(adult_data, lamda)
        # 测试数据
        test_data = self.read_file('adult-dataset/adult.test', flag)
        adultCreator.distinguish(test_data)

    def eva_adaboost_nb(self, lamda, m):
        adultCreator = Adaboost_Naive_Bayes(lamda, m)
        adultCreator.adaboost()
        adultCreator.test_sample()

    def eva_adaboost_nbtree(self, lamda, m):
        adultCreator = Adaboost_NBTree(lamda, m)
        adultCreator.adaboost_nbtree()
        adultCreator.test_sample()