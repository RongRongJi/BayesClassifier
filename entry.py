# utf-8
from utils.adult_data import Adult
from classifier.naive_bayes import NaiveBayes
from classifier.TAN_bayes import TanBayes
import codecs


class Entry:

    handleclass = Adult()
    adult_data = None

    def read_file(self):
        # 读入数据
        f = codecs.open('adult-dataset/adult.data', 'r')
        line = f.readline()
        adult = []
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

    def eva_tan_bayes(self, lamda):
        # 生成分类器
        adultCreator = TanBayes(self.adult_data, lamda)
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

