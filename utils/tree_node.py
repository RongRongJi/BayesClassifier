from classifier.naive_bayes3 import NaiveBayes3
from classifier.semi_nb3 import Semi_Naive_Bayes3
import utils.const as const


class Node:
    def __init__(self, key, value, utility, dataset, disc, lamda):
        self.key = key                          # 属性序号
        self.value = value                      # 属性值
        self.utility = utility                  # 实用参数
        self.dataset = dataset                  # 划分集合
        self.node_list = []                     # 子节点
        self.recall_naive_bayes = False         # 是否调用朴素贝叶斯
        self.disc = disc                        # 离散化表
        self.nb = NaiveBayes3(self.dataset, lamda, self.disc)
        # self.nb = Semi_Naive_Bayes3(self.dataset, lamda, self.disc)

    def next_node_list(self, node):
        self.node_list.append(node)

    def recall_bayes(self):
        self.recall_naive_bayes = True

    def is_recall(self):
        return self.recall_naive_bayes

    def test_result(self, test_set):
        above_total, above_suc, under_total, under_suc = 0, 0, 0, 0
        for data in test_set:
            if data[const.IF_OVER_50K] == '<=50K.':
                under_total += float(data[const.FNLWGT])
                if self.nb.distinguish(data):
                    under_suc += float(data[const.FNLWGT])
            else:
                above_total += float(data[const.FNLWGT])
                if self.nb.distinguish(data):
                    above_suc += float(data[const.FNLWGT])
        print('key:',self.key,' value:',self.value,' util:',self.utility)
        print('under_total:',under_total,' under_suc:',under_suc,' above_total:',above_total,' above_suc:',above_suc)
        print('------')
        return above_total, above_suc, under_total, under_suc

    def test_single(self, data, single=False):
        if single:
            if self.nb.distinguish(data):
                return True
            else:
                return False
        else:
            if self.nb.distinguish(data, under_info='<=50K', above_info='>50K'):
                return True
            else:
                return False

