# utf-8
import numpy as np
from classifier.naive_bayes import NaiveBayes
import utils.const as const
from utils.discretization import Discretization


# TAN分类器中的属性变量应为离散变量，因此，需要将连续变量进行离散化处理
class TanBayes(NaiveBayes):

    # 数据处理
    def __init__(self, dataSet, lamda):
        NaiveBayes.__init__(self,dataSet,lamda)
        # 对连续型变量进行离散化处理
        continuous = [const.AGE,const.CAPITAL_GAIN, const.CAPITAL_LOSS, const.HOURS_PER_WEEK]
        group = [10, 15, 15, 20]
        threshold = [0, 0, 0, 0]
        self.disc_under50K = dict()
        self.disc_above50K = dict()
        index = 0
        for con in continuous:
            dis = Discretization(group=group[index], threshold=threshold[index])
            data = dis.loadData(self.attrUnder50KDict[con])
            dis.train(data)
            dis.set_data()
            self.disc_under50K[con] = dis
            self.attrUnder50KDict[con] = dis.__dict__()
            dis = Discretization(group=group[index], threshold=threshold[index])
            data = dis.loadData(self.attrAbove50KDict[con])
            dis.train(data)
            dis.set_data()
            self.disc_above50K[con] = dis
            self.attrAbove50KDict[con] = dis.__dict__()
            index += 1
            print(con,"训练完成")
        print("离散化处理完毕")
        pheromone = self.cal_pheromone()
        print("互信息计算完毕")
        self.create_mswt(pheromone,0)
        print("TAN树建立完毕")
        self.probability_precal()

    # 统计所有属性
    def __calAttr__(self):
        self.Under50KSet = []
        self.Above50KSet = []
        for arr in self.adultSet:
            if arr[const.IF_OVER_50K] == '<=50K':
                self.totalUnder50K += 1
                self.Under50KSet.append(arr)
                for i in range(0,const.IF_OVER_50K):
                    if arr[i] not in self.attrUnder50KDict[i]:
                        self.attrUnder50KDict[i][arr[i]] = 1
                    else:
                        self.attrUnder50KDict[i][arr[i]] += 1
                    if arr[i] not in self.laplaceCorrect[i]:
                        self.laplaceCorrect[i][arr[i]] = 1
                    else:
                        self.laplaceCorrect[i][arr[i]] += 1
            elif arr[const.IF_OVER_50K] == '>50K':
                self.totalAbove50K += 1
                self.Above50KSet.append(arr)
                for i in range(0, const.IF_OVER_50K):
                    if arr[i] not in self.attrAbove50KDict[i]:
                        self.attrAbove50KDict[i][arr[i]] = 1
                    else:
                        self.attrAbove50KDict[i][arr[i]] += 1
                    if arr[i] not in self.laplaceCorrect[i]:
                        self.laplaceCorrect[i][arr[i]] = 1
                    else:
                        self.laplaceCorrect[i][arr[i]] += 1

    # 计算互信息
    def cal_pheromone(self):
        # 计算互信息
        pheromone_arr = [ [-1] * const.IF_OVER_50K for i in range(0,const.IF_OVER_50K)]
        for i in range(0, const.IF_OVER_50K):
            for j in range(i + 1, const.IF_OVER_50K):
                print(i,' ',j,' 互信息计算中...')
                if i != j and i != const.FNLWGT and j != const.FNLWGT and i != const.NATIVE_COUNTRY and j != const.NATIVE_COUNTRY:
                    # 计算联合分布 / 计算互信息I（Xi,Yj)
                    information_ij = 0
                    # 两者都是连续型变量
                    if i == const.AGE\
                            or i == const.CAPITAL_GAIN or i == const.CAPITAL_LOSS or i == const.HOURS_PER_WEEK:
                        if j == const.AGE\
                                or j == const.CAPITAL_GAIN or j == const.CAPITAL_LOSS or j == const.HOURS_PER_WEEK:
                            # Under 50K
                            for ki, vi in self.attrUnder50KDict[i].items():
                                for kj, vj in self.attrUnder50KDict[j].items():
                                    joint_probability = 0
                                    for adult in self.Under50KSet:
                                        ti = self.disc_under50K[i].classification(float(adult[i]))
                                        tj = self.disc_under50K[j].classification(float(adult[j]))
                                        if ti == float(ki) and tj == float(kj):
                                            joint_probability += 1
                                    joint_probability /= self.totalUnder50K
                                    xi_probability = vi / self.totalUnder50K
                                    xj_probability = vj / self.totalUnder50K
                                    if joint_probability != 0:
                                        information_ij += self.priorUnder50K * joint_probability * np.log(
                                            joint_probability / (xi_probability * xj_probability))
                            # Above 50K
                            for ki, vi in self.attrAbove50KDict[i].items():
                                for kj, vj in self.attrAbove50KDict[j].items():
                                    joint_probability = 0
                                    for adult in self.Above50KSet:
                                        ti = self.disc_above50K[i].classification(float(adult[i]))
                                        tj = self.disc_above50K[j].classification(float(adult[j]))
                                        if ti == float(ki) and tj == float(kj):
                                            joint_probability += 1
                                    joint_probability /= self.totalAbove50K
                                    xi_probability = vi / self.totalAbove50K
                                    xj_probability = vj / self.totalAbove50K
                                    if joint_probability != 0:
                                        information_ij += self.priorAbove50K * joint_probability * np.log(
                                            joint_probability / (xi_probability * xj_probability))
                        # i 是连续型 j 是离散型
                        else:
                            for ki, vi in self.attrUnder50KDict[i].items():
                                for kj, vj in self.attrUnder50KDict[j].items():
                                    joint_probability = 0
                                    for adult in self.Under50KSet:
                                        ti = self.disc_under50K[i].classification(float(adult[i]))
                                        if ti == float(ki) and adult[j] == kj:
                                            joint_probability += 1
                                    joint_probability /= self.totalUnder50K
                                    xi_probability = vi / self.totalUnder50K
                                    xj_probability = vj / self.totalUnder50K
                                    if joint_probability != 0:
                                        information_ij += self.priorUnder50K * joint_probability * np.log(
                                            joint_probability / (xi_probability * xj_probability))
                            for ki, vi in self.attrAbove50KDict[i].items():
                                for kj, vj in self.attrAbove50KDict[j].items():
                                    joint_probability = 0
                                    for adult in self.Above50KSet:
                                        ti = self.disc_above50K[i].classification(float(adult[i]))
                                        if ti == float(ki) and adult[j] == kj:
                                            joint_probability += 1
                                    joint_probability /= self.totalAbove50K
                                    xi_probability = vi / self.totalAbove50K
                                    xj_probability = vj / self.totalAbove50K
                                    if joint_probability != 0:
                                        information_ij += self.priorAbove50K * joint_probability * np.log(
                                            joint_probability / (xi_probability * xj_probability))
                    # i 是离散型 j 是连续型
                    elif j == const.AGE\
                        or j == const.CAPITAL_GAIN or j == const.CAPITAL_LOSS or j == const.HOURS_PER_WEEK:
                        for ki, vi in self.attrUnder50KDict[i].items():
                            for kj, vj in self.attrUnder50KDict[j].items():
                                joint_probability = 0
                                for adult in self.Under50KSet:
                                    tj = self.disc_under50K[j].classification(float(adult[j]))
                                    if tj == float(kj) and adult[i] == ki and adult[const.IF_OVER_50K] == '<=50K':
                                        joint_probability += 1
                                joint_probability /= self.totalUnder50K
                                xi_probability = vi / self.totalUnder50K
                                xj_probability = vj / self.totalUnder50K
                                if joint_probability != 0:
                                    information_ij += self.priorUnder50K * joint_probability * np.log(
                                        joint_probability / (xi_probability * xj_probability))
                        for ki, vi in self.attrAbove50KDict[i].items():
                            for kj, vj in self.attrAbove50KDict[j].items():
                                joint_probability = 0
                                for adult in self.Above50KSet:
                                    tj = self.disc_above50K[j].classification(float(adult[j]))
                                    if tj == float(kj) and adult[i] == ki:
                                        joint_probability += 1
                                joint_probability /= self.totalAbove50K
                                xi_probability = vi / self.totalAbove50K
                                xj_probability = vj / self.totalAbove50K
                                if joint_probability != 0:
                                    information_ij += self.priorAbove50K * joint_probability * np.log(
                                        joint_probability / (xi_probability * xj_probability))
                    # 都是离散型
                    else:
                        for ki, vi in self.attrUnder50KDict[i].items():
                            for kj, vj in self.attrUnder50KDict[j].items():
                                joint_probability = 0
                                for adult in self.Under50KSet:
                                    if adult[i] == ki and adult[j] == kj:
                                        joint_probability += 1
                                joint_probability /= self.totalUnder50K
                                xi_probability = vi / self.totalUnder50K
                                xj_probability = vj / self.totalUnder50K
                                if joint_probability != 0:
                                    information_ij += self.priorUnder50K * joint_probability * np.log(
                                        joint_probability / (xi_probability * xj_probability))
                        for ki, vi in self.attrAbove50KDict[i].items():
                            for kj, vj in self.attrAbove50KDict[j].items():
                                joint_probability = 0
                                for adult in self.Above50KSet:
                                    if adult[i] == ki and adult[j] == kj:
                                        joint_probability += 1
                                joint_probability /= self.totalUnder50K
                                xi_probability = vi / self.totalUnder50K
                                xj_probability = vj / self.totalUnder50K
                                if joint_probability != 0:
                                    information_ij += self.priorAbove50K * joint_probability * np.log(
                                        joint_probability / (xi_probability * xj_probability))
                    # 归一化处理
                    # 计算焓变
                    ent_i = 0
                    ent_j = 0
                    for vi in self.attrUnder50KDict[i].values():
                        prob = vi / self.totalUnder50K
                        ent_i -= prob * np.log2(prob)
                    for vj in self.attrUnder50KDict[j].values():
                        prob = vj / self.totalUnder50K
                        ent_j -= prob * np.log2(prob)
                    for vi in self.attrAbove50KDict[i].values():
                        prob = vi / self.totalAbove50K
                        ent_i -= prob * np.log2(prob)
                    for vj in self.attrAbove50KDict[j].values():
                        prob = vj / self.totalAbove50K
                        ent_j -= prob * np.log2(prob)
                    information_ij = information_ij / np.sqrt(ent_i * ent_j)
                    print(i,' ',j,' ',information_ij)
                    pheromone_arr[i][j] = information_ij
                    pheromone_arr[j][i] = information_ij
        self.pheromone_arr = pheromone_arr
        return pheromone_arr

    # 构建最大权生成树
    def create_mswt(self, pheromone_arr, tree_num):
        '''
        1.对于给定的分布P(x)，对于所有的i≠j，计算联合分布P(xi|xj)；
        2.使用第1步得到的概率分布，计算任意两个结点的互信息I(Xi,Yj)，并把I(Xi,Yj)作为这两个结点连接边的权值；
        3.计算最大权生成树(Maximum-weight spanning tree)
        a. 初始状态：n个变量(结点)，0条边
        b. 插入最大权重的边
        c. 找到下一个最大的边，并且加入到树中；要求加入后，没有环生成。否则，查找次大的边；
        d. 重复上述过程c过程直到插入了n-1条边(树建立完成)
        4.选择任意结点作为根，从根到叶子标识边的方向；
        5.可以保证，这课树的近似联合概率P'(x)和原贝叶斯网络的联合概率P(x)的相对熵最小。
        :return:
        '''
        # 构建最大权生成树
        # mwst_matrix = [[-1] * const.IF_OVER_50K in range(const.IF_OVER_50K)]
        child_label_list =  []
        for i in range(0,const.IF_OVER_50K):
            child_label_list.append([])
        selected_node = [0]
        candidate_node = []
        for i in range(1,const.IF_OVER_50K):
            # 去除fnlwgt
            if i == const.FNLWGT or i == const.NATIVE_COUNTRY:
                continue
            candidate_node.append(i)
        while len(candidate_node) > tree_num:
            max = 0
            tmps = 0
            tmpc = 0
            for s in selected_node:
                for c in candidate_node:
                    if pheromone_arr[s][c] > max:
                        max = pheromone_arr[s][c]
                        tmps = s
                        tmpc = c
            # mwst_matrix[tmps][tmpc] = max
            # mwst_matrix[tmpc][tmps] = max
            selected_node.append(tmpc)
            candidate_node.remove(tmpc)
            child_label_list[tmps].append(tmpc)

        # 指定父节点
        self.parent_label_list = [-1] * const.IF_OVER_50K
        for i in range(0,const.IF_OVER_50K):
            tmp = self.find_parent(child_label_list,i)
            self.parent_label_list[i] = tmp
        return self.parent_label_list

    # 寻找父节点
    def find_parent(self,child_label_list,node):
        for i in range(0,const.IF_OVER_50K):
            if len(child_label_list[i]) == 0:
                continue
            for arr in child_label_list[i]:
                if arr == node:
                    return i
        return -1

    # 生成树
    def util_tree(self,root,matrix,visit,parent_label_list):
        visit[root] = 1
        for i in range(0,const.IF_OVER_50K):
            if matrix[root][i] > 0 and visit[i] != 1:
                parent_label_list[root].append(i)
                self.util_tree(i,matrix,visit,parent_label_list)

    # 概率预计算
    def probability_precal(self):
        self.pre_cal_Under50K = {}
        self.pre_cal_Above50K = {}
        for i in range(0,const.IF_OVER_50K):
            parent = self.parent_label_list[i]
            if parent != -1:
                # Under 50K
                self.pre_cal_Under50K[i] = {}
                if i == const.AGE or i == const.CAPITAL_GAIN or i == const.CAPITAL_LOSS \
                  or i == const.HOURS_PER_WEEK:
                    if parent == const.AGE or parent == const.CAPITAL_GAIN \
                      or parent == const.CAPITAL_LOSS or parent == const.HOURS_PER_WEEK:
                        for key_i in self.attrUnder50KDict[i]:
                            self.pre_cal_Under50K[i][key_i] = {}
                            for key_p in self.attrUnder50KDict[parent]:
                                total = 0
                                for adult in self.Under50KSet:
                                    if self.disc_under50K[i].classification(float(adult[i])) == key_i and \
                                      self.disc_under50K[parent].classification(float(adult[parent])) == key_p:
                                        total += 1
                                total /= self.totalUnder50K
                                self.pre_cal_Under50K[i][key_i][key_p] = total
                    else:
                        for key_i in self.attrUnder50KDict[i]:
                            self.pre_cal_Under50K[i][key_i] = {}
                            for key_p in self.attrUnder50KDict[parent]:
                                total = 0
                                for adult in self.Under50KSet:
                                    if self.disc_under50K[i].classification(float(adult[i])) == key_i and \
                                      adult[parent] == key_p:
                                        total += 1
                                total /= self.totalUnder50K
                                self.pre_cal_Under50K[i][key_i][key_p] = total
                else:
                    if parent == const.AGE or parent == const.CAPITAL_GAIN \
                      or parent == const.CAPITAL_LOSS or parent == const.HOURS_PER_WEEK:
                        for key_i in self.attrUnder50KDict[i]:
                            self.pre_cal_Under50K[i][key_i] = {}
                            for key_p in self.attrUnder50KDict[parent]:
                                total = 0
                                for adult in self.Under50KSet:
                                    if self.disc_under50K[parent].classification(float(adult[parent])) == key_p \
                                      and adult[i] == key_i:
                                        total += 1
                                total /= self.totalUnder50K
                                self.pre_cal_Under50K[i][key_i][key_p] = total
                    else:
                        for key_i in self.attrUnder50KDict[i]:
                            self.pre_cal_Under50K[i][key_i] = {}
                            for key_p in self.attrUnder50KDict[parent]:
                                total = 0
                                for adult in self.Under50KSet:
                                    if adult[i] == key_i and adult[parent] == key_p:
                                        total += 1
                                total /= self.totalUnder50K
                                self.pre_cal_Under50K[i][key_i][key_p] = total
                # Above 50K
                self.pre_cal_Above50K[i] = {}
                if i == const.AGE or i == const.CAPITAL_GAIN or i == const.CAPITAL_LOSS \
                        or i == const.HOURS_PER_WEEK:
                    if parent == const.AGE or parent == const.EDUCATION_NUM or parent == const.CAPITAL_GAIN \
                            or parent == const.CAPITAL_LOSS or parent == const.HOURS_PER_WEEK:
                        for key_i in self.attrAbove50KDict[i]:
                            self.pre_cal_Above50K[i][key_i] = {}
                            for key_p in self.attrAbove50KDict[parent]:
                                total = 0
                                for adult in self.Above50KSet:
                                    if self.disc_above50K[i].classification(float(adult[i])) == key_i and \
                                            self.disc_above50K[parent].classification(float(adult[parent])) == key_p:
                                        total += 1
                                total /= self.totalAbove50K
                                self.pre_cal_Above50K[i][key_i][key_p] = total
                    else:
                        for key_i in self.attrAbove50KDict[i]:
                            self.pre_cal_Above50K[i][key_i] = {}
                            for key_p in self.attrAbove50KDict[parent]:
                                total = 0
                                for adult in self.Above50KSet:
                                    if self.disc_above50K[i].classification(float(adult[i])) == key_i and \
                                            adult[parent] == key_p:
                                        total += 1
                                total /= self.totalAbove50K
                                self.pre_cal_Above50K[i][key_i][key_p] = total
                else:
                    if parent == const.AGE or parent == const.CAPITAL_GAIN \
                            or parent == const.CAPITAL_LOSS or parent == const.HOURS_PER_WEEK:
                        for key_i in self.attrAbove50KDict[i]:
                            self.pre_cal_Above50K[i][key_i] = {}
                            for key_p in self.attrAbove50KDict[parent]:
                                total = 0
                                for adult in self.Above50KSet:
                                    if self.disc_above50K[parent].classification(float(adult[parent])) == key_p \
                                            and adult[i] == key_i:
                                        total += 1
                                total /= self.totalAbove50K
                                self.pre_cal_Above50K[i][key_i][key_p] = total
                    else:
                        for key_i in self.attrAbove50KDict[i]:
                            self.pre_cal_Above50K[i][key_i] = {}
                            for key_p in self.attrAbove50KDict[parent]:
                                total = 0
                                for adult in self.Above50KSet:
                                    if adult[i] == key_i and adult[parent] == key_p:
                                        total += 1
                                total /= self.totalAbove50K
                                self.pre_cal_Above50K[i][key_i][key_p] = total
            print(i,' 概率预处理完成')

    # 判别
    def distinguish(self,data):
        under50Krate = 1
        for i in range(0,const.IF_OVER_50K):
            if i == const.AGE \
                    or i == const.CAPITAL_GAIN or i == const.CAPITAL_LOSS or i == const.HOURS_PER_WEEK:
                # 查找父类
                parent = self.parent_label_list[i]
                # 无父类
                if parent == -1:
                    part_i = self.disc_under50K[i].classification(float(data[i]))
                    under50Krate *= (self.attrUnder50KDict[i][part_i] + self.lamda) / (
                        self.totalUnder50K + len(self.laplaceCorrect[i]) * self.lamda)
                # 有父类 父类是连续型
                elif parent == const.AGE \
                    or parent == const.CAPITAL_GAIN or parent == const.CAPITAL_LOSS or parent == const.HOURS_PER_WEEK:
                    part_i = self.disc_under50K[i].classification(float(data[i]))
                    part_p = self.disc_under50K[parent].classification(float(data[parent]))
                    under50Krate *= self.pre_cal_Under50K[i][part_i][part_p]
                # 有父类 父类是离散型
                else:
                    part_i = self.disc_under50K[i].classification(float(data[i]))
                    if data[parent] in self.pre_cal_Under50K[i][part_i]:
                        under50Krate *= self.pre_cal_Under50K[i][part_i][data[parent]]
                    else:
                        under50Krate *= self.lamda / (self.totalUnder50K + len(self.laplaceCorrect[i]) * self.lamda)
            else:
                # i 是离散型
                # 查找父类
                parent = self.parent_label_list[i]
                # 无父类
                if parent == -1:
                    if data[i] in self.attrUnder50KDict[i]:
                        under50Krate *= (self.attrUnder50KDict[i][data[i]] + self.lamda) / (
                                    self.totalUnder50K + len(self.laplaceCorrect[i]) * self.lamda)
                    else:
                        under50Krate *= self.lamda / (self.totalUnder50K + len(self.laplaceCorrect[i]) * self.lamda)
                elif parent == const.AGE \
                    or parent == const.CAPITAL_GAIN or parent == const.CAPITAL_LOSS or parent == const.HOURS_PER_WEEK:
                    part_p = self.disc_under50K[parent].classification(float(data[parent]))
                    if data[i] in self.pre_cal_Above50K[i]:
                        under50Krate *= self.pre_cal_Under50K[i][data[i]][part_p]
                    else:
                        under50Krate *= self.lamda / (self.totalUnder50K + len(self.laplaceCorrect[i]) * self.lamda)
                else:
                    if data[i] in self.pre_cal_Under50K[i] and data[parent] in self.pre_cal_Under50K[i][data[i]]:
                        under50Krate *= self.pre_cal_Under50K[i][data[i]][data[parent]]
                    else:
                        under50Krate *= self.lamda / (self.totalUnder50K + len(self.laplaceCorrect[i]) * self.lamda)
            # print('under50Krate:',under50Krate)
        above50Krate = 1
        for i in range(0, const.IF_OVER_50K):
            if i == const.AGE \
                    or i == const.CAPITAL_GAIN or i == const.CAPITAL_LOSS or i == const.HOURS_PER_WEEK:
                # 查找父类
                parent = self.parent_label_list[i]
                # 无父类
                if parent == -1:
                    part_i = self.disc_above50K[i].classification(float(data[i]))
                    above50Krate *= (self.attrAbove50KDict[i][part_i] + self.lamda) / (
                        self.totalAbove50K + len(self.laplaceCorrect[i]) * self.lamda)
                # 有父类 父类是连续型
                elif parent == const.AGE \
                        or parent == const.CAPITAL_GAIN or parent == const.CAPITAL_LOSS or parent == const.HOURS_PER_WEEK:
                    part_i = self.disc_above50K[i].classification(float(data[i]))
                    part_p = self.disc_above50K[parent].classification(float(data[parent]))
                    above50Krate *= self.pre_cal_Above50K[i][part_i][part_p]
                # 有父类 父类是离散型
                else:
                    part_i = self.disc_above50K[i].classification(float(data[i]))
                    if data[parent] in self.pre_cal_Above50K[i][part_i]:
                        above50Krate *= self.pre_cal_Above50K[i][part_i][data[parent]]
                    else:
                        above50Krate *= self.lamda / (self.totalAbove50K + len(self.laplaceCorrect[i]) * self.lamda)
            else:
                # i 是离散型
                # 查找父类
                parent = self.parent_label_list[i]
                # 无父类
                if parent == -1:
                    if data[i] in self.attrAbove50KDict[i]:
                        above50Krate *= (self.attrAbove50KDict[i][data[i]] + self.lamda) / (
                                self.totalAbove50K + len(self.laplaceCorrect[i]) * self.lamda)
                    else:
                        above50Krate *= self.lamda / (self.totalAbove50K + len(self.laplaceCorrect[i]) * self.lamda)
                elif parent == const.AGE \
                        or parent == const.CAPITAL_GAIN or parent == const.CAPITAL_LOSS or parent == const.HOURS_PER_WEEK:
                    part_p = self.disc_above50K[parent].classification(float(data[parent]))
                    if data[i] in self.pre_cal_Above50K[i]:
                        above50Krate *= self.pre_cal_Above50K[i][data[i]][part_p]
                    else:
                        above50Krate *= self.lamda / (self.totalAbove50K + len(self.laplaceCorrect[i]) * self.lamda)
                else:
                    if data[i] in self.pre_cal_Above50K[i] and data[parent] in self.pre_cal_Above50K[i][data[i]]:
                        above50Krate *= self.pre_cal_Above50K[i][data[i]][data[parent]]
                    else:
                        above50Krate *= self.lamda / (self.totalAbove50K + len(self.laplaceCorrect[i]) * self.lamda)
            # print('above50Krate:', above50Krate)
        above50Krate *= self.priorAbove50K
        under50Krate *= self.priorUnder50K
        # print(above50Krate,' ',under50Krate,end=' ')
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