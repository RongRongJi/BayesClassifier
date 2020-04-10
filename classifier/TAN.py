# utf-8
import numpy as np
import utils.const as const
from utils.const import is_continuous
from utils.discretization import Discretization


class TanBayes:
    adultSet = []
    lamda = 1
    attrUnder50KDict = []
    attrAbove50KDict = []
    totalUnder50K = 0
    totalAbove50K = 0

    def __init__(self, dataSet, lamda):
        self.adultSet = dataSet
        self.lamda = lamda
        self.__calAttr__() # 数据处理
        self.probability_precal() # 概率预计算
        self.pheromone_arr = self.cal_pheromone()
        # print("互信息计算完毕")
        self.create_mswt(self.pheromone_arr,0)

    # 数据处理
    def __calAttr__(self):
        for arr in self.adultSet:
            if arr[const.IF_OVER_50K] == '<=50K':
                self.attrUnder50KDict.append(arr)
                self.totalUnder50K += float(arr[const.FNLWGT])
            elif arr[const.IF_OVER_50K] == '>50K':
                self.attrAbove50KDict.append(arr)
                self.totalAbove50K += float(arr[const.FNLWGT])
        continuous = [const.AGE, const.CAPITAL_GAIN, const.CAPITAL_LOSS, const.HOURS_PER_WEEK]
        self.disc_under50K = dict()
        self.disc_above50K = dict()
        group = [10, 30, 30, 15]
        threshold = [0, 0, 0, 0]
        iso = [[], [-1], [-1], []]
        index = 0
        for con in continuous:
            under_sourceData, above_sourceData = {}, {}
            for arr in self.adultSet:
                if arr[const.IF_OVER_50K] == '<=50K':
                    if arr[con] not in under_sourceData:
                        under_sourceData[arr[con]] = float(arr[const.FNLWGT])
                    else:
                        under_sourceData[arr[con]] += float(arr[const.FNLWGT])
                else:
                    if arr[con] not in above_sourceData:
                        above_sourceData[arr[con]] = float(arr[const.FNLWGT])
                    else:
                        above_sourceData[arr[con]] += float(arr[const.FNLWGT])
            dis = Discretization(group=group[index], threshold=threshold[index], isolated=iso[index])
            data = dis.loadData(under_sourceData)
            dis.train(data)
            dis.set_data()
            self.disc_under50K[con] = dis
            dis = Discretization(group=group[index], threshold=threshold[index])
            data = dis.loadData(above_sourceData)
            dis.train(data)
            dis.set_data()
            self.disc_above50K[con] = dis
            index += 1
            # print(con, "训练完成")
        # print("离散化处理完毕")

    # 概率预计算
    def probability_precal(self):
        self.priorAbove50K = self.totalAbove50K / (self.totalAbove50K+self.totalUnder50K)
        self.priorUnder50K = self.totalUnder50K / (self.totalAbove50K+self.totalUnder50K)
        self.underdis = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]              # 零膨胀缓存
        self.underdis_dis = [[{}] * const.IF_OVER_50K for i in range(0, const.IF_OVER_50K)]    # 离散离散缓存
        self.abovedis = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]              # 零膨胀缓存
        self.abovedis_dis = [[{}] * const.IF_OVER_50K for i in range(0, const.IF_OVER_50K)]  # 离散离散缓存

    # 互信息计算
    def cal_pheromone(self):
        pheromone_arr = [[-1] * const.IF_OVER_50K for i in range(0, const.IF_OVER_50K)]
        for i in range(0,const.IF_OVER_50K):
            if i == const.FNLWGT:                           # FNLWGT 不计算在内
                continue
            for j in range(0, const.IF_OVER_50K):
                if i == j or j == const.FNLWGT:                       # FNLWGT 不计算在内
                    continue
                if not is_continuous(i):                    # 父节点i是离散型
                    if not is_continuous(j):                # 子节点j是离散型
                        if pheromone_arr[i][j] != -1:
                            continue
                        information_ij = self.discrete_discrete_info(i, j, False, 0)
                        information_ij += self.discrete_discrete_info(i, j, True, 0)
                        pheromone_arr[i][j] = information_ij
                        pheromone_arr[j][i] = information_ij
                    else:                                   # 子节点j是连续型
                        if pheromone_arr[i][j] != -1:
                            continue
                        information_ij = self.discrete_discrete_info(i, j, False, 1)
                        information_ij += self.discrete_discrete_info(i, j, True, 1)
                        pheromone_arr[i][j] = information_ij
                        pheromone_arr[j][i] = information_ij
                else:                                       # 父节点i是连续型
                    if not is_continuous(j):                # 子节点j是离散型
                        if pheromone_arr[i][j] != -1:
                            continue
                        information_ij = self.discrete_discrete_info(i, j, False, 2)
                        information_ij += self.discrete_discrete_info(i, j, True, 2)
                        pheromone_arr[i][j] = information_ij
                        pheromone_arr[j][i] = information_ij
                    else:                                   # 子节点j是连续型
                        if pheromone_arr[i][j] != -1:
                            continue
                        information_ij = self.discrete_discrete_info(i, j, False, 3)
                        information_ij += self.discrete_discrete_info(i, j, True, 3)
                        pheromone_arr[i][j] = information_ij
                        pheromone_arr[j][i] = information_ij
        return pheromone_arr

    # 计算互信息
    def discrete_discrete_info(self, i, j, is_above, flag):
        attrDict, totalNum = self.dictNtotal(is_above)
        information_ij = 0
        temp_dict, temp_x, temp_y = {}, {}, {}
        if is_above:
            dis = self.disc_above50K
        else:
            dis = self.disc_under50K
        if flag == 0:                               # 离散→离散
            for data in attrDict:
                self.add_new_item(temp_dict, (data[i], data[j]), float(data[const.FNLWGT]))
                self.add_new_item(temp_x, data[i], float(data[const.FNLWGT]))
                self.add_new_item(temp_y, data[j], float(data[const.FNLWGT]))
        elif flag == 1:                             # 离散→连续
            for data in attrDict:
                dj = dis[j].classification(float(data[j]))
                self.add_new_item(temp_dict, (data[i], dj), float(data[const.FNLWGT]))
                self.add_new_item(temp_x, data[i], float(data[const.FNLWGT]))
                self.add_new_item(temp_y, dj, float(data[const.FNLWGT]))
        elif flag == 2:                             # 连续→离散
            for data in attrDict:
                di = dis[i].classification(float(data[i]))
                self.add_new_item(temp_dict, (di, data[j]), float(data[const.FNLWGT]))
                self.add_new_item(temp_x, di, float(data[const.FNLWGT]))
                self.add_new_item(temp_y, data[j], float(data[const.FNLWGT]))
        elif flag == 3:                             # 连续→连续
            for data in attrDict:
                di = dis[i].classification(float(data[i]))
                dj = dis[j].classification(float(data[j]))
                self.add_new_item(temp_dict, (di, dj), float(data[const.FNLWGT]))
                self.add_new_item(temp_x, di, float(data[const.FNLWGT]))
                self.add_new_item(temp_y, dj, float(data[const.FNLWGT]))
        for ki, vi in temp_x.items():
            for kj, vj in temp_y.items():
                if (ki, kj) in temp_dict:
                    tmp1, tmp2 = vi / totalNum, vj / totalNum
                    p = temp_dict[(ki,kj)] / totalNum
                    information_ij += p * np.log(p / (tmp1 * tmp2))
        return information_ij

    # 辅助函数： 字典增项
    def add_new_item(self, dict, item, content):
        if item not in dict:
            dict[item] = content
        else:
            dict[item] += content

    # 辅助函数： 定义字典与总量
    def dictNtotal(self, is_above):
        if not is_above:
            attrDict = self.attrUnder50KDict
            totalNum = self.totalUnder50K
        else:
            attrDict = self.attrAbove50KDict
            totalNum = self.totalAbove50K
        return attrDict, totalNum

    # 构建最大权生成树
    def create_mswt(self, pheromone_arr, tree_num):
        # mwst_matrix = [[-1] * const.IF_OVER_50K in range(const.IF_OVER_50K)]
        child_label_list =  []
        for i in range(0,const.IF_OVER_50K):
            child_label_list.append([])
        selected_node = [0]
        candidate_node = []
        for i in range(0,const.IF_OVER_50K):
            # 去除fnlwgt
            if i == const.FNLWGT or i == const.NATIVE_COUNTRY or i == 0:
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
        # print("TAN树建立完毕", self.parent_label_list)
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

    # 判别
    def distinguish(self,data):
        under50Krate = self.util_distinguish(data, False)
        above50Krate = self.util_distinguish(data, True)
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

    # 概率计算
    def util_distinguish(self, data, is_above):
        rate = 1
        attrDict, totalNum = self.dictNtotal(is_above)
        for i in range(0, const.IF_OVER_50K):
            if i == const.FNLWGT or i == const.NATIVE_COUNTRY:
                continue
            parent = self.parent_label_list[i]
            if parent == -1:  # 无父类
                tmp = self.util_con0(i, attrDict, data, is_above)
                if tmp == 0:
                    pro = self.lamda / totalNum
                else:
                    pro = tmp / totalNum
                rate *= pro
            else:
                t, t_p = self.util_dis_dis(i, parent, attrDict, data, is_above)
                if t_p == 0 or t == 0:
                    rate *= self.lamda / totalNum
                else:
                    pro = t / t_p
                    rate *= pro
        return rate

    # 判别辅助函数：
    def util_dis_dis(self, i, p, attrDict, data, is_above):
        if is_above:
            dis_dis = self.abovedis_dis[i][p]
            discreation = self.disc_above50K
        else:
            dis_dis = self.underdis_dis[i][p]
            discreation = self.disc_under50K
        if is_continuous(i):
            if is_continuous(p):
                x, y = discreation[i].classification(float(data[i])), discreation[p].classification(float(data[p]))
                if (x,y) not in dis_dis:
                    t, t_p = 0, 0
                    for d in attrDict:
                        if discreation[p].classification(float(d[p])) == y:
                            t_p += float(d[const.FNLWGT])
                            if discreation[i].classification(float(d[i])) == x:
                                t += float(d[const.FNLWGT])
                    dis_dis[(x,y)] = {"t": t, "t_p": t_p}
                    return t, t_p
                else:
                    return dis_dis[(x,y)]["t"], dis_dis[(x,y)]["t_p"]
            else:
                x = discreation[i].classification(float(data[i]))
                if (x,data[p]) not in dis_dis:
                    t, t_p = 0, 0
                    for d in attrDict:
                        if d[p] == data[p]:
                            t_p += float(d[const.FNLWGT])
                            if discreation[i].classification(float(d[i])) == x:
                                t += float(d[const.FNLWGT])
                    dis_dis[(x,data[p])] = {"t": t, "t_p": t_p}
                    return t, t_p
                else:
                    return dis_dis[(x,data[p])]["t"], dis_dis[(x,data[p])]["t_p"]
        else:
            if is_continuous(p):
                y = discreation[p].classification(float(data[p]))
                if (data[i],y) not in dis_dis:
                    t, t_p = 0, 0
                    for d in attrDict:
                        if discreation[p].classification(float(d[p])) == y:
                            t_p += float(d[const.FNLWGT])
                            if d[i] == data[i]:
                                t += float(d[const.FNLWGT])
                    dis_dis[(data[i],y)] = {"t": t, "t_p": t_p}
                    return t, t_p
                else:
                    return dis_dis[(data[i],y)]["t"], dis_dis[(data[i],y)]["t_p"]
            else:
                tuple = (data[i],data[p])
                if tuple not in dis_dis:
                    t, t_p = 0, 0
                    for d in attrDict:
                        if d[p] == data[p]:
                            t_p += float(d[const.FNLWGT])
                            if d[i] == data[i]:
                                t += float(d[const.FNLWGT])
                    dis_dis[tuple] = {"t": t, "t_p": t_p}
                    return t, t_p
                else:
                    return dis_dis[tuple]["t"], dis_dis[tuple]["t_p"]

    # 判别辅助函数：
    def util_con0(self, i, attrDict, data, is_above):
        if is_above:
            dis = self.abovedis
            discreation = self.disc_above50K
        else:
            dis = self.underdis
            discreation = self.disc_under50K
        if is_continuous(i):   # 如果是连续型离散化
            num = discreation[i].classification(float(data[i]))
            if num not in dis[i]:
                tmp = 0
                for d in attrDict:
                    if discreation[i].classification(float(d[i])) == num:
                        tmp += float(d[const.FNLWGT])
                dis[i][num] = tmp
                return tmp
            else:
                return dis[i][num]
        else:       # 如果是离散变量
            if data[i] not in dis[i]:
                tmp = 0
                for d in attrDict:
                    if d[i] == data[i]:
                        tmp += float(d[const.FNLWGT])
                dis[i][data[i]] = tmp
                return tmp
            else:
                return dis[i][data[i]]
