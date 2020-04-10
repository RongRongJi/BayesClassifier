# utf-8
import numpy as np
import utils.const as const
from utils.const import is_continuous


class TanBayes2:
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
        self.__continus_pro__(is_above=False)
        self.__continus_pro__(is_above=True) # 连续变量预处理
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
        self.continus = {}
        self.continus[const.AGE] = {}
        self.continus[const.HOURS_PER_WEEK] = {}
        self.continus[const.CAPITAL_GAIN] = {}
        self.continus[const.CAPITAL_LOSS] = {}

    # 概率预计算
    def probability_precal(self):
        self.priorAbove50K = self.totalAbove50K / (self.totalAbove50K+self.totalUnder50K)
        self.priorUnder50K = self.totalUnder50K / (self.totalAbove50K+self.totalUnder50K)
        self.undercov = [[{}] * const.IF_OVER_50K for i in range(0, const.IF_OVER_50K)]        # 协方差缓存
        self.underdis_con = [[{}] * const.IF_OVER_50K for i in range(0, const.IF_OVER_50K)]    # 离散连续缓存
        self.underdis_con0 = [[{}] * const.IF_OVER_50K for i in range(0, const.IF_OVER_50K)]   # 离散零膨胀缓存
        self.undercon0 = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, -1, -1, {}, {}]              # 零膨胀缓存
        self.underdis_dis = [[{}] * const.IF_OVER_50K for i in range(0, const.IF_OVER_50K)]    # 离散离散缓存
        self.abovecov = [[{}] * const.IF_OVER_50K for i in range(0, const.IF_OVER_50K)]  # 协方差缓存
        self.abovedis_con = [[{}] * const.IF_OVER_50K for i in range(0, const.IF_OVER_50K)]  # 离散连续缓存
        self.abovedis_con0 = [[{}] * const.IF_OVER_50K for i in range(0, const.IF_OVER_50K)]  # 离散零膨胀缓存
        self.abovecon0 = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, -1, -1, {}, {}]              # 零膨胀缓存
        self.abovedis_dis = [[{}] * const.IF_OVER_50K for i in range(0, const.IF_OVER_50K)]  # 离散离散缓存

    # 互信息计算
    # I = Σp(xi,xj)log( p(xi,xj) / p(xi)p(xj) )
    # I = H(x) - H(x|y)
    # 离散 → 离散 \ 连续
    # 连续 → 连续
    # 连续型父节点不能有离散型子节点
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
                        information_ij = self.discrete_discrete_info(i, j, False)
                        information_ij += self.discrete_discrete_info(i, j, True)
                        pheromone_arr[i][j] = information_ij
                        pheromone_arr[j][i] = information_ij
                    elif j == const.AGE or j == const.HOURS_PER_WEEK:   # 子节点j是连续型 服从高斯分布
                        information_ij = self.discrete_continus_info(i, j, False)
                        information_ij += self.discrete_continus_info(i, j, True)
                        pheromone_arr[i][j] = information_ij
                    else:                                   # 子节点是零膨胀连续型
                        information_ij = self.discrete_continus_zip_info(i, j, False)
                        information_ij += self.discrete_continus_zip_info(i, j, True)
                        pheromone_arr[i][j] = information_ij
                elif i == const.AGE or i == const.HOURS_PER_WEEK:   # 父节点i是连续型 服从高斯分布
                    if j == const.AGE or j == const.HOURS_PER_WEEK: # 子节点j是连续型 服从高斯分布
                        if pheromone_arr[i][j] != -1:
                            continue
                        information_ij = self.continus_continus_info(i, j, False)
                        information_ij += self.continus_continus_info(i, j, True)
                        pheromone_arr[i][j] = information_ij
                        pheromone_arr[j][i] = information_ij
        return pheromone_arr

    # 计算互信息 离散→离散
    def discrete_discrete_info(self, i, j, is_above):
        attrDict, totalNum = self.dictNtotal(is_above)
        information_ij = 0
        temp_dict, temp_x, temp_y = {}, {}, {}
        for data in attrDict:
            self.add_new_item(temp_dict, (data[i], data[j]), float(data[const.FNLWGT]))
            self.add_new_item(temp_x, data[i], float(data[const.FNLWGT]))
            self.add_new_item(temp_y, data[j], float(data[const.FNLWGT]))
        for ki, vi in temp_x.items():
            for kj, vj in temp_y.items():
                if (ki, kj) in temp_dict:
                    tmp1, tmp2 = vi / totalNum, vj / totalNum
                    p = temp_dict[(ki,kj)] / totalNum
                    information_ij += p * np.log(p / (tmp1 * tmp2))
        return information_ij

    # 计算互信息 离散→连续
    def discrete_continus_info(self, i, j, is_above):
        attrDict, totalNum = self.dictNtotal(is_above)
        temp = {}
        for data in attrDict:
            if data[i] not in temp:
                temp[data[i]] = []
            temp[data[i]].append((float(data[j]), float(data[const.FNLWGT])))
        H_x, H_xy = 0, 0
        for k, v in temp.items():
            aver, variance, total = self.averNvairance(v)
            tmp_xy = 0
            cnki_x, cnki_xy = {}, {}
            for num in v:
                if num[0] not in cnki_x:
                    px = self.Gauss_pro(num[0], self.continus[j]['aver'+str(is_above)],
                                        self.continus[j]['var'+str(is_above)])
                    H_x -= px * np.log(px)
                    cnki_x[num[0]] = 1
                if num[0] not in cnki_xy:
                    pxy = self.Gauss_pro(num[0], aver, variance)
                    tmp_xy -= pxy * np.log(pxy)
                    cnki_xy[num[0]] = 1
            tmp_xy *= total
            H_xy += tmp_xy
        H_xy /= totalNum
        return H_x - H_xy

    # 计算互信息 离散→连续（零膨胀）
    def discrete_continus_zip_info(self, i, j, is_above):
        attrDict, totalNum = self.dictNtotal(is_above)
        temp, con0 = {}, {}
        for data in attrDict:
            if data[i] not in temp:
                temp[data[i]] = []
                con0[data[i]] = 0
            if float(data[j]) == -1:
                con0[data[i]] += float(data[const.FNLWGT])
            else:
                temp[data[i]].append((float(data[j]), float(data[const.FNLWGT])))
        H_x, H_xy = 0, 0
        for k, v in temp.items():
            aver, variance, total = self.averNvairance(v)
            total += con0[k]
            if con0[k] == 0:
                tmp_xy = 0
            else:
                tmp_xy = -con0[k] / total * np.log(con0[k] / total)
                H_x -= con0[k] / totalNum * np.log(con0[k] / totalNum)
            cnki_x, cnki_xy = {}, {}
            for num in v:
                if num[0] not in cnki_x:
                    px = self.Gauss_pro(num[0], self.continus[j]['aver' + str(is_above)],
                                        self.continus[j]['var' + str(is_above)])
                    H_x -= px * np.log(px)
                    cnki_x[num[0]] = 1
                if num[0] not in cnki_xy:
                    pxy = self.Gauss_pro(num[0], aver, variance)
                    tmp_xy -= pxy * np.log(pxy)
                    cnki_xy[num[0]] = 1
            tmp_xy *= total
            H_xy += tmp_xy
        H_xy /= totalNum
        return H_x - H_xy

    # 计算互信息 连续→连续
    def continus_continus_info(self, i, j, is_above):
        attrDict, totalNum = self.dictNtotal(is_above)
        varX, varY, eX, eY, p = self.Cov_roll(i, j, attrDict, totalNum, is_above)
        return -0.5*np.log(1-p*p)

    # 辅助函数： 计算协方差率
    def Cov_roll(self, i, j, attrDict, totalNum, is_above):
        if is_above:
            cov = self.abovecov
        else:
            cov = self.undercov
        if "cov" not in cov[i][j]:
            varX, varY, eX, eY, Cov_xy = 0, 0, 0, 0, 0
            for data in attrDict:
                eX, eY = eX + float(data[i]) * float(data[const.FNLWGT]), eY + float(data[j]) * float(data[const.FNLWGT])
            eX, eY = eX / totalNum, eY / totalNum
            for data in attrDict:
                varX += float(data[const.FNLWGT])*np.power(float(data[i])-eX,2)
                varY += float(data[const.FNLWGT])*np.power(float(data[j])-eY,2)
            varX, varY = varX / totalNum, varY / totalNum
            for data in attrDict:
                Cov_xy += (float(data[i])-eX)*(float(data[j])-eY)*float(data[const.FNLWGT])
            Cov_xy /= totalNum
            Cov_xy /= np.sqrt(varX * varY)
            cov[i][j] = {"cov":Cov_xy,"varX":varX,"varY":varY,"eX":eX,"eY":eY}
            cov[j][i] = {"cov":Cov_xy,"varX":varX,"varY":varY,"eX":eX,"eY":eY}
            return varX, varY, eX, eY, Cov_xy
        else:
            return cov[i][j]["varX"], cov[i][j]["varY"], cov[i][j]["eX"], cov[i][j]["eY"], cov[i][j]["cov"]

    # 辅助函数： 定义字典与总量
    def dictNtotal(self, is_above):
        if not is_above:
            attrDict = self.attrUnder50KDict
            totalNum = self.totalUnder50K
        else:
            attrDict = self.attrAbove50KDict
            totalNum = self.totalAbove50K
        return attrDict, totalNum

    # 辅助函数： 字典增项
    def add_new_item(self, dict, item, content):
        if item not in dict:
            dict[item] = content
        else:
            dict[item] += content

    # 辅助函数： 计算平均数和方差
    def averNvairance(self, list):
        if len(list) == 0:
            return 0, 0, 0
        aver, variance, total = 0, 0, 0
        for num in list:
            aver += num[0] * num[1]
            total += num[1]
        aver /= total
        for num in list:
            variance += num[1]* np.power(num[0]-aver,2)
        variance /= total
        return aver, variance, total

    # 辅助函数： 高斯概率
    def Gauss_pro(self, num, aver, var):
        if var == 0:
            return 1
        return 1 / (np.sqrt(2 * np.pi * var)) * np.exp(-(np.power(num - aver, 2)) / (2 * var))

    # 连续型变量概率预计算
    def __continus_pro__(self, is_above):
        attrDict, totalNum = self.dictNtotal(is_above)
        for i in range(const.IF_OVER_50K):
            if i == const.AGE or i == const.HOURS_PER_WEEK:
                aver, variance, total = 0, 0, 0
                for data in attrDict:
                    aver += float(data[i]) * float(data[const.FNLWGT])
                    total += float(data[const.FNLWGT])
                aver /= total
                for data in attrDict:
                    variance += float(data[const.FNLWGT]) * np.power(float(data[i])-aver,2)
                variance /= total
                self.continus[i]['aver'+str(is_above)] = aver
                self.continus[i]['var'+str(is_above)] = variance
            elif i == const.CAPITAL_LOSS or i == const.CAPITAL_GAIN:
                aver, variance, total = 0, 0, 0
                for data in attrDict:
                    if float(data[i]) != -1:
                        aver += float(data[i]) * float(data[const.FNLWGT])
                        total += float(data[const.FNLWGT])
                aver /= total
                for data in attrDict:
                    if float(data[i]) != -1:
                        variance += float(data[const.FNLWGT]) * np.power(float(data[i]) - aver, 2)
                variance /= total
                self.continus[i]['aver' + str(is_above)] = aver
                self.continus[i]['var' + str(is_above)] = variance

    # 构建最大权生成树
    def create_mswt(self, pheromone_arr, tree_num):
        # mwst_matrix = [[-1] * const.IF_OVER_50K in range(const.IF_OVER_50K)]
        child_label_list =  []
        for i in range(0,const.IF_OVER_50K):
            child_label_list.append([])
        selected_node = [const.WORKCLASS]
        candidate_node = []
        for i in range(0,const.IF_OVER_50K):
            # 去除fnlwgt
            if i == const.FNLWGT or i == const.NATIVE_COUNTRY or i == const.WORKCLASS:
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

    # 判别辅助函数：
    def util_dis_con(self, i, parent, attrDict, data, is_above):
        if is_above:
            dis_con = self.abovedis_con[i][parent]
        else:
            dis_con = self.underdis_con[i][parent]
        if (data[i],data[parent]) not in dis_con:
            list = []
            for d in attrDict:
                if data[parent] == d[parent]:
                    list.append((float(d[i]), float(d[const.FNLWGT])))
            aver, variance, total = self.averNvairance(list)
            dis_con[(data[i],data[parent])] = {'aver':aver,'var':variance,'total':total}
            return aver, variance, total
        else:
            return dis_con[(data[i],data[parent])]['aver'],dis_con[(data[i],data[parent])]['var'],dis_con[(data[i],data[parent])]['total']

    # 判别辅助函数：
    def util_con0(self, i, attrDict, data, is_con0, is_above):
        if is_above:
            con0 = self.abovecon0
        else:
            con0 = self.undercon0
        if is_con0 and con0[i] == -1:
            tmp = 0
            for d in attrDict:
                if float(d[i]) == -1:
                    tmp += float(d[const.FNLWGT])
            con0[i] = tmp
            return tmp
        elif not is_con0 and data[i] not in con0[i]:
            tmp = 0
            for d in attrDict:
                if d[i] == data[i]:
                    tmp += float(d[const.FNLWGT])
            con0[i][data[i]] = tmp
            return tmp
        else:
            if is_con0:
                return con0[i]
            else:
                return con0[i][data[i]]

    # 判别辅助函数：
    def util_con0withParent(self, i, parent, attrDict, data, is_above):
        if is_above:
            dis_con0 = self.abovedis_con0[i][parent]
        else:
            dis_con0 = self.underdis_con0[i][parent]
        if (data[i],data[parent]) not in dis_con0:
            con0, list = 0, []
            for d in attrDict:
                if data[parent] == d[parent]:
                    if float(d[i]) == -1:
                        con0 += float(d[const.FNLWGT])
                    else:
                        list.append((float(d[i]), float(d[const.FNLWGT])))
            aver, variance, total = self.averNvairance(list)
            dis_con0[(data[i],data[parent])] = {'con0':con0,'aver':aver,'var':variance,'total':total}
            return con0, aver, variance, total
        else:
            return dis_con0[(data[i],data[parent])]['con0'],dis_con0[(data[i],data[parent])]['aver'],dis_con0[(data[i],data[parent])]['var'],dis_con0[(data[i],data[parent])]['total']

    # 判别辅助函数：
    def util_dis_dis(self, i, parent, attrDict, data, is_above):
        if is_above:
            dis_dis = self.abovedis_dis[i][parent]
        else:
            dis_dis = self.underdis_dis[i][parent]
        if (data[i],data[parent]) not in dis_dis:
            t, t_p = 0, 0
            for d in attrDict:
                if d[parent] == data[parent]:
                    t_p += float(d[const.FNLWGT])
                    if d[i] == data[i]:
                        t += float(d[const.FNLWGT])
            dis_dis[(data[i],data[parent])] = {"t":t, "t_p":t_p}
            return t, t_p
        else:
            return dis_dis[(data[i],data[parent])]["t"], dis_dis[(data[i],data[parent])]["t_p"]

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
            # print(i, " ", rate, end="||")
            if i == const.AGE or i == const.HOURS_PER_WEEK:             # 连续型 服从高斯分布
                parent = self.parent_label_list[i]                      # 查找父类
                if parent == -1:                                        # 无父类
                    pro = self.Gauss_pro(float(data[i]), self.continus[i]['aver' + str(is_above)],
                                        self.continus[i]['var' + str(is_above)])
                    rate *= pro
                elif parent == const.AGE or parent == const.HOURS_PER_WEEK:     # 父类 连续型 服从高斯分布
                    varX, varY, eX, eY, p = self.Cov_roll(i, parent, attrDict, totalNum, is_above)
                    tmp = 1-p*p
                    pro = 1/np.sqrt(2*np.pi*varX*varX*tmp)*np.exp(-np.power(float(data[i])-eX-(p*varX*(float(data[parent])-eY))/varY,2)/(2*varX*varX*tmp))
                    rate *= pro
                elif not is_continuous(parent):                         # 父类 离散型
                    aver, variance, total = self.util_dis_con(i,parent,attrDict,data, is_above)
                    pro = self.Gauss_pro(float(data[i]), aver, variance)
                    rate *= pro
            elif i == const.CAPITAL_LOSS or i == const.CAPITAL_GAIN:    # 连续型 零膨胀
                parent = self.parent_label_list[i]
                if parent == -1:                                        # 无父类
                    con0 = self.util_con0(i,attrDict,data,True, is_above)
                    if float(data[i]) == -1:
                        pro = con0 / totalNum
                        rate *= pro
                    else:
                        pro = self.Gauss_pro(float(data[i]), self.continus[i]['aver' + str(is_above)],
                                        self.continus[i]['var' + str(is_above)])
                        rate *= pro
                elif not is_continuous(parent):                         # 父类 离散型
                    con0, aver, variance, total = self.util_con0withParent(i, parent, attrDict, data, is_above)
                    if con0+total == 0:
                        rate *= self.lamda / totalNum
                    else:
                        if float(data[i]) == -1:
                            pro = con0 / (con0+total)
                            rate *= pro
                        else:
                            pro = self.Gauss_pro(float(data[i]), self.continus[i]['aver' + str(is_above)],
                                            self.continus[i]['var' + str(is_above)])
                            rate *= pro
            elif not is_continuous(i):                                  # 离散型
                parent = self.parent_label_list[i]
                if parent == -1:                                        # 无父类
                    tmp = self.util_con0(i, attrDict, data, False, is_above)
                    if tmp == 0:
                        pro = self.lamda / totalNum
                    else:
                        pro = tmp / totalNum
                    rate *= pro
                else:
                    t, t_p = self.util_dis_dis(i, parent, attrDict, data, is_above)
                    if t_p == 0:
                        rate *= self.lamda / totalNum
                    else:
                        pro = t / t_p
                        rate *= pro
        return rate
