# utf-8
from entry import Entry

entry = Entry()
# entry.eva_naive_bayes_gau(lamda=1,flag=False)
# entry.read_file(True)
# entry.eva_naive_bayes_zero_inflation(lamda=10000, flag=True) # 高斯化
entry.eva_tan_bayes_discretization(lamda=10000, flag=True)  # 离散化
# entry.eva_tan_bayes_zero_inflation(lamda=10000, flag=True) # 高斯化
# entry.network_bayes(lamda=10000, flag=True, network=[1,-1,-1,6,3,1,-1,1,1,3,-1,-1,1,-1])
# entry.eva_NBTree(lamda=10000, flag=True)
# entry.eva_adaboost_nb(lamda=1, m=8)
# entry.eva_adaboost_nbtree(lamda=1, m=8)