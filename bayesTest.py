# utf-8
from entry import Entry

entry = Entry()
# entry.read_file()
# entry.eva_naive_bayes(lamda=1)
entry.read_file(True)
# entry.eva_naive_bayes2(lamda=10000) # 高斯化
# entry.eva_tan_bayes(lamda=10000)  # 离散化
# entry.eva_tan_bayes2(lamda=10000) # 高斯化
# entry.network_bayes(lamda=10000)
# entry.eva_NBTree(lamda=10000)
# entry.eva_adaboost_nb(lamda=1, m=11)
entry.eva_adaboost_nbtree(lamda=1, m=8)