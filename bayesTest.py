# utf-8
from entry import Entry

entry = Entry()
entry.read_file()
# entry.eva_naive_bayes(lamda=1)
entry.eva_tan_bayes(lamda=1)