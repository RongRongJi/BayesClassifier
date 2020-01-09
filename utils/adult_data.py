# utf-8
import utils.const as const


class Adult:
    data = []

    # 数据初始化
    def dataInit(self,information):
        # 数据分割
        infoCut = information.strip('\n').split(", ")
        if len(infoCut) != const.IF_OVER_50K+1:
            return False, None
        self.data = infoCut
        # 数据清洗
        #   删除所有包含有?的信息
        for d in self.data:
            if d == "?":
                return False, None
        #   移除native-country属性
        self.data[const.NATIVE_COUNTRY] = 0
        self.data[const.FNLWGT] = 0
        return True, self.data

    # 信息打印
    def printInfo(self,data):
        print(data)
        print(data[const.IF_OVER_50K])
        for i in range(0,const.IF_OVER_50K+1):
            if i!=const.NATIVE_COUNTRY:
                print(data[i],end=' ')





