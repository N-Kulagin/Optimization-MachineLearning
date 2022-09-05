import math

class FenwickTree():
    def __init__(self,array):
        self.__tree = [0] * len(array)
        self.__length = len(array)

        for i in range(self.__length):
            for j in range(i & (i+1),i+1):
                self.__tree[i] += array[j]

    def sum(self,r):
        index = r - 1
        result = 0
        while index >= 0:
            result += self.__tree[index]
            index = (index & (index + 1)) - 1
        return result

    def insert(self,index,value):
        j = index

        while j < self.__length:
            self.__tree[j] += value
            j = j | (j+1)

    @property
    def len(self):
        return self.__length





a = [1,6,2,4,6,9,2,35,7,4,6,0,1,9]
t = FenwickTree(a)
print(t.len)
print(t.sum(12))
print(sum(a))
