import math

class SegmentTree():
    def __init__(self,array):
        powers_of_two = {*[2**x for x in range(100)]}
        if len(array) not in powers_of_two:
            raise Exception("A tree array must have length equal to some power of two")

        self.__length = len(array)
        self.__t = [0] * (2 * self.__length - 1)  # len(array) = 2^k
        for i in range(self.__length):
            self.__t[-1 - i] = array[-1 - i]

        for i in range(self.__length - 2, -1, -1):
            self.__t[i] = self.__t[2 * i + 1] + self.__t[2 * i + 2]

    def set(self, position, value):
        if position >= self.__length:
            raise IndexError("Index is out of range")
        x = position + self.__length - 1
        self.__t[x] = value

        while x != 0:
            x = math.floor((x - 1) / 2)
            self.__t[x] = self.__t[2 * x + 1] + self.__t[2 * x + 2]

    def sum(self,left,right):
        l = left + self.__length - 1
        r = right + self.__length - 2
        answer = 0

        if r >= len(self.__t) or l + 1 < self.__length:
            raise IndexError("Index is out of range")

        while l <= r:
            if l % 2 == 0:
                answer += self.__t[l]
            l = math.floor(l/ 2)

            if r % 2 == 1:
                answer += self.__t[r]
            r = math.floor(r/2) - 1

        return answer

    def __str__(self):
        return str(self.__t) + "\t<-- a tree"

    @property
    def len(self):
        return self.__length


a = [5,2,4,1,2,8,3,1]


tree = SegmentTree(a)
tree.set(7,5)
print(tree.sum(0,8))
print(tree)