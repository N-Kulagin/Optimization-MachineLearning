import math


class Heap:
    def __init__(self,num):
        self.__array = [None] * num
        self.__element_count = 0
        if len(self.__array) == 0:
            raise Exception("Heap cannot be initialized with 0 elements")

    def get_min(self):
        return self.__array[0]

    def insert(self,x):
        if len(self.__array) < self.__element_count + 1:
            raise Exception("Cannot add more elements to heap. Heap is full.")
        self.__array[self.__element_count] = x
        self.__element_count += 1

        if self.__element_count == 1:
            return

        i = self.__element_count-1

        while i > 0 and self.__array[i] < self.__array[math.floor(  (i-1)/2    )]:
            self.__array[i],self.__array[math.floor((i-1)/2)] = self.__array[math.floor((i-1)/2)],self.__array[i] #swap
            i = math.floor((i-1)/2)

    def remove_min(self):
        if self.__element_count == 0:
            raise Exception("Cannot remove element from an empty heap")
        self.__array[0] = self.__array[self.__element_count-1]
        self.__array[self.__element_count-1] = None
        self.__element_count -= 1

        i = 0
        while 2*i + 1 < self.__element_count:
            j = 2*i+1
            if 2*i + 2 < self.__element_count and self.__array[2*i+2] < self.__array[j]:
                j = 2*i+2
            if self.__array[i] <= self.__array[j]:
                break
            else:
                self.__array[i], self.__array[j] = self.__array[j], self.__array[i]
                i = j


p = [7,3,5,6,9,10,4,3]
g = Heap(len(p))
print(p)


for i in range(len(p)):
    g.insert(p[i])
for j in range(len(p)):
    p[j] = g.get_min()
    g.remove_min()

print(p)