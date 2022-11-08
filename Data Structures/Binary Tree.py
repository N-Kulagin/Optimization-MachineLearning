class Node:
    __Root = False
    __value = None
    __Left = None
    __Right = None
    __visited = False

    def __init__(self,val=None,root=False):
        self.__Root = root
        self.value = val

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self,val):
        self.__value = val

    @property
    def hasLeft(self):
        return self.__Left != None

    @property
    def hasRight(self):
        return self.__Right != None

    @property
    def hasValue(self):
        return self.__value != None

    @property
    def Right(self):
        return self.__Right

    @Right.setter
    def Right(self,value):
        self.__Right = Node(value)

    @property
    def Left(self):
        return self.__Left

    @Left.setter
    def Left(self,value):
        self.__Left = Node(value)

    def Obkhod(self):
        if self.__Root == False:
            raise ValueError('Нельзя начать обход с некорневого узла')

        def recursive(link):
            s = link
            if s.__visited == False:
                print(s.value)
                s.__visited = True
            while s.hasLeft and not s.Left.__visited:
                recursive(s.Left)
            while s.hasRight and not s.Right.__visited:
                recursive(s.Right)
        recursive(self)

    def AddValue(self,val):
        s = self
        while val < s.value:
            if s.hasLeft:
                s = s.Left
            else:
                s.Left = val
                break
        while val >= s.value:
            if s.hasRight:
                s = s.Right
            else:
                s.Right = val
                break


n = Node(10,True)
n.AddValue(7)
n.AddValue(3)
n.AddValue(8)
n.AddValue(9)
n.AddValue(11)
n.AddValue(14)
n.AddValue(13)
#n.AddValue(8.5)
n.Obkhod()
print(n.__dict__)

'''
n.Left = 7
n.Right = 3


s = n

while s.hasLeft:
    s = s.Left

print(s.value)
print(s.__dict__)
print(n.__dict__)
print(Node.__dict__)
n.Obkhod()
'''

