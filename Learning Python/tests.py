class Square:
    def __init__(self, s):
        self.__side = s
        self.__area = None

    @property
    def side(self):
        return self.__side

    @side.setter
    def side(self, s):
        if self.__side != s:
            self.__side = s
            self.__area = None

    @property
    def area(self):
        if self.__area == None:
            self.__area = self.__side ** 2
        return self.__area


a = Square(5)
print(a.area)
print(a.side)
a.side = 7
print(a.area)
print(a.side)