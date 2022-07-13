class Node:
    __Value = None
    __Left = None
    __Right = None
    __Parent = None

    def __init__(self,val,parent=None):
        self.Value = val
        self.Parent = parent

    def hasRight(self):
        return self.Right != None

    def hasLeft(self):
        return self.Left != None

    @property
    def Right(self):
        return self.__Right

    @Right.setter
    def Right(self,val):
        self.__Right = Node(val)

    @property
    def Left(self):
        return self.__Left

    @Left.setter
    def Left(self,val):
        self.__Left = Node(val)

    @property
    def Value(self):
        return self.__value

    @Value.setter
    def Value(self,val):
        self.__value = val

    @property
    def Parent(self):
        return self.__Parent

    @Parent.setter
    def Parent(self,parent):
        self.__Parent = parent

    def Add(self,val):
        def recursive(link,v):
            if v < link.Value:
                if link.Left == None:
                    link.Left = v
                    link.Left.Parent = link
                else:
                    recursive(link.Left,v)

            else:
                if link.Right == None:
                    link.Right = v
                    link.Right.Parent = link
                else:
                    recursive(link.Right,v)
        recursive(self,val)

    def Find(self,val):
        s = self
        while s.Value != val:
            if s.Left == None and s.Right == None and s.Value != val:
                return False
            if val < s.Value:
                s = s.Left
            elif val > s.Value:
                s = s.Right
        return True

    def Print(self):  # по сути - полный обход дерева

        def recursive(link):
            if link.Left != None:
                recursive(link.Left)
            print(link.Value)  # Принт, следуя вторым, позволяет выводить элементы по возрастанию
            if link.Right != None:
                recursive(link.Right)
        recursive(self)


tree = Node(10)


tree_list = [7,3,8,9,11,14,8.5]
for el in tree_list:
    tree.Add(el)

for el in tree_list:
    print(tree.Find(el))

print(tree.Find(17))

print()
tree.Print()