
class DisjoinedSet():

    def __init__(self, size):
        self.elements = [None] * size
        self.rank = [0] * size

    def addSet(self,i):
        if self.elements[i] == None:
            self.elements[i] = i
            self.rank[i] = 0

    def get(self,x):
        if x != self.elements[x]:
            x = self.get(self.elements[x])
        return self.elements[x]

    def Union(self,x,y):
        i,j = self.get(x),self.get(y)

        if i == j:
            return

        if self.rank[i] > self.rank[j]:
            self.elements[j] = i
        else:
            self.elements[i] = j
            if self.rank[i] == self.rank[j]:
                self.rank[j] += 1


p = [1,2,3,4,5,6]
t = DisjoinedSet(len(p)+1)

for el in p:
    t.addSet(el)

print(t.get(4))
t.Union(4,2)
t.Union(2,5)
t.Union(1,3)
print(t.get(5))