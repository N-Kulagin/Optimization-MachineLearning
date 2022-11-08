import math

def binary_search(x,array):
    l = 0
    r = len(array) - 1
    while r-l>=1:
        m = math.floor((l+r)/2)
        if array[m] == x:
            return m
        if array[m] < x:
            l = m+1
        if array[m] > x:
            r = m-1
    return r


def lower_bound(x,array):
    l = 0
    r = len(array) - 1
    while r - l != 1:
        m = math.floor((l + r) / 2)
        if array[m] < x:
            l = m
        else:
            r = m
    return array[r]



p = [-4,-1,-1,-1,-1,0,3,5,6,7,7,7,7,7,7,9,11,12,15]
print(binary_search(0,p))
print(lower_bound(-0.5,p))