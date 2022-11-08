import random


def quick_sort(l,r,array):
    if r-l<=1:
        return

    x = array[random.randint(l,r-1)]
    l0,l1,l2 = 0,0,0
    for i in range(l,r):
        if array[i] > x:
            l2 += 1
            continue
        if array[i] < x and l1 >= 1:
            array[i], array[l+l0] = array[l+l0], array[i]
            l0 += 1
            l1 -= 1
        array[i], array[l+l0+l1] = array[l+l0+l1], array[i]

        if array[l+l0+l1] < x:
            l0 += 1
        else:
            l1 += 1

    quick_sort(l,l+l0,array)
    quick_sort(l+l0+l1,r,array)





p = [5,2,4,2,3,8,7,5,1,0,-1,3,6,-10]
quick_sort(0,len(p),p)
print(p)
