def merge(a,b):
    ret_arr = []
    i = 0
    j = 0

    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            ret_arr.append(a[i])
            i += 1
        else:
            ret_arr.append(b[j])
            j += 1
    ret_arr += a[i:] + b[j:]
    return ret_arr

def merge_sort(array):
    N = len(array) // 2

    left_arr = array[:N]
    right_arr = array[N:]
    if len(left_arr) > 1:
        left_arr = merge_sort(left_arr)
    if len(right_arr) > 1:
        right_arr = merge_sort(right_arr)

    return merge(left_arr,right_arr)


my_arr = [1,-2,0,6,2,8,6,9,0,25,7]
print(merge_sort([-1,-2,3,2,5]))
print(merge_sort(my_arr))