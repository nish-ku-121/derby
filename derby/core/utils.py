import heapq
import numpy as np



def kth_largest(nums, k, default=None):
    if (nums == None) or (k < 0) or (len(nums) == 0) or (len(nums) < k):
        return default
    min_heap = []
    for num in nums:
    	if (len(min_heap) < k):
    		heapq.heappush(min_heap, num)
    	else:
    		if (num > min_heap[0]):
    			heapq.heapreplace(min_heap, num)
    return min_heap[0]

def flatten_2d(arr):
    flattened = []
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            flattened.append(arr[i][j])
    return flattened

def np_slice_i_of_dim_k(arr, i, k):
    '''
    takes an np array of shape [dim_0, dim_1,..., dim_k,..., dim_{n-1}]
    and returns an np array of shape [dim_0, dim_1,..., dim_{n-1}],
    where the ith slice of dim_k is selected.
    '''
    # slice (i.e. ':') symbol
    slice_symb = np.index_exp[:]
    # ':' symbol repeated k times for dims 0 to k-1
    slices_symbs_up_to_k = slice_symb * k
    slice_i = np.index_exp[i]
    return arr[slices_symbs_up_to_k + slice_i]