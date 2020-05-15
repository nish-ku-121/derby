import heapq



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