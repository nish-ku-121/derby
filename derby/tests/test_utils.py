import unittest
import numpy as np
from derby.core.utils import kth_largest, flatten_2d, np_slice_i_of_dim_k



class TestUtils(unittest.TestCase):

	def test_kth_largest_1(self):
		self.assertEqual(kth_largest(None, 1), None)

	def test_kth_largest_2(self):
		self.assertEqual(kth_largest([2], 1), 2)

	def test_kth_largest_3(self):
		self.assertEqual(kth_largest([1.4, 1.1], 1), 1.4)

	def test_kth_largest_4(self):
		self.assertEqual(kth_largest([1.4, 1.1], 2), 1.1)

	def test_kth_largest_5(self):
		self.assertEqual(kth_largest([1.4, 1.1], 3), None)

	def test_kth_largest_6(self):
		self.assertEqual(kth_largest([1.4, 1.1], 3, 0.0), 0.0)

	def test_kth_largest_7(self):
		self.assertEqual(kth_largest([1.4, 1.1, 0.1], 3, 0.0), 0.1)

	def test_np_slice_i_of_dim_k_1(self):
		arr = np.array(
			[[[[ 10., 100.,   1.,   0.,   0.,   0.],
         	   [ 10., 100.,   2.,   0.,   0.,   0.],
         	   [ 10., 100.,   3.,   0.,   0.,   0.]],
         	  [[ 10., 100.,   1.,   5.,   5.,   5.],
         	   [ 10., 100.,   2.,   5.,   5.,   5.],
         	   [ 10., 100.,   3.,   5.,   5.,   5.]]]]
			)

		self.assertTrue((np_slice_i_of_dim_k(arr, 0, 0) == arr[0]).all())
		self.assertTrue((np_slice_i_of_dim_k(arr, 0, 1) == arr[:,0]).all())
		self.assertTrue((np_slice_i_of_dim_k(arr, 0, 2) == arr[:,:,0]).all())


if __name__ == '__main__':
	unittest.main()