import unittest
from derby.core.utils import kth_largest



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

if __name__ == '__main__':
	unittest.main()