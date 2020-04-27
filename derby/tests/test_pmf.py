import unittest
from derby.core.pmf import PMF



class TestPMF(unittest.TestCase):

	def test_1(self):
		pmf = PMF({
					frozenset({"foo"}) : 1,
					frozenset({"bar"}) : 0
			})
		result_1 = pmf.draw_n(1)
		result_2 = pmf.draw_n(3, replace=True)
		self.assertEqual(result_1, {"foo"})
		self.assertEqual(result_2[0], {"foo"})
		self.assertEqual(result_2[1], {"foo"})
		self.assertEqual(result_2[2], {"foo"})

	def test_2(self):
		pmf = PMF({
					frozenset({"foo"}) : 1
			})
		pmf.add_items({frozenset({"bar"}) : 1})
		pmf.add_items({frozenset({"bar"}) : 0}, update_existing=True)
		result_1 = pmf.draw_n(1)
		result_2 = pmf.draw_n(3, replace=True)
		self.assertEqual(result_1, {"foo"})
		self.assertEqual(result_2[0], {"foo"})
		self.assertEqual(result_2[1], {"foo"})
		self.assertEqual(result_2[2], {"foo"})

	def test_3(self):
		pmf = PMF({
					frozenset({"foo"}) : 1
			})
		pmf.add_items({frozenset({"bar"}) : 1})
		pmf.add_items({frozenset({"bar"}) : 0}, update_existing=True)
		pmf.delete_item(frozenset({"bar"}))
		result_1 = pmf.draw_n(1)
		result_2 = pmf.draw_n(3, replace=True)
		self.assertEqual(result_1, {"foo"})
		self.assertEqual(result_2[0], {"foo"})
		self.assertEqual(result_2[1], {"foo"})
		self.assertEqual(result_2[2], {"foo"})


if __name__ == '__main__':
	unittest.main()