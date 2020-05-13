import unittest
from derby.core.pmfs import PMF, AuctionItemPMF
from derby.core.basic_structures import AuctionItem



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


class TestAuctionItemPMF(unittest.TestCase):

	def test_1(self):
		item1 = AuctionItem(name="male1", item_type={"male"}, owner=None)
		item2 = AuctionItem(name="female1", item_type={"female"}, owner=None)
		pmf = AuctionItemPMF({
					item1 : 1,
					item2 : 0
			})
		result_1 = pmf.draw_n(1)
		result_2 = pmf.draw_n(3)
		self.assertTrue(AuctionItem.is_copy(result_1[0], item1))
		self.assertTrue(AuctionItem.is_copy(result_2[0], item1))
		self.assertTrue(AuctionItem.is_copy(result_2[1], item1))
		self.assertTrue(AuctionItem.is_copy(result_2[2], item1))

	def test_2(self):
		item1 = AuctionItem(name="male1", item_type={"male"}, owner=None)
		item2 = AuctionItem(name="female1", item_type={"female"}, owner=None)
		pmf = AuctionItemPMF({
					item1 : 1
			})
		pmf.add_items({item2 : 1})
		pmf.add_items({item1 : 0}, update_existing=True)
		result_1 = pmf.draw_n(1)
		result_2 = pmf.draw_n(3)
		self.assertTrue(AuctionItem.is_copy(result_1[0], item2))
		self.assertTrue(AuctionItem.is_copy(result_2[0], item2))
		self.assertTrue(AuctionItem.is_copy(result_2[1], item2))
		self.assertTrue(AuctionItem.is_copy(result_2[2], item2))


if __name__ == '__main__':
	unittest.main()