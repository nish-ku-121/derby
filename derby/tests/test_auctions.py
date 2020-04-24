import unittest
from derby.core.basic_structures import AuctionItem, Bid, AuctionResults
from derby.core.auctions import KthPriceAuction



class TestAuctions(unittest.TestCase):

	def setUp(self):
		self.auction_items = [
						AuctionItem(name="horse1", item_type={"black", "horse"}, owner=None),
						AuctionItem(name="horse2", item_type={"white", "horse"}, owner=None),
						AuctionItem(name="horse3", item_type={"horse", "white"}, owner=None)
		]
		self.first_price_auction = KthPriceAuction(1)
		self.second_price_auction = KthPriceAuction(2)
		

	def test_first_price_1(self):
		bids = [
				Bid("bidder1", self.auction_items[0], bid_per_item=1.0, total_limit=1.0),
				Bid("bidder2", self.auction_items[0], bid_per_item=2.0, total_limit=2.0)
		]
		results = self.first_price_auction.run_auction(bids, self.auction_items)
		bid_1_result = results.get_result(bids[0])
		bid_2_result = results.get_result(bids[1])
		unallocated = results.get_unallocated()
		
		self.assertEqual(bid_1_result, { })
		self.assertEqual(bid_2_result, { self.auction_items[0] : 2.0 })
		self.assertEqual(unallocated, { self.auction_items[1] : 0.0, 
									    self.auction_items[2] : 0.0 })

if __name__ == '__main__':
	unittest.main()