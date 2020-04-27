import unittest
from derby.core.basic_structures import AuctionItem, Bid, AuctionResults
from derby.core.auctions import KthPriceAuction



class TestFirstPriceAuction(unittest.TestCase):

	def setUp(self):
		self.auction_items = [
						AuctionItem(name="horse1", item_type={"black", "horse"}, owner=None),
						AuctionItem(name="horse2", item_type={"white", "horse"}, owner=None),
						AuctionItem(name="horse3", item_type={"horse", "white"}, owner=None)
		]
		self.first_price_auction = KthPriceAuction(1)		

	def test_first_price_1(self):
		auction_items = self.auction_items
		bids = [
				Bid("bidder1", auction_items[0], bid_per_item=1.0, total_limit=1.0),
				Bid("bidder2", auction_items[0], bid_per_item=2.0, total_limit=2.0)
		]

		results = self.first_price_auction.run_auction(bids, auction_items)
		bid_1_result = results.get_result(bids[0])
		bid_2_result = results.get_result(bids[1])
		unallocated = results.get_unallocated()
		
		self.assertEqual(bid_1_result, { })
		self.assertEqual(bid_2_result, { auction_items[0] : 2.0 })
		self.assertEqual(unallocated, { auction_items[1] : 0.0, 
									    auction_items[2] : 0.0 })

	def test_first_price_2(self):
		auction_items = self.auction_items
		bids = [
				Bid("bidder1", auction_items[0], bid_per_item=1.0, total_limit=1.0),
				Bid("bidder2", auction_items[1], bid_per_item=2.0, total_limit=2.0)
		]

		results = self.first_price_auction.run_auction(bids, auction_items)
		bid_1_result = results.get_result(bids[0])
		bid_2_result = results.get_result(bids[1])
		unallocated = results.get_unallocated()
		
		self.assertEqual(bid_1_result, { auction_items[0] : 1.0 })
		self.assertEqual(bid_2_result, { auction_items[1] : 2.0 })
		self.assertEqual(unallocated, { auction_items[2] : 0.0 })

	def test_first_price_3(self):
		auction_items = self.auction_items
		bids = [
				Bid("bidder1", self.auction_items[0], bid_per_item=1.0, total_limit=1.0),
				Bid("bidder2", self.auction_items[1], bid_per_item=2.0, total_limit=2.0),
				Bid("bidder3", self.auction_items[2], bid_per_item=1.5, total_limit=1.5)
		]
		results = self.first_price_auction.run_auction(bids, auction_items)
		bid_1_result = results.get_result(bids[0])
		bid_2_result = results.get_result(bids[1])
		bid_3_result = results.get_result(bids[2])
		unallocated = results.get_unallocated()
		
		self.assertEqual(bid_1_result, { auction_items[0] : 1.0 })
		self.assertEqual(bid_2_result, { auction_items[1] : 2.0 })
		self.assertEqual(bid_3_result, { auction_items[2] : 1.5 })
		self.assertEqual(unallocated, { })

	def test_first_price_4(self):
		auction_items = self.auction_items
		bids = [
				Bid("bidder1", self.auction_items[0], bid_per_item=1.0, total_limit=1.0),
				Bid("bidder1", self.auction_items[0], bid_per_item=2.0, total_limit=2.0),
				Bid("bidder1", self.auction_items[1], bid_per_item=1.5, total_limit=1.5)
		]
		results = self.first_price_auction.run_auction(bids, auction_items)
		bid_1_result = results.get_result(bids[0])
		bid_2_result = results.get_result(bids[1])
		bid_3_result = results.get_result(bids[2])
		unallocated = results.get_unallocated()
		
		self.assertEqual(bid_1_result, { })
		self.assertEqual(bid_2_result, { auction_items[0] : 2.0 })
		self.assertEqual(bid_3_result, { auction_items[1] : 1.5 })
		self.assertEqual(unallocated, { auction_items[2] : 0 })

	def test_first_price_4(self):
		auction_items = self.auction_items
		temp_item = AuctionItem(name=None, item_type={"white", "horse"}, owner=None)
		bids = [
				Bid("bidder1", temp_item, bid_per_item=1.0, total_limit=2.0),
		]
		results = self.first_price_auction.run_auction(bids, auction_items, self.first_price_auction.items_to_bids_by_item_type)
		bid_1_result = results.get_result(bids[0])
		unallocated = results.get_unallocated()

		self.assertEqual(bid_1_result, { auction_items[1] : 1.0, 
										 auction_items[2] : 1.0 
										})
		self.assertEqual(unallocated, { auction_items[0] : 0 })

	def test_first_price_5(self):
		auction_items = self.auction_items
		temp_item = AuctionItem(name=None, item_type={"white", "horse"}, owner=None)
		bids = [
				Bid("bidder1", temp_item, bid_per_item=1.5, total_limit=2.0),
		]
		results = self.first_price_auction.run_auction(bids, auction_items, self.first_price_auction.items_to_bids_by_item_type)
		bid_1_result = results.get_result(bids[0])
		unallocated = results.get_unallocated()

		self.assertEqual(bid_1_result, { auction_items[1] : 1.5
										})
		self.assertEqual(unallocated, { auction_items[0] : 0.0,
										auction_items[2] : 0.0 
		 								})


class TestSecondPriceAuction(unittest.TestCase):

	def setUp(self):
		self.auction_items = [
						AuctionItem(name="horse1", item_type={"black", "horse"}, owner=None),
						AuctionItem(name="horse2", item_type={"white", "horse"}, owner=None),
						AuctionItem(name="horse3", item_type={"horse", "white"}, owner=None)
		]
		self.auction = KthPriceAuction(2)

	def test_1(self):
		auction_items = self.auction_items
		bids = [
				Bid("bidder1", auction_items[0], bid_per_item=1.0, total_limit=1.0),
				Bid("bidder2", auction_items[0], bid_per_item=2.0, total_limit=2.0)
		]

		results = self.auction.run_auction(bids, auction_items)
		bid_1_result = results.get_result(bids[0])
		bid_2_result = results.get_result(bids[1])
		unallocated = results.get_unallocated()
		
		self.assertEqual(bid_1_result, { })
		self.assertEqual(bid_2_result, { auction_items[0] : 1.0 })
		self.assertEqual(unallocated, { auction_items[1] : 0.0, 
									    auction_items[2] : 0.0 })

	def test_2(self):
		auction_items = self.auction_items
		bids = [
				Bid("bidder1", auction_items[0], bid_per_item=1.0, total_limit=1.0),
				Bid("bidder2", auction_items[1], bid_per_item=2.0, total_limit=2.0)
		]

		results = self.auction.run_auction(bids, auction_items)
		bid_1_result = results.get_result(bids[0])
		bid_2_result = results.get_result(bids[1])
		unallocated = results.get_unallocated()
		
		self.assertEqual(bid_1_result, { auction_items[0] : 0.0 })
		self.assertEqual(bid_2_result, { auction_items[1] : 0.0 })
		self.assertEqual(unallocated, { auction_items[2] : 0.0 })

	def test_3(self):
		auction_items = self.auction_items
		bids = [
				Bid("bidder1", auction_items[0], bid_per_item=1.0, total_limit=1.0),
				Bid("bidder1", auction_items[0], bid_per_item=2.0, total_limit=2.0)
		]

		results = self.auction.run_auction(bids, auction_items)
		bid_1_result = results.get_result(bids[0])
		bid_2_result = results.get_result(bids[1])
		unallocated = results.get_unallocated()
		
		self.assertEqual(bid_1_result, { })
		self.assertEqual(bid_2_result, { auction_items[0] : 0.0 })
		self.assertEqual(unallocated, { auction_items[1] : 0.0, auction_items[2] : 0.0 })

	def test_4(self):
		auction_items = self.auction_items
		item_to_bids_mapping_func = self.auction.items_to_bids_by_item_type
		temp_item = AuctionItem(name=None, item_type={"white", "horse"}, owner=None)
		bids = [
				Bid("bidder1", temp_item, bid_per_item=1.0, total_limit=1.0),
				Bid("bidder2", temp_item, bid_per_item=2.0, total_limit=2.0)
		]

		results = self.auction.run_auction(bids, auction_items, item_to_bids_mapping_func)
		bid_1_result = results.get_result(bids[0])
		bid_2_result = results.get_result(bids[1])
		unallocated = results.get_unallocated()
		
		self.assertEqual(bid_1_result, { auction_items[2] : 0.0 })
		self.assertEqual(bid_2_result, { auction_items[1] : 1.0 })
		self.assertEqual(unallocated, { auction_items[0] : 0.0 })

	def test_4(self):
		auction_items = self.auction_items
		item_to_bids_mapping_func = self.auction.items_to_bids_by_item_type_submatch
		temp_item = AuctionItem(name=None, item_type={"horse"}, owner=None)
		bids = [
				Bid("bidder1", temp_item, bid_per_item=1.0, total_limit=3.0),
		]

		results = self.auction.run_auction(bids, auction_items, item_to_bids_mapping_func)
		bid_1_result = results.get_result(bids[0])
		unallocated = results.get_unallocated()
		
		self.assertEqual(bid_1_result, { auction_items[0] : 0.0,
										 auction_items[1] : 0.0,
										 auction_items[2] : 0.0 })
		self.assertEqual(unallocated, { })

	def test_5(self):
		auction_items = self.auction_items
		item_to_bids_mapping_func = self.auction.items_to_bids_by_item_type_submatch
		temp_item = AuctionItem(name=None, item_type={"white"}, owner=None)
		bids = [
				Bid("bidder1", temp_item, bid_per_item=2.1, total_limit=3.0),
				Bid("bidder2", temp_item, bid_per_item=2.0, total_limit=3.0),
				Bid("bidder3", auction_items[0], bid_per_item=1.1, total_limit=3.0),
				Bid("bidder4", auction_items[0], bid_per_item=1.2, total_limit=3.0),
		]

		results = self.auction.run_auction(bids, auction_items, item_to_bids_mapping_func)
		bid_1_result = results.get_result(bids[0])
		bid_2_result = results.get_result(bids[1])
		bid_3_result = results.get_result(bids[2])
		bid_4_result = results.get_result(bids[3])
		unallocated = results.get_unallocated()

		self.assertEqual(bid_1_result, { auction_items[1] : 2.0 })
		self.assertEqual(bid_2_result, { auction_items[2] : 0.0 })
		self.assertEqual(bid_3_result, { })
		self.assertEqual(bid_4_result, { auction_items[0] : 1.1 })
		self.assertEqual(unallocated, { })

		

	

if __name__ == '__main__':
	unittest.main()