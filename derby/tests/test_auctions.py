import unittest
from derby.core.basic_structures import AuctionItemSpecification, AuctionItem, Bid, AuctionResults
from derby.core.auctions import KthPriceAuction



class TestFirstPriceAuction(unittest.TestCase):

	def setUp(self):
		self.auction_item_specs = [
						AuctionItemSpecification(name="horse1", item_type={"black", "horse"}),
						AuctionItemSpecification(name="horse2", item_type={"white", "horse"}),
						AuctionItemSpecification(name="horse3", item_type={"horse", "white"})
		]
		self.auction_items = [
						AuctionItem(self.auction_item_specs[0], owner=None),
						AuctionItem(self.auction_item_specs[1], owner=None),
						AuctionItem(self.auction_item_specs[2], owner=None)
		]
		self.first_price_auction = KthPriceAuction(1)		

	def test_first_price_1(self):
		auction = self.first_price_auction
		auction_item_specs = self.auction_item_specs
		auction_items = self.auction_items
		bids = [
				Bid("bidder1", auction_item_specs[0], bid_per_item=1.0, total_limit=1.0),
				Bid("bidder2", auction_item_specs[0], bid_per_item=2.0, total_limit=2.0)
		]

		results = auction.run_auction(bids, auction_items)
		bid_1_result = results.get_result(bids[0])
		bid_2_result = results.get_result(bids[1])
		unallocated = results.get_unallocated()
		
		self.assertEqual(bid_1_result, { })
		self.assertEqual(bid_2_result, { auction_items[0] : 2.0 })
		self.assertEqual(unallocated, { auction_items[1] : 0.0, 
									    auction_items[2] : 0.0 })

	def test_first_price_2(self):
		auction = self.first_price_auction
		auction_item_specs = self.auction_item_specs
		auction_items = self.auction_items
		bids = [
				Bid("bidder1", auction_item_specs[0], bid_per_item=1.0, total_limit=1.0),
				Bid("bidder2", auction_item_specs[1], bid_per_item=2.0, total_limit=2.0)
		]

		results = auction.run_auction(bids, auction_items)
		bid_1_result = results.get_result(bids[0])
		bid_2_result = results.get_result(bids[1])
		unallocated = results.get_unallocated()
		
		self.assertEqual(bid_1_result, { auction_items[0] : 1.0 })
		self.assertEqual(bid_2_result, { auction_items[1] : 2.0 })
		self.assertEqual(unallocated, { auction_items[2] : 0.0 })

	def test_first_price_3(self):
		auction = self.first_price_auction
		auction_item_specs = self.auction_item_specs
		auction_items = self.auction_items
		bids = [
				Bid("bidder1", auction_item_specs[0], bid_per_item=1.0, total_limit=1.0),
				Bid("bidder2", auction_item_specs[1], bid_per_item=2.0, total_limit=2.0),
				Bid("bidder3", auction_item_specs[2], bid_per_item=1.5, total_limit=1.5)
		]
		results = auction.run_auction(bids, auction_items)
		bid_1_result = results.get_result(bids[0])
		bid_2_result = results.get_result(bids[1])
		bid_3_result = results.get_result(bids[2])
		unallocated = results.get_unallocated()
		
		self.assertEqual(bid_1_result, { auction_items[0] : 1.0 })
		self.assertEqual(bid_2_result, { auction_items[1] : 2.0 })
		self.assertEqual(bid_3_result, { auction_items[2] : 1.5 })
		self.assertEqual(unallocated, { })

	def test_first_price_4(self):
		auction = self.first_price_auction
		auction_item_specs = self.auction_item_specs
		auction_items = self.auction_items
		bids = [
				Bid("bidder1", auction_item_specs[0], bid_per_item=1.0, total_limit=1.0),
				Bid("bidder1", auction_item_specs[0], bid_per_item=2.0, total_limit=2.0),
				Bid("bidder1", auction_item_specs[1], bid_per_item=1.5, total_limit=1.5)
		]
		results = auction.run_auction(bids, auction_items)
		bid_1_result = results.get_result(bids[0])
		bid_2_result = results.get_result(bids[1])
		bid_3_result = results.get_result(bids[2])
		unallocated = results.get_unallocated()
		
		self.assertEqual(bid_1_result, { })
		self.assertEqual(bid_2_result, { auction_items[0] : 2.0 })
		self.assertEqual(bid_3_result, { auction_items[1] : 1.5 })
		self.assertEqual(unallocated, { auction_items[2] : 0 })

	def test_first_price_4(self):
		auction = self.first_price_auction
		auction_items = self.auction_items
		temp_spec = AuctionItemSpecification(name=None, item_type={"white", "horse"})
		bids = [
				Bid("bidder1", temp_spec, bid_per_item=1.0, total_limit=2.0),
		]
		item_matches_bid_spec_func = lambda item, bid: AuctionItemSpecification.is_item_type_match(item.auction_item_spec, bid.auction_item_spec)
		results = auction.run_auction(bids, auction_items, item_matches_bid_spec_func)
		bid_1_result = results.get_result(bids[0])
		unallocated = results.get_unallocated()

		self.assertEqual(bid_1_result, { auction_items[1] : 1.0, 
										 auction_items[2] : 1.0 
										})
		self.assertEqual(unallocated, { auction_items[0] : 0 })

	def test_first_price_5(self):
		auction = self.first_price_auction
		auction_items = self.auction_items
		temp_spec = AuctionItemSpecification(name=None, item_type={"white", "horse"})
		bids = [
				Bid("bidder1", temp_spec, bid_per_item=1.5, total_limit=2.0),
		]
		item_matches_bid_spec_func = lambda item, bid: AuctionItemSpecification.is_item_type_match(item.auction_item_spec, bid.auction_item_spec)
		results = auction.run_auction(bids, auction_items, item_matches_bid_spec_func)
		bid_1_result = results.get_result(bids[0])
		unallocated = results.get_unallocated()

		self.assertEqual(bid_1_result, { auction_items[1] : 1.5
										})
		self.assertEqual(unallocated, { auction_items[0] : 0.0,
										auction_items[2] : 0.0 
		 								})


class TestSecondPriceAuction(unittest.TestCase):

	def setUp(self):
		self.auction_item_specs = [
						AuctionItemSpecification(name="horse1", item_type={"black", "horse"}),
						AuctionItemSpecification(name="horse2", item_type={"white", "horse"}),
						AuctionItemSpecification(name="horse3", item_type={"horse", "white"})
		]
		self.auction_items = [
						AuctionItem(self.auction_item_specs[0], owner=None),
						AuctionItem(self.auction_item_specs[1], owner=None),
						AuctionItem(self.auction_item_specs[2], owner=None)
		]
		self.second_price_auction = KthPriceAuction(2)

	def test_1(self):
		auction = self.second_price_auction
		auction_item_specs = self.auction_item_specs
		auction_items = self.auction_items
		bids = [
				Bid("bidder1", auction_item_specs[0], bid_per_item=1.0, total_limit=1.0),
				Bid("bidder2", auction_item_specs[0], bid_per_item=2.0, total_limit=2.0)
		]

		results = auction.run_auction(bids, auction_items)
		bid_1_result = results.get_result(bids[0])
		bid_2_result = results.get_result(bids[1])
		unallocated = results.get_unallocated()
		
		self.assertEqual(bid_1_result, { })
		self.assertEqual(bid_2_result, { auction_items[0] : 1.0 })
		self.assertEqual(unallocated, { auction_items[1] : 0.0, 
									    auction_items[2] : 0.0 })

	def test_2(self):
		auction = self.second_price_auction
		auction_item_specs = self.auction_item_specs
		auction_items = self.auction_items
		bids = [
				Bid("bidder1", auction_item_specs[0], bid_per_item=1.0, total_limit=1.0),
				Bid("bidder2", auction_item_specs[1], bid_per_item=2.0, total_limit=2.0)
		]

		results = auction.run_auction(bids, auction_items)
		bid_1_result = results.get_result(bids[0])
		bid_2_result = results.get_result(bids[1])
		unallocated = results.get_unallocated()
		
		self.assertEqual(bid_1_result, { auction_items[0] : 0.0 })
		self.assertEqual(bid_2_result, { auction_items[1] : 0.0 })
		self.assertEqual(unallocated, { auction_items[2] : 0.0 })

	def test_3(self):
		auction = self.second_price_auction
		auction_item_specs = self.auction_item_specs
		auction_items = self.auction_items
		bids = [
				Bid("bidder1", auction_item_specs[0], bid_per_item=1.0, total_limit=1.0),
				Bid("bidder1", auction_item_specs[0], bid_per_item=2.0, total_limit=2.0)
		]

		results = auction.run_auction(bids, auction_items)
		bid_1_result = results.get_result(bids[0])
		bid_2_result = results.get_result(bids[1])
		unallocated = results.get_unallocated()
		
		self.assertEqual(bid_1_result, { })
		self.assertEqual(bid_2_result, { auction_items[0] : 0.0 })
		self.assertEqual(unallocated, { auction_items[1] : 0.0, auction_items[2] : 0.0 })

	def test_4(self):
		auction = self.second_price_auction
		auction_item_specs = self.auction_item_specs
		auction_items = self.auction_items
		temp_spec = AuctionItemSpecification(name=None, item_type={"white", "horse"})
		bids = [
				Bid("bidder1", temp_spec, bid_per_item=1.0, total_limit=1.0),
				Bid("bidder2", temp_spec, bid_per_item=2.0, total_limit=2.0)
		]
		item_matches_bid_spec_func = lambda item, bid: AuctionItemSpecification.is_item_type_match(item.auction_item_spec, bid.auction_item_spec)
		results = auction.run_auction(bids, auction_items, item_matches_bid_spec_func)
		bid_1_result = results.get_result(bids[0])
		bid_2_result = results.get_result(bids[1])
		unallocated = results.get_unallocated()
		
		self.assertEqual(bid_1_result, { auction_items[2] : 0.0 })
		self.assertEqual(bid_2_result, { auction_items[1] : 1.0 })
		self.assertEqual(unallocated, { auction_items[0] : 0.0 })

	def test_5(self):
		auction = self.second_price_auction
		auction_item_specs = self.auction_item_specs
		auction_items = self.auction_items
		temp_spec = AuctionItemSpecification(name=None, item_type={"horse"})
		bids = [
				Bid("bidder1", temp_spec, bid_per_item=1.0, total_limit=3.0),
		]
		item_matches_bid_spec_func = lambda item, bid: AuctionItemSpecification.is_a_type_of(item.auction_item_spec, bid.auction_item_spec)
		results = auction.run_auction(bids, auction_items, item_matches_bid_spec_func)
		bid_1_result = results.get_result(bids[0])
		unallocated = results.get_unallocated()
		
		self.assertEqual(bid_1_result, { auction_items[0] : 0.0,
										 auction_items[1] : 0.0,
										 auction_items[2] : 0.0 })
		self.assertEqual(unallocated, { })

	def test_6(self):
		auction = self.second_price_auction
		auction_item_specs = self.auction_item_specs
		auction_items = self.auction_items
		temp_spec = AuctionItemSpecification(name=None, item_type={"white"})
		bids = [
				Bid("bidder1", temp_spec, bid_per_item=2.1, total_limit=3.0),
				Bid("bidder2", temp_spec, bid_per_item=2.0, total_limit=3.0),
				Bid("bidder3", auction_item_specs[0], bid_per_item=1.1, total_limit=3.0),
				Bid("bidder4", auction_item_specs[0], bid_per_item=1.2, total_limit=3.0),
		]
		item_matches_bid_spec_func = lambda item, bid: AuctionItemSpecification.is_a_type_of(item.auction_item_spec, bid.auction_item_spec)
		results = auction.run_auction(bids, auction_items, item_matches_bid_spec_func)
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