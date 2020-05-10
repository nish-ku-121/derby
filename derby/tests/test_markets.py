import unittest
from derby.core.basic_structures import AuctionItem, Bid, AuctionResults
from derby.core.auctions import KthPriceAuction
from derby.core.pmf import PMF, AuctionItemPMF
from derby.core.ad_structures import Campaign
from derby.core.states import CampaignBidderState
from derby.core.markets import OneCampaignMarket



class TestMarkets(unittest.TestCase):

	def setUp(self):
		self.auction_items = [
						AuctionItem(name="male1", item_type={"male"}, owner=None),
						AuctionItem(name="female1", item_type={"female"}, owner=None)
		]
		self.campaigns = [
						Campaign(100, 1000, self.auction_items[0]),
						Campaign(100, 1000, self.auction_items[1])
		]
		self.first_price_auction = KthPriceAuction(1)
		self.second_price_auction = KthPriceAuction(2)

	def test_1(self):
		auction_items = self.auction_items
		campaigns = self.campaigns
		auction = self.first_price_auction
		pmf = AuctionItemPMF({
					auction_items[0] : 1,
					auction_items[1] : 0
		})
		bidder_states = [
						CampaignBidderState("bidder1", campaigns[0]),
						CampaignBidderState("bidder2", campaigns[0])
		]
		bids = [
				Bid("bidder1", auction_items[0], bid_per_item=1.0, total_limit=1.0),
				Bid("bidder2", auction_items[0], bid_per_item=2.0, total_limit=2.0)
		]
		market = OneCampaignMarket(auction, bidder_states, pmf, num_of_items_per_timestep=1)
		results = market.run_auction(bids, items_to_bids_mapping_func=auction.items_to_bids_by_item_type_submatch)

		bid_1_result = results.get_result(bids[0])
		bid_2_result = results.get_result(bids[1])
		unallocated = results.get_unallocated()

		self.assertEqual(bid_1_result, { })
		self.assertTrue(AuctionItem.is_copy(list(bid_2_result.keys())[0], auction_items[0]))
		self.assertEqual(list(bid_2_result.values())[0], 2.0)
		self.assertEqual(unallocated, { })

	def test_2(self):
		auction_items = self.auction_items
		campaigns = self.campaigns
		auction = self.first_price_auction
		pmf = AuctionItemPMF({
					auction_items[0] : 1,
					auction_items[1] : 0
		})
		bidder_states = [
						CampaignBidderState("bidder1", campaigns[0]),
						CampaignBidderState("bidder2", campaigns[0])
		]
		bids = [
				Bid("bidder1", auction_items[0], bid_per_item=1.0, total_limit=1.0),
				Bid("bidder2", auction_items[0], bid_per_item=2.0, total_limit=6.0)
		]
		market = OneCampaignMarket(auction, bidder_states, pmf, num_of_items_per_timestep=3)
		results = market.run_auction(bids, items_to_bids_mapping_func=auction.items_to_bids_by_item_type_submatch)

		bid_1_result = results.get_result(bids[0])
		bid_2_result = results.get_result(bids[1])
		unallocated = results.get_unallocated()

		self.assertEqual(bid_1_result, { })
		self.assertTrue(AuctionItem.is_copy(list(bid_2_result.keys())[0], auction_items[0]))
		self.assertTrue(AuctionItem.is_copy(list(bid_2_result.keys())[1], auction_items[0]))
		self.assertTrue(AuctionItem.is_copy(list(bid_2_result.keys())[2], auction_items[0]))
		self.assertEqual(list(bid_2_result.values())[0], 2.0)
		self.assertEqual(list(bid_2_result.values())[1], 2.0)
		self.assertEqual(list(bid_2_result.values())[2], 2.0)
		self.assertEqual(unallocated, { })

	def test_3(self):
		auction_items = self.auction_items
		campaigns = self.campaigns
		auction = self.first_price_auction
		pmf = AuctionItemPMF({
					auction_items[0] : 0,
					auction_items[1] : 1
		})
		bidder_states = [
						CampaignBidderState("bidder1", campaigns[0]),
						CampaignBidderState("bidder2", campaigns[0])
		]
		bids = [
				Bid("bidder1", auction_items[1], bid_per_item=1.0, total_limit=1.0),
				Bid("bidder2", auction_items[1], bid_per_item=2.0, total_limit=4.0)
		]
		market = OneCampaignMarket(auction, bidder_states, pmf, num_of_items_per_timestep=3)
		results = market.run_auction(bids, items_to_bids_mapping_func=auction.items_to_bids_by_item_type_submatch)

		bid_1_result = results.get_result(bids[0])
		bid_2_result = results.get_result(bids[1])
		unallocated = results.get_unallocated()

		self.assertTrue(AuctionItem.is_copy(list(bid_1_result.keys())[0], auction_items[1]))
		self.assertTrue(AuctionItem.is_copy(list(bid_2_result.keys())[0], auction_items[1]))
		self.assertTrue(AuctionItem.is_copy(list(bid_2_result.keys())[1], auction_items[1]))
		self.assertEqual(list(bid_1_result.values())[0], 1.0)
		self.assertEqual(list(bid_2_result.values())[0], 2.0)
		self.assertEqual(list(bid_2_result.values())[1], 2.0)
		self.assertEqual(unallocated, { })

	def test_4(self):
		auction_items = self.auction_items
		campaigns = self.campaigns
		auction = self.first_price_auction
		pmf = AuctionItemPMF({
					auction_items[0] : 1,
					auction_items[1] : 1
		})
		bidder_states = [
						CampaignBidderState("bidder1", campaigns[0]),
						CampaignBidderState("bidder2", campaigns[0])
		]
		bids = [
				Bid("bidder1", auction_items[0], bid_per_item=1.0, total_limit=1.0),
				Bid("bidder2", auction_items[1], bid_per_item=2.0, total_limit=2.0)
		]
		market = OneCampaignMarket(auction, bidder_states, pmf, num_of_items_per_timestep=100)
		results = market.run_auction(bids, items_to_bids_mapping_func=auction.items_to_bids_by_item_type_submatch)

		bid_1_result = results.get_result(bids[0])
		bid_2_result = results.get_result(bids[1])
		unallocated = results.get_unallocated()

		self.assertTrue(AuctionItem.is_copy(list(bid_1_result.keys())[0], auction_items[0]))
		self.assertTrue(AuctionItem.is_copy(list(bid_2_result.keys())[0], auction_items[1]))

		self.assertEqual(list(bid_1_result.values())[0], 1.0)
		self.assertEqual(list(bid_2_result.values())[0], 2.0)
		self.assertEqual(len(unallocated), 98)

		self.assertEqual(bidder_states[0].spend, 1.0)
		self.assertEqual(bidder_states[0].impressions, 1)
		self.assertEqual(bidder_states[0].timestep, 1)

		self.assertEqual(bidder_states[1].spend, 2.0)
		self.assertEqual(bidder_states[1].impressions, 1)
		self.assertEqual(bidder_states[1].timestep, 1)

	def test_5(self):
		auction_items = self.auction_items
		campaigns = self.campaigns
		auction = self.first_price_auction
		pmf = AuctionItemPMF({
					auction_items[0] : 1,
					auction_items[1] : 1
		})
		bidder_states = [
						CampaignBidderState("bidder1", campaigns[0]),
						CampaignBidderState("bidder2", campaigns[0])
		]
		market = OneCampaignMarket(auction, bidder_states, pmf, num_of_items_per_timestep=100)
		results = None
		for i in range(10):
			bids = [
				Bid("bidder1", auction_items[0], bid_per_item=1.0, total_limit=1.0),
				Bid("bidder2", auction_items[1], bid_per_item=2.0, total_limit=2.0)
			]
			results = market.run_auction(bids, items_to_bids_mapping_func=auction.items_to_bids_by_item_type_submatch)

		bid_1_result = results.get_result(bids[0])
		bid_2_result = results.get_result(bids[1])
		unallocated = results.get_unallocated()

		self.assertTrue(AuctionItem.is_copy(list(bid_1_result.keys())[0], auction_items[0]))
		self.assertTrue(AuctionItem.is_copy(list(bid_2_result.keys())[0], auction_items[1]))

		self.assertEqual(list(bid_1_result.values())[0], 1.0)
		self.assertEqual(list(bid_2_result.values())[0], 2.0)
		self.assertEqual(len(unallocated), 98)

		self.assertEqual(bidder_states[0].spend, 10.0)
		self.assertEqual(bidder_states[0].impressions, 10)
		self.assertEqual(bidder_states[0].timestep, 10)
		
		self.assertEqual(bidder_states[1].spend, 20.0)
		self.assertEqual(bidder_states[1].impressions, 10)
		self.assertEqual(bidder_states[1].timestep, 10)

	def test_6(self):
		auction_items = self.auction_items
		campaigns = self.campaigns
		auction = self.first_price_auction
		pmf = AuctionItemPMF({
					auction_items[0] : 1,
					auction_items[1] : 1
		})
		bidder_states = [
						CampaignBidderState("bidder1", campaigns[0]),
						CampaignBidderState("bidder2", campaigns[0])
		]
		market = OneCampaignMarket(auction, bidder_states, pmf, num_of_items_per_timestep=100)
		results = None
		horizon = 7
		for i in range(horizon):
			bids = [
				Bid("bidder1", auction_items[0], bid_per_item=2.0, total_limit=4.0),
				Bid("bidder2", auction_items[0], bid_per_item=1.5, total_limit=3.0),
				Bid("bidder1", auction_items[1], bid_per_item=1.5, total_limit=3.0),
				Bid("bidder2", auction_items[1], bid_per_item=3.0, total_limit=6.0)
			]
			results = market.run_auction(bids, items_to_bids_mapping_func=auction.items_to_bids_by_item_type_submatch)

		bid_1_result = results.get_result(bids[0])
		bid_2_result = results.get_result(bids[1])
		unallocated = results.get_unallocated()

		self.assertEqual(len(unallocated), 92)

		self.assertEqual(bidder_states[0].spend, 7.0*horizon)
		self.assertEqual(bidder_states[0].impressions, 4*horizon)
		self.assertEqual(bidder_states[0].timestep, 1*horizon)
		
		self.assertEqual(bidder_states[1].spend, 9.0*horizon)
		self.assertEqual(bidder_states[1].impressions, 4*horizon)
		self.assertEqual(bidder_states[1].timestep, 1*horizon)


if __name__ == '__main__':
	unittest.main()