import unittest
from derby.core.basic_structures import AuctionItemSpecification, AuctionItem, Bid, AuctionResults
from derby.core.auctions import KthPriceAuction
from derby.core.pmfs import PMF
from derby.core.ad_structures import Campaign
from derby.core.states import CampaignBidderState
from derby.core.markets import OneCampaignMarket



class TestMarkets(unittest.TestCase):

    def setUp(self):
        self.auction_item_specs = [
                        AuctionItemSpecification(name="male", item_type={"male"}),
                        AuctionItemSpecification(name="female", item_type={"female"})
        ]
        self.campaigns = [
                        Campaign(100, 1000, self.auction_item_specs[0]),
                        Campaign(100, 1000, self.auction_item_specs[1])
        ]
        self.first_price_auction = KthPriceAuction(1)
        self.second_price_auction = KthPriceAuction(2)

    def test_1(self):
        auction_item_specs = self.auction_item_specs
        campaigns = self.campaigns
        auction = self.first_price_auction
        bidder_states = [
                        CampaignBidderState("bidder1", campaigns[0]),
                        CampaignBidderState("bidder2", campaigns[0])
        ]
        auction_item_spec_pmf = PMF({
                    auction_item_specs[0] : 1,
                    auction_item_specs[1] : 0
        })
        bids = [
                Bid("bidder1", auction_item_specs[0], bid_per_item=1.0, total_limit=1.0),
                Bid("bidder2", auction_item_specs[0], bid_per_item=2.0, total_limit=2.0)
        ]
        item_matches_bid_spec_func = lambda item, bid: AuctionItemSpecification.is_a_type_of(item.auction_item_spec, bid.auction_item_spec)
        market = OneCampaignMarket(auction, bidder_states, auction_item_spec_pmf, num_of_items_per_timestep=1)
        results = market.run_auction(bids, item_matches_bid_spec_func)

        bid_1_result = results.get_result(bids[0])
        bid_2_result = results.get_result(bids[1])
        unallocated = results.get_unallocated()

        self.assertEqual(bid_1_result, { })
        self.assertEqual(list(bid_2_result.keys())[0].auction_item_spec.item_type, auction_item_specs[0].item_type)
        self.assertEqual(list(bid_2_result.values())[0], 2.0)
        self.assertEqual(unallocated, { })

    def test_2(self):
        auction_item_specs = self.auction_item_specs
        campaigns = self.campaigns
        auction = self.first_price_auction
        bidder_states = [
                        CampaignBidderState("bidder1", campaigns[0]),
                        CampaignBidderState("bidder2", campaigns[0])
        ]
        auction_item_spec_pmf = PMF({
                    auction_item_specs[0] : 1,
                    auction_item_specs[1] : 0
        })
        bids = [
                Bid("bidder1", auction_item_specs[0], bid_per_item=1.0, total_limit=1.0),
                Bid("bidder2", auction_item_specs[0], bid_per_item=2.0, total_limit=6.0)
        ]
        item_matches_bid_spec_func = lambda item, bid: AuctionItemSpecification.is_a_type_of(item.auction_item_spec, bid.auction_item_spec)
        market = OneCampaignMarket(auction, bidder_states, auction_item_spec_pmf, num_of_items_per_timestep=3)
        results = market.run_auction(bids, item_matches_bid_spec_func)

        bid_1_result = results.get_result(bids[0])
        bid_2_result = results.get_result(bids[1])
        unallocated = results.get_unallocated()

        self.assertEqual(bid_1_result, { })
        self.assertEqual(list(bid_2_result.keys())[0].auction_item_spec.item_type, auction_item_specs[0].item_type)
        self.assertEqual(list(bid_2_result.keys())[1].auction_item_spec.item_type, auction_item_specs[0].item_type)
        self.assertEqual(list(bid_2_result.keys())[2].auction_item_spec.item_type, auction_item_specs[0].item_type)
        self.assertEqual(list(bid_2_result.values())[0], 2.0)
        self.assertEqual(list(bid_2_result.values())[1], 2.0)
        self.assertEqual(list(bid_2_result.values())[2], 2.0)
        self.assertEqual(unallocated, { })

    def test_3(self):
        auction_item_specs = self.auction_item_specs
        campaigns = self.campaigns
        auction = self.first_price_auction
        bidder_states = [
                        CampaignBidderState("bidder1", campaigns[0]),
                        CampaignBidderState("bidder2", campaigns[0])
        ]
        auction_item_spec_pmf = PMF({
                    auction_item_specs[0] : 0,
                    auction_item_specs[1] : 1
        })
        bids = [
                Bid("bidder1", auction_item_specs[1], bid_per_item=1.0, total_limit=1.0),
                Bid("bidder2", auction_item_specs[1], bid_per_item=2.0, total_limit=4.0)
        ]
        item_matches_bid_spec_func = lambda item, bid: AuctionItemSpecification.is_a_type_of(item.auction_item_spec, bid.auction_item_spec)
        market = OneCampaignMarket(auction, bidder_states, auction_item_spec_pmf, num_of_items_per_timestep=3)
        results = market.run_auction(bids, item_matches_bid_spec_func)

        bid_1_result = results.get_result(bids[0])
        bid_2_result = results.get_result(bids[1])
        unallocated = results.get_unallocated()

        self.assertEqual(list(bid_1_result.keys())[0].auction_item_spec.item_type, auction_item_specs[1].item_type)
        self.assertEqual(list(bid_2_result.keys())[0].auction_item_spec.item_type, auction_item_specs[1].item_type)
        self.assertEqual(list(bid_2_result.keys())[1].auction_item_spec.item_type, auction_item_specs[1].item_type)
        self.assertEqual(list(bid_1_result.values())[0], 1.0)
        self.assertEqual(list(bid_2_result.values())[0], 2.0)
        self.assertEqual(list(bid_2_result.values())[1], 2.0)
        self.assertEqual(unallocated, { })

    def test_4(self):
      auction_item_specs = self.auction_item_specs
      campaigns = self.campaigns
      auction = self.first_price_auction
      bidder_states = [
                      CampaignBidderState("bidder1", campaigns[0]),
                      CampaignBidderState("bidder2", campaigns[1])
      ]
      auction_item_spec_pmf = PMF({
                  auction_item_specs[0] : 1,
                  auction_item_specs[1] : 1
      })
      bids = [
              Bid("bidder1", auction_item_specs[0], bid_per_item=1.0, total_limit=1.0),
              Bid("bidder2", auction_item_specs[1], bid_per_item=2.0, total_limit=2.0)
      ]
      item_matches_bid_spec_func = lambda item, bid: AuctionItemSpecification.is_a_type_of(item.auction_item_spec, bid.auction_item_spec)
      market = OneCampaignMarket(auction, bidder_states, auction_item_spec_pmf, num_of_items_per_timestep=100)
      results = market.run_auction(bids, item_matches_bid_spec_func)

      bid_1_result = results.get_result(bids[0])
      bid_2_result = results.get_result(bids[1])
      unallocated = results.get_unallocated()

      self.assertEqual(list(bid_1_result.keys())[0].auction_item_spec.item_type, auction_item_specs[0].item_type)
      self.assertEqual(list(bid_2_result.keys())[0].auction_item_spec.item_type, auction_item_specs[1].item_type)

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
      auction_item_specs = self.auction_item_specs
      campaigns = self.campaigns
      auction = self.first_price_auction
      bidder_states = [
                      CampaignBidderState("bidder1", campaigns[0]),
                      CampaignBidderState("bidder2", campaigns[1])
      ]
      auction_item_spec_pmf = PMF({
                  auction_item_specs[0] : 1,
                  auction_item_specs[1] : 1
      })
      item_matches_bid_spec_func = lambda item, bid: AuctionItemSpecification.is_a_type_of(item.auction_item_spec, bid.auction_item_spec)
      market = OneCampaignMarket(auction, bidder_states, auction_item_spec_pmf, num_of_items_per_timestep=100)
      results = None
      for i in range(10):
          bids = [
              Bid("bidder1", auction_item_specs[0], bid_per_item=1.0, total_limit=1.0),
              Bid("bidder2", auction_item_specs[1], bid_per_item=2.0, total_limit=2.0)
          ]
          results = market.run_auction(bids, item_matches_bid_spec_func)

      bid_1_result = results.get_result(bids[0])
      bid_2_result = results.get_result(bids[1])
      unallocated = results.get_unallocated()

      self.assertEqual(list(bid_1_result.keys())[0].auction_item_spec.item_type, auction_item_specs[0].item_type)
      self.assertEqual(list(bid_2_result.keys())[0].auction_item_spec.item_type, auction_item_specs[1].item_type)

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
      auction_item_specs = self.auction_item_specs
      campaigns = self.campaigns
      auction = self.first_price_auction
      bidder_states = [
                      CampaignBidderState("bidder1", campaigns[0]),
                      CampaignBidderState("bidder2", campaigns[1])
      ]
      auction_item_spec_pmf = PMF({
                  auction_item_specs[0] : 1,
                  auction_item_specs[1] : 1
      })
      item_matches_bid_spec_func = lambda item, bid: AuctionItemSpecification.is_a_type_of(item.auction_item_spec, bid.auction_item_spec)
      market = OneCampaignMarket(auction, bidder_states, auction_item_spec_pmf, num_of_items_per_timestep=100)
      results = None
      horizon = 7
      for i in range(horizon):
          bids = [
              Bid("bidder1", auction_item_specs[0], bid_per_item=2.0, total_limit=4.0),
              Bid("bidder1", auction_item_specs[1], bid_per_item=1.5, total_limit=3.0),
              Bid("bidder2", auction_item_specs[0], bid_per_item=1.5, total_limit=3.0),
              Bid("bidder2", auction_item_specs[1], bid_per_item=3.0, total_limit=6.0)
          ]
          results = market.run_auction(bids, item_matches_bid_spec_func)

      bid_1_result = results.get_result(bids[0])
      bid_2_result = results.get_result(bids[1])
      unallocated = results.get_unallocated()

      self.assertEqual(len(unallocated), 92)

      self.assertEqual(bidder_states[0].spend, 7.0*horizon)
      self.assertEqual(bidder_states[0].impressions, 2*horizon)
      self.assertEqual(bidder_states[0].timestep, 1*horizon)
        
      self.assertEqual(bidder_states[1].spend, 9.0*horizon)
      self.assertEqual(bidder_states[1].impressions, 2*horizon)
      self.assertEqual(bidder_states[1].timestep, 1*horizon)


if __name__ == '__main__':
    unittest.main()