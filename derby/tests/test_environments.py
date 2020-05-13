import unittest
import numpy as np
from derby.core.basic_structures import AuctionItem, Bid
from derby.core.ad_structures import Campaign
from derby.core.auctions import KthPriceAuction
from derby.core.pmfs import PMF, AuctionItemPMF
from derby.core.environments import OneCampaignNDaysEnv



class TestOneCampaignNDaysEnv(unittest.TestCase):

    def setUp(self):
        self.auction_items = [
                        AuctionItem(name="male1", item_type={"male"}, owner=None),
                        AuctionItem(name="female1", item_type={"female"}, owner=None)
        ]
        self.campaigns = [
                        Campaign(10, 100, self.auction_items[0]),
                        Campaign(10, 100, self.auction_items[1])
        ]
        self.first_price_auction = KthPriceAuction(1)
        self.second_price_auction = KthPriceAuction(2)

    def test_1(self):
        auction_items = self.auction_items
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_pmf = AuctionItemPMF({
                    auction_items[0] : 1,
                    auction_items[1] : 1
        })
        campaign_pmf = PMF({
                    campaigns[0] : 1,
                    campaigns[1] : 1
            })

        for c in campaigns:
            print(c)

        num_items_per_timestep_min = 1000
        num_items_per_timestep_max = 1001
        env = OneCampaignNDaysEnv(auction, num_items_per_timestep_min, num_items_per_timestep_max, 
                                    campaign_pmf, auction_item_pmf)

        max_horizon_length = 100
        num_of_trajs = 2
        num_of_days = 5
        agents = ["agent1", "agent2"]
        env.init(agents, num_of_days)
        for i in range(num_of_trajs):
            print("=== Traj {} ===".format(i))
            agent_states = env.reset()
            print(agent_states)
            print()
            for j in range(max_horizon_length):
                actions = np.array([
                    # agent1 bids
                    [
                        #Bid("agent1", auction_items[0], bid_per_item=1.0, total_limit=1.0)
                        [auction_items[0].item_id, 1.0, 1.0]
                    ],
                    # agent2 bids
                    [
                        #Bid("agent2", auction_items[1], bid_per_item=2.0, total_limit=2.0)
                        [auction_items[1].item_id, 2.0, 2.0]
                    ]
                ])
                agent_states, rewards, done = env.step(actions)
                print(agent_states)
                print(rewards)
                print(done)
                print()
                if done:
                    break
        

if __name__ == '__main__':
    unittest.main()