import unittest
import numpy as np
from derby.core.basic_structures import AuctionItemSpecification, Bid
from derby.core.ad_structures import Campaign
from derby.core.auctions import KthPriceAuction
from derby.core.pmfs import PMF
from derby.core.environments import OneCampaignNDaysEnv



class TestOneCampaignNDaysEnv(unittest.TestCase):

    def setUp(self):
        self.auction_item_specs = [
                        AuctionItemSpecification(name="male", item_type={"male"}),
                        AuctionItemSpecification(name="female", item_type={"female"})
        ]
        self.campaigns = [
                        Campaign(10, 100, self.auction_item_specs[0]),
                        Campaign(10, 100, self.auction_item_specs[1])
        ]
        self.first_price_auction = KthPriceAuction(1)
        self.second_price_auction = KthPriceAuction(2)

    def test_1(self):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    auction_item_specs[0] : 1,
                    auction_item_specs[1] : 1
        })
        campaign_pmf = PMF({
                    campaigns[0] : 1,
                    campaigns[1] : 1
        })

# DEBUG
        for c in campaigns:
            print(c)
        print()
#

        num_items_per_timestep_min = 1000
        num_items_per_timestep_max = 1001
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)

        max_horizon_length = 100
        num_of_trajs = 2 # number of times to train
        num_of_days = 5 # how long the game lasts
        agents = ["agent1", "agent2"]

        env.vectorize = True
        env.init(agents, num_of_days)
        for i in range(num_of_trajs):
            agent_states = env.reset()
# DEBUG
            print("=== Traj {} ===".format(i))
            print(agent_states)
            print()
#
            for j in range(max_horizon_length):
                actions = [
                        # agent1 bids
                        [
                            Bid("agent1", auction_item_specs[0], bid_per_item=1.0, total_limit=1.0)
                        ],
                        # agent2 bids
                        [
                            Bid("agent2", auction_item_specs[1], bid_per_item=2.0, total_limit=2.0)
                        ]
                ]
                if env.vectorize:
                    for i in range(len(actions)):
                        for j in range(len(actions[i])):
                            actions[i][j] = actions[i][j].to_vector()
                    
                agent_states, rewards, done = env.step(actions)
# DEBUG
                print(agent_states)
                print(rewards)
                print(done)
                print()
#
                if done:
                    break
        

if __name__ == '__main__':
    unittest.main()