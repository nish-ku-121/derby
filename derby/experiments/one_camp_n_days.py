import numpy as np
from derby.core.basic_structures import AuctionItemSpecification
from derby.core.ad_structures import Campaign
from derby.core.auctions import KthPriceAuction
from derby.core.pmfs import PMF
from derby.core.environments import OneCampaignNDaysEnv
from derby.core.agents import Agent
from derby.core.policies import DummyPolicy1, DummyPolicy2, BudgetPerReachPolicy
from pprint import pprint



class Experiment:

    def __init__(self):
        self.auction_item_specs = [
                        AuctionItemSpecification(name="male", item_type={"male"}),
                        AuctionItemSpecification(name="female", item_type={"female"})
        ]
        self.campaigns = [
                        Campaign(10, 100, self.auction_item_specs[0]),
                        Campaign(10, 100, self.auction_item_specs[1])
        ]
        self.auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 1,
                    self.auction_item_specs[1] : 1
        })
        self.campaign_pmf = PMF({
                    self.campaigns[0] : 1,
                    self.campaigns[1] : 1
        })
        self.first_price_auction = KthPriceAuction(1)
        self.second_price_auction = KthPriceAuction(2)

    def exp_1(self, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = self.auction_item_spec_pmf
        campaign_pmf = self.campaign_pmf
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1000
        num_items_per_timestep_max = 1001
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)

        horizon_cutoff = 100
        num_of_trajs = 2 # how many times to run the game from start to finish
        num_of_days = 5 # how long the game lasts
        agents = [
                    Agent("agent1", DummyPolicy1(auction_item_specs[0], 1.0, 1.0)), 
                    Agent("agent2", DummyPolicy1(auction_item_specs[1], 2.0, 2.0))
        ]

        env.vectorize = True
        env.init(agents, num_of_days)
        states, actions, rewards = type(env).generate_trajectories(env, num_of_trajs, horizon_cutoff, debug=debug)
        return states, actions, rewards

    def exp_2(self, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = self.auction_item_spec_pmf
        campaign_pmf = self.campaign_pmf
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1000
        num_items_per_timestep_max = 1001
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)

        horizon_cutoff = 100
        num_of_trajs = 2 # how many times to run the game from start to finish
        num_of_days = 5 # how long the game lasts
        agents = [
                    Agent("agent1", None), 
                    Agent("agent2", None)
                ]
        p0 = DummyPolicy2(agents[0], auction_item_specs[0], 1.0, 1.0)
        p1 = DummyPolicy2(agents[1], auction_item_specs[1], 2.0, 2.0)
        agents[0].policy = p0
        agents[1].policy = p1

        env.init(agents, num_of_days)
        env.vectorize = False
        states, actions, rewards = type(env).generate_trajectories(env, agents, num_of_trajs, horizon_cutoff, debug=debug)
        return states, actions, rewards


    def exp_3(self, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = self.auction_item_spec_pmf
        campaign_pmf = self.campaign_pmf
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 2
        num_items_per_timestep_max = 3
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)

        horizon_cutoff = 100
        num_of_trajs = 2 # how many times to run the game from start to finish
        num_of_days = 5 # how long the game lasts
        agents = [
                    Agent("agent1", BudgetPerReachPolicy()), 
                    Agent("agent2", BudgetPerReachPolicy())
        ]

        env.vectorize = True
        env.init(agents, num_of_days)
        states, actions, rewards = type(env).generate_trajectories(env, num_of_trajs, horizon_cutoff, debug=debug)
        return states, actions, rewards


if __name__ == '__main__':
    experiment = Experiment()
    states, actions, rewards = experiment.exp_3(debug=False)
    if states is not None:
        print("states shape: {}".format(states.shape))
        print(states)
        print()
    if actions is not None:
        print("actions shape: {}".format(actions.shape))
        print(actions)
        print()
    if rewards is not None:
        print("rewards shape: {}".format(rewards.shape))
        print(rewards)