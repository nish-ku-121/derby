import numpy as np
from derby.core.basic_structures import AuctionItemSpecification
from derby.core.ad_structures import Campaign
from derby.core.auctions import KthPriceAuction
from derby.core.pmfs import PMF
from derby.core.environments import train, generate_trajectories, OneCampaignNDaysEnv
from derby.core.agents import Agent
from derby.core.policies import DummyPolicy1, DummyPolicy2, BudgetPerReachPolicy, DummyREINFORCE
from pprint import pprint
import os
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



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
        states, actions, rewards = generate_trajectories(env, num_of_trajs, horizon_cutoff, debug=debug)
        return states, actions, rewards

    def exp_2(self, debug=True):
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
        states, actions, rewards = generate_trajectories(env, agents, num_of_trajs, horizon_cutoff, debug=debug)
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
        states, actions, rewards = generate_trajectories(env, num_of_trajs, horizon_cutoff, debug=debug)
        return states, actions, rewards

    def exp_4(self, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 1,
                    self.auction_item_specs[1] : 1
        })
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 100
        num_items_per_timestep_max = 101
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
  
        num_of_days = 5 # how long the game lasts
        num_of_trajs = 1 # how many times to run the game from start to finish
        horizon_cutoff = 100
        agents = [
                    Agent("agent1", DummyREINFORCE()), 
#                    Agent("agent2", DummyPolicy1(auction_item_specs[1], 0.5, 0.5))
        ]   
        env.vectorize = True
        env.init(agents, num_of_days)
        
        NUM_EPOCHS = 50
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS))
        import time
        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))
        
        end = time.time()
        avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
        print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        print("Took {} sec to train".format(end-start))
        print("model: {}".format(agents[0].policy))

        return None, None, None

    def exp_5(self, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 1,
                    self.auction_item_specs[1] : 1
        })
        campaign_pmf = PMF({
                    self.campaigns[0] : 1,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 100
        num_items_per_timestep_max = 101
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
  
        num_of_days = 5 # how long the game lasts
        num_of_trajs = 1 # how many times to run the game from start to finish
        horizon_cutoff = 100
        agents = [
                    Agent("agent1", DummyREINFORCE()), 
        ]
        env.vectorize = True
        env.init(agents, num_of_days)
        
        NUM_EPOCHS = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS))
        import time
        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))
        
        end = time.time()
        avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
        print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        print("Took {} sec to train".format(end-start))
        print("model: {}".format(agents[0].policy))


if __name__ == '__main__':
    experiment = Experiment()
    exp_func = experiment.exp_4
    print("Running experiment {}".format(exp_func.__name__))
    states, actions, rewards = exp_func(debug=False)
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