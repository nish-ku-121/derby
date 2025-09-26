import unittest
import numpy as np
import logging
from derby.core.basic_structures import AuctionItemSpecification, AuctionItem
from derby.core.ad_structures import Campaign
from derby.core.auctions import KthPriceAuction
from derby.core.pmfs import PMF
from derby.core.environments import generate_trajectories, SequentialAuctionEnv
from derby.core.agents import Agent
from derby.core.policies import DummyPolicy1, DummyPolicy2
import pprint

logger = logging.getLogger(__name__)



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
        self.first_price_auction = KthPriceAuction(1)
        self.second_price_auction = KthPriceAuction(2)
        self.campaign_pmf = PMF({
                    self.campaigns[0] : 1,
                    self.campaigns[1] : 1
        })

    def exp_1(self):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        campaign_pmf = self.campaign_pmf
        all_auction_items = [
                AuctionItem(auction_item_specs[0]),
                AuctionItem(auction_item_specs[1]),
                AuctionItem(auction_item_specs[0]),
                AuctionItem(auction_item_specs[0]),
                AuctionItem(auction_item_specs[1])
        ]
        # Always log campaign details at debug level; visibility controlled by logger level.
        for c in campaigns:
            logger.debug("campaign: %s", c)

        num_of_items_per_timestep = 1
        env = SequentialAuctionEnv(auction, all_auction_items, campaign_pmf, num_of_items_per_timestep)

        horizon_cutoff = 100
        num_of_trajs = 2 # how many times to run the game from start to finish
        num_of_days = (len(all_auction_items) + (num_of_items_per_timestep - 1)) // num_of_items_per_timestep # how long the game lasts
        agents = [
                    Agent("agent1", DummyPolicy1(auction_item_specs[0], 1.0, 1.0)), 
                    Agent("agent2", DummyPolicy1(auction_item_specs[1], 2.0, 2.0))
        ]

        env.vectorize = True
        env.init(agents, num_of_days)
        # Pass through legacy debug flag only if generate_trajectories still expects it; otherwise omit.
        # Legacy debug flag removed; always rely on logger level for verbosity.
        try:
            trajs, rewards = generate_trajectories(env, agents, num_of_trajs, horizon_cutoff)
        except TypeError:
            trajs, rewards = generate_trajectories(env, agents, num_of_trajs, horizon_cutoff)
        return trajs, rewards


if __name__ == '__main__':
    experiment = Experiment()
    trajs, rewards = experiment.exp_1()
    if trajs is not None:
        logger.debug("trajs shape: %s", trajs.shape)
        logger.debug("%s", trajs)
    if rewards is not None:
        logger.debug("rewards shape: %s", rewards.shape)
        logger.debug("%s", rewards)