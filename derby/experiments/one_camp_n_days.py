import numpy as np
import logging
# NOTE (legacy experiments):
# The functions named exp_* in this module are intentionally left with their
# original print statements. They are considered legacy / exploratory experiments
# where preserving the exact historical console output is useful for comparison
# and reproducibility. Do NOT migrate prints inside any def exp_* to the central
# logging framework unless the migration policy changes. Non-exp_* helpers
# (e.g. run, setup_*, get_* transformers) have been updated to use the module
# logger for structured logging.
from derby.core.basic_structures import AuctionItemSpecification
from derby.core.ad_structures import Campaign
from derby.core.auctions import KthPriceAuction
from derby.core.pmfs import PMF
from derby.core.environments import train, generate_trajectories, OneCampaignNDaysEnv
from derby.core.agents import Agent
from derby.core.policies import *
from pprint import pprint
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
import sys
import os
import tensorflow as tf

logger = logging.getLogger(__name__)

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class Experiment:

    def __init__(self, seed: int | None = None):
        # Reset class-level UID generators so that spec and campaign IDs are predictable
        # within each experiment run. This avoids cross-run drift when multiple processes
        # or repeated runs occur in the same Python interpreter.
        try:
            import itertools
            from derby.core.basic_structures import AuctionItemSpecification
            from derby.core.ad_structures import Campaign
            AuctionItemSpecification._uid_generator = itertools.count(1)
            Campaign._uid_generator = itertools.count(1)
        except Exception:
            # Best-effort; if modules are not yet imported, continue.
            pass

        # Optional global seeding for reproducibility
        # Only performed if seed is not None to preserve prior stochastic behavior by default.
        self.seed = seed
        if seed is not None:
            try:
                import random
                random.seed(seed)
            except Exception:
                pass
            try:
                np.random.seed(seed)
            except Exception:
                pass
            try:
                import tensorflow as tf  # noqa
                tf.random.set_seed(seed)  # type: ignore[attr-defined]
            except Exception:
                pass
            # Help make hashing deterministic in some Python ops
            os.environ.setdefault('PYTHONHASHSEED', str(seed))
            # Informational print (kept minimal to avoid noisy logs)
            logger.info("[Experiment] Seed set to %s", seed)
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


    def setup_1(self):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        for c in campaigns:
            logger.debug("campaign: %s", c)

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
        return env, auction_item_spec_ids


    def setup_2(self):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 1,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [spec.uid for spec in self.auction_item_specs]
        campaign_pmf = PMF({
                    self.campaigns[0] : 1,
                    self.campaigns[1] : 1
        })
        for c in campaigns:
            logger.debug("campaign: %s", c)

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
        return env, auction_item_spec_ids


    def get_states_scaler_descaler(self, samples): 
        '''
        :param samples: an array of shape [num_of_samples, num_of_agents, state_size]
        containing samples of states of the environment.
        '''
        samples_shape = samples.shape
        # reshape to [num_of_samples * num_of_agents, state_size]
        samples = samples.reshape(-1, samples_shape[-1])
        states_scaler = MinMaxScaler()
        states_scaler.fit(samples)

        def scale_states_func(states):
            # Input states is an array of shape [batch_size, episode_length, folded_state_size]
            # reshape to [batch_size * episode_length * fold_size, state_size].
            states_reshp = states.reshape(-1, samples_shape[-1])
            # Scale the states.
            scaled_states = states_scaler.transform(states_reshp)
            # Reshape back to original shape.
            return scaled_states.reshape(states.shape)

        def descale_states_func(states):
            # Input states is an array of shape [batch_size, episode_length, folded_state_size]
            # reshape to [batch_size * episode_length * fold_size, state_size].
            states_reshp = states.reshape(-1, samples_shape[-1])
            # Scale the states.
            descaled_states = states_scaler.inverse_transform(states_reshp)
            # Reshape back to original shape.
            return descaled_states.reshape(states.shape)

        return states_scaler, scale_states_func, descale_states_func


    def get_actions_scaler_descaler(self, samples):
        '''
        :param samples: an array of shape [num_of_samples, num_of_agents, state_size]
        containing samples of states of the environment.
        '''
        samples_shape = samples.shape
        # reshape to [num_of_samples * num_of_agents, state_size]
        samples = samples.reshape(-1, samples_shape[-1])
        # an array of shape [num_of_samples * num_of_agents, 1], containing the sample budgets.
        budget_samples = samples[:,1:2]
        actions_scaler = MinMaxScaler()
        actions_scaler.fit(budget_samples)

        def descale_actions_func(scaled_actions):
            # scaled_actions: [batch, episode, num_subactions, subactions_size]
            # subactions_size: [auction_item_spec_id, bid_per_item, total_limit]
            sa_without_ais = scaled_actions[:, :, :, 1:]
            sa_reshaped = sa_without_ais.reshape(-1, sa_without_ais.shape[-1])  # (N, 2)
            # Apply the scaler (fit on (N,1)) to both columns by repeating the scaler's inverse_transform
            # on each column independently, then stacking
            bid_per_item = sa_reshaped[:, 0:1]
            total_limit = sa_reshaped[:, 1:2]
            bid_per_item_descaled = actions_scaler.inverse_transform(bid_per_item)
            total_limit_descaled = actions_scaler.inverse_transform(total_limit)
            descaled = np.concatenate([bid_per_item_descaled, total_limit_descaled], axis=1)
            descaled_actions_without_ais = descaled.reshape(sa_without_ais.shape)
            descaled_actions = np.concatenate((scaled_actions[:, :, :, 0:1], descaled_actions_without_ais), axis=3)
            return descaled_actions

        def scale_actions_func(descaled_actions):
            # descaled_actions: [batch, episode, num_subactions, subactions_size]
            da_without_ais = descaled_actions[:, :, :, 1:]
            da_reshaped = da_without_ais.reshape(-1, da_without_ais.shape[-1])  # (N, 2)
            bid_per_item = da_reshaped[:, 0:1]
            total_limit = da_reshaped[:, 1:2]
            bid_per_item_scaled = actions_scaler.transform(bid_per_item)
            total_limit_scaled = actions_scaler.transform(total_limit)
            scaled = np.concatenate([bid_per_item_scaled, total_limit_scaled], axis=1)
            scaled_actions_without_ais = scaled.reshape(da_without_ais.shape)
            scaled_actions = np.concatenate((descaled_actions[:, :, :, 0:1], scaled_actions_without_ais), axis=3)
            return scaled_actions

        return actions_scaler, scale_actions_func, descale_actions_func


    def get_transformed(self, env):
        # An array of shape [num_of_samples, num_of_agents, state_size].
        # If agents have not been set, num_of_agents defaults to 1.
        samples = env.get_states_samples(10000)
        _, scale_states_func, _  = self.get_states_scaler_descaler(samples)
        actions_scaler, scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(samples)
        
        # shape (num_of_valid_samples, state_size)
        # Use only samples with reach > 0 to avoid division-by-zero (inf) and NaNs.
        reach = samples[:, :, 0]
        valid_mask = reach > 0
        valid_samples = tf.boolean_mask(samples, valid_mask)

        def _mean_bpr():
            return tf.reduce_mean(valid_samples[:, 1] / valid_samples[:, 0])

        def _fallback_bpr():
            # Sensible default if no valid samples exist
            return tf.constant(1.0, dtype=samples.dtype)

        num_valid = tf.reduce_sum(tf.cast(valid_mask, tf.int32))
        avg_budget_per_reach = tf.cond(num_valid > 0, _mean_bpr, _fallback_bpr)

        # MinMaxScaler expects a numeric Python/NumPy value
        avg_bpr_float = float(avg_budget_per_reach.numpy())
        scaled_avg_bpr = actions_scaler.transform([[avg_bpr_float]])[0][0]
        return scale_states_func, actions_scaler, scale_actions_func, descale_actions_func, scaled_avg_bpr


    def run(self, env, agents, num_days, num_trajs, num_epochs, horizon_cutoff, vectorize=True):
        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = horizon_cutoff
        logger.info("days per traj: %s, trajs per epoch: %s, EPOCHS: %s", num_of_days, num_of_trajs, NUM_EPOCHS)

        env.vectorize = vectorize
        env.init(agents, num_of_days)
        logger.info("agent policies: %s", [agent.policy for agent in env.agents])

        start = time.time()
        # Rolling window (epoch means only) per agent for the last 50 epochs
        last_50_epoch_means = {agent.name: [] for agent in env.agents}

        for i in range(NUM_EPOCHS):
            # Debug flag removed; always rely on logger level for verbosity
            train(env, num_of_trajs, horizon_cutoff)
            # Per-epoch mean/std across trajectories for each agent
            epoch_stats = []
            for agent in env.agents:
                last_trajs = agent.cumulative_rewards[-num_of_trajs:]
                mean_epoch = float(np.mean(last_trajs)) if last_trajs.size > 0 else float("nan")
                std_epoch = float(np.std(last_trajs)) if last_trajs.size > 0 else float("nan")
                epoch_stats.append((agent.name, mean_epoch, std_epoch))
                # Maintain rolling 50 means
                l = last_50_epoch_means[agent.name]
                l.append(mean_epoch)
                if len(l) > 50:
                    l.pop(0)
            logger.info("epoch: %s, avg and std rwds: %s", i, epoch_stats)

            if ((i + 1) % 50) == 0:
                # Compute avg/std over the stored epoch means (not raw trajectories)
                avg_and_std_last_50 = []
                max_last_50 = []
                for agent in env.agents:
                    window = last_50_epoch_means[agent.name]
                    if len(window) > 0:
                        avg_ = float(np.mean(window))
                        std_ = float(np.std(window))
                        max_ = float(np.max(window))
                    else:
                        avg_ = std_ = max_ = float("nan")
                    avg_and_std_last_50.append((agent.name, avg_, std_))
                    max_last_50.append((agent.name, max_))
                logger.info("Avg. of last 50 epochs: %s", avg_and_std_last_50)
                logger.info("Max of last 50 epochs (epoch means): %s", max_last_50)
        
        end = time.time()
        logger.info("Took %.2f sec to train", end-start)

    def exp_1(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = self.auction_item_spec_pmf
        campaign_pmf = self.campaign_pmf
        if debug:
            from pprint import pprint as _pprint
            for c in campaigns:
                _pprint(c)
                print()

        num_items_per_timestep_min = 1000
        num_items_per_timestep_max = 1001
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)

        agents = [
                    Agent("agent1", FixedBidPolicy(1.0, 1.0, auction_item_spec=auction_item_specs[0])), 
                    Agent("agent2", FixedBidPolicy(2.0, 2.0, auction_item_spec=auction_item_specs[1]))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS))

        # Vectorize is True
        env.vectorize = True
        env.init(agents, num_of_days)
        states, actions, rewards = generate_trajectories(env, num_of_trajs, horizon_cutoff)
        return states, actions, rewards


    def exp_2(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = self.auction_item_spec_pmf
        campaign_pmf = self.campaign_pmf
        if debug:
            from pprint import pprint as _pprint
            for c in campaigns:
                _pprint(c)
                print()

        num_items_per_timestep_min = 1000
        num_items_per_timestep_max = 1001
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)

        agents = [
                    Agent("agent1", FixedBidPolicy(1.0, 1.0, auction_item_spec=auction_item_specs[0])), 
                    Agent("agent2", FixedBidPolicy(2.0, 2.0, auction_item_spec=auction_item_specs[1]))
                ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS))

        # Vectorize is False
        env.vectorize = False
        env.init(agents, num_of_days)
        states, actions, rewards = generate_trajectories(env, num_of_trajs, horizon_cutoff)
        return states, actions, rewards


    def exp_3(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = self.auction_item_spec_pmf
        campaign_pmf = self.campaign_pmf
        if debug:
            from pprint import pprint as _pprint
            for c in campaigns:
                _pprint(c)
                print()

        num_items_per_timestep_min = 2
        num_items_per_timestep_max = 3
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)

        agents = [
                    Agent("agent1", BudgetPerReachPolicy()), 
                    Agent("agent2", BudgetPerReachPolicy())
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS))

        env.vectorize = True
        env.init(agents, num_of_days)
        states, actions, rewards = generate_trajectories(env, num_of_trajs, horizon_cutoff)
        return states, actions, rewards


    def exp_4(self, num_days, num_trajs, num_epochs, lr, debug=False):
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
            from pprint import pprint as _pprint
            for c in campaigns:
                _pprint(c)
                print()

        num_items_per_timestep_min = 100
        num_items_per_timestep_max = 101
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
  
        agents = [
                    Agent("agent1", DummyREINFORCE(learning_rate=0.0002)), 
#                    Agent("agent2", FixedBidPolicy(0.5, 0.5, auction_item_spec=auction_item_specs[1]))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS))

        env.vectorize = True
        env.init(agents, num_of_days)
        
        # an array of shape [num_of_samples, num_of_agents, state_size]
        agent_states_samples = env.get_states_samples(10000)
        agent_states_samples = agent_states_samples.reshape(-1, *agent_states_samples.shape[2:])
        # scaler = ColumnTransformer([
        #                         ('0', MinMaxScaler(), [0]), 
        #                         ('1', MinMaxScaler(), [1]), 
        #                         #('2', 'passthrough', [2]), 
        #                         ('2', MinMaxScaler(), [2]), 
        #                         ('3', MinMaxScaler(), [3]), 
        #                         ('4', MinMaxScaler(), [4]), 
        #                         ('5', MinMaxScaler(), [5])
        # ])
        scaler = MinMaxScaler()
        scaler.fit(agent_states_samples)
        scale_states_func = lambda states: scaler.transform(states)
        # pprint(scaler.inverse_transform(scaler.transform([[10, 100, 1, 100, 0, 0]])))

        NUM_EPOCHS = 150
        agents[0].policy.build( (NUM_EPOCHS, num_of_trajs, len(agents) * len(agent_states_samples[0])) )
        agents[0].policy.summary()
        print("optimizer: {}, learning_rate: {}".format(agents[0].policy.optimizer, agents[0].policy.learning_rate))
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS))
        
        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff, scale_states_func=scale_states_func, debug=debug)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))
        
        end = time.time()

        avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                    np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
        print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        print("Took {} sec to train".format(end-start))

        return None, None, None


    def exp_5(self, num_days, num_trajs, num_epochs, lr, debug=False):
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
            from pprint import pprint as _pprint
            for c in campaigns:
                _pprint(c)
                print()

        num_items_per_timestep_min = 100
        num_items_per_timestep_max = 101
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
  
        agents = [
                    Agent("agent1", DummyREINFORCE(learning_rate=0.0002)), 
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS))

        env.vectorize = True
        env.init(agents, num_of_days)
        
        # an array of shape [num_of_samples, num_of_agents, state_size]
        agent_states_samples = env.get_states_samples(10000)
        agent_states_samples = agent_states_samples.reshape(-1, *agent_states_samples.shape[2:])
        scaler = MinMaxScaler()
        scaler.fit(agent_states_samples)
        scale_states_func = lambda states: scaler.transform(states)

        NUM_EPOCHS = 150
        agents[0].policy.build( (NUM_EPOCHS, num_of_trajs, len(agents) * len(agent_states_samples[0])) )
        agents[0].policy.summary()
        print("optimizer: {}, learning_rate: {}".format(agents[0].policy.optimizer, agents[0].policy.learning_rate))
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS))
        
        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff, scale_states_func=scale_states_func, debug=debug)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))
        
        end = time.time()

        avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
        print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        print("Took {} sec to train".format(end-start))

        return None, None, None


    def exp_6(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr
                            ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True)
        return None, None, None


    def exp_7(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr
                            ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True)
        return None, None, None


    def exp_8(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_TD_Gaussian_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr
                            ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True)
        return None, None, None


    def exp_9(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_Q_Gaussian_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr
                            ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True)
        return None, None, None


    def exp_10(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Uniform_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr
                            ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True)
        return None, None, None


    def exp_11(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Triangular_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr
                            ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True)
        return None, None, None


    def exp_12(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Gaussian_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr
                            ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True)
        return None, None, None


    def exp_13(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Triangular_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr
                            ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True)
        return None, None, None


    def exp_14(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_TD_Triangular_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr
                            ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True)
        return None, None, None
    

    def exp_15(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_Q_Triangular_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr
                            ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True)
        return None, None, None
    

    def exp_16(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Triangular_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr
                            ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True)
        return None, None, None


    def exp_17(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Baseline_V_Gaussian_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr
                            ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True)
        return None, None, None

      
    def exp_18(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Tabu_Gaussian_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True)
        return None, None, None
                  
        
    def exp_19(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Tabu_Gaussian_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True)
        return None, None, None

      
    def exp_20(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1",
                        AC_Q_Fourier_Gaussian_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True)
        return None, None, None


    def exp_21(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1",
                        REINFORCE_Tabu_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True)
        return None, None, None


    def exp_22(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1",
                        REINFORCE_Tabu_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True)
        return None, None, None
                      
        
    def exp_100(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", REINFORCE_Gaussian_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(1, 1))
        ]
        
        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)
   
        print("agent policies: {}".format([agent.policy for agent in env.agents]))

        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))

            if ((i+1) % 50) == 0:
                avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_101(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", REINFORCE_Baseline_Gaussian_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)

        print("agent policies: {}".format([agent.policy for agent in env.agents]))

        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))

            if ((i+1) % 50) == 0:
                avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_102(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", AC_TD_Gaussian_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)

        print("agent policies: {}".format([agent.policy for agent in env.agents]))

        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))

            if ((i+1) % 50) == 0:
                avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_103(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", AC_Q_Gaussian_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)

        print("agent policies: {}".format([agent.policy for agent in env.agents]))

        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))

            if ((i+1) % 50) == 0:
                avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_104(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", REINFORCE_Uniform_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)
   
        print("agent policies: {}".format([agent.policy for agent in env.agents]))

        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))

            if ((i+1) % 50) == 0:
                avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_105(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", REINFORCE_Triangular_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)
   
        print("agent policies: {}".format([agent.policy for agent in env.agents]))

        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))

            if ((i+1) % 50) == 0:
                avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_106(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", AC_SARSA_Gaussian_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(1, 1))
        ]
        
        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)

        print("agent policies: {}".format([agent.policy for agent in env.agents]))

        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))

            if ((i+1) % 50) == 0:
                avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
         
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_107(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", REINFORCE_Baseline_Triangular_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)
   
        print("agent policies: {}".format([agent.policy for agent in env.agents]))

        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))

            if ((i+1) % 50) == 0:
                avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_108(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", AC_TD_Triangular_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(1, 1))
        ]
        
        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)
   
        print("agent policies: {}".format([agent.policy for agent in env.agents]))

        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))

            if ((i+1) % 50) == 0:
                avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_109(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", AC_Q_Triangular_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)

        print("agent policies: {}".format([agent.policy for agent in env.agents]))

        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))

            if ((i+1) % 50) == 0:
                avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_110(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", AC_SARSA_Triangular_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)

        print("agent policies: {}".format([agent.policy for agent in env.agents]))

        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))

            if ((i+1) % 50) == 0:
                avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_111(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", REINFORCE_Gaussian_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]
        
        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)
   
        print("agent policies: {}".format([agent.policy for agent in env.agents]))

        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))

            if ((i+1) % 50) == 0:
                avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_112(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", REINFORCE_Baseline_Gaussian_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)

        print("agent policies: {}".format([agent.policy for agent in env.agents]))

        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))

            if ((i+1) % 50) == 0:
                avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_113(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", AC_TD_Gaussian_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)

        print("agent policies: {}".format([agent.policy for agent in env.agents]))

        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))

            if ((i+1) % 50) == 0:
                avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_114(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", AC_Q_Gaussian_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)

        print("agent policies: {}".format([agent.policy for agent in env.agents]))

        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))

            if ((i+1) % 50) == 0:
                avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_115(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", REINFORCE_Uniform_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)
   
        print("agent policies: {}".format([agent.policy for agent in env.agents]))

        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))

            if ((i+1) % 50) == 0:
                avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_116(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", REINFORCE_Triangular_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)
   
        print("agent policies: {}".format([agent.policy for agent in env.agents]))

        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))

            if ((i+1) % 50) == 0:
                avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_117(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", AC_SARSA_Gaussian_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]
        
        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)

        print("agent policies: {}".format([agent.policy for agent in env.agents]))

        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))

            if ((i+1) % 50) == 0:
                avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
         
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_118(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", REINFORCE_Baseline_Triangular_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)
   
        print("agent policies: {}".format([agent.policy for agent in env.agents]))

        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))

            if ((i+1) % 50) == 0:
                avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_119(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", AC_TD_Triangular_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]
        
        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)
   
        print("agent policies: {}".format([agent.policy for agent in env.agents]))

        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))

            if ((i+1) % 50) == 0:
                avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_120(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", AC_Q_Triangular_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)

        print("agent policies: {}".format([agent.policy for agent in env.agents]))

        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))

            if ((i+1) % 50) == 0:
                avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_121(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", AC_SARSA_Triangular_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)

        print("agent policies: {}".format([agent.policy for agent in env.agents]))

        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))

            if ((i+1) % 50) == 0:
                avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None

    
    def exp_400(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_2()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True)
        return None, None, None

    
    def exp_1000(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True)
        return None, None, None


    def exp_1001(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True)
        return None, None, None


    def exp_1002(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True)
        return None, None, None


    def exp_1003(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1004(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_Q_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None
   

    def exp_1005(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_Q_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1006(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1007(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1008(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1009(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None

    def exp_1010(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_TD_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1011(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_TD_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1012(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1013(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1014(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v4_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1015(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v4_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1016(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_TD_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1017(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_TD_Gaussian_v4_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1018(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_Q_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1019(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_Q_Gaussian_v4_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1020(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Baseline_V_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1021(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Baseline_V_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1022(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Baseline_V_Gaussian_v4_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1023(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_Q_Baseline_V_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1024(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_Q_Baseline_V_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1025(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_LogNormal_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2000(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2001(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2002(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2003(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2004(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_Q_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None
   

    def exp_2005(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_Q_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2006(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2007(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2008(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2009(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None

    def exp_2010(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_TD_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2011(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_TD_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2012(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2013(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2014(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v4_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2015(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v4_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None

    
    def exp_3000(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3001(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3002(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3003(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3004(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_Q_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None
   

    def exp_3005(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_Q_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3006(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3007(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3008(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3009(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None

    def exp_3010(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_TD_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3011(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_TD_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3012(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3013(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3014(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v4_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3015(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v4_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_4000(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_2()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v2_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_4001(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_2()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_4002(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_2()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_4003(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_2()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_TD_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_4004(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_2()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Baseline_V_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


if __name__ == '__main__':
    # CLI usage (legacy): python one_camp_n_days.py exp num_days num_trajs num_epochs lr [debug] [seed]
    # Added optional seed argument at position 7.
    exp = sys.argv[1]
    num_days = int(sys.argv[2])
    num_trajs = int(sys.argv[3])
    num_epochs = int(sys.argv[4])
    lr = float(sys.argv[5])
    try:
        debug_str = sys.argv[6].strip().lower()
        if (debug_str == 't') or (debug_str == 'true'):
            debug = True
        else:
            debug = False
    except Exception:
        debug = False
    # Optional seed (position 7)
    seed = None
    try:
        seed = int(sys.argv[7])
    except Exception:
        seed = None
    experiment = Experiment(seed=seed)
    function_mappings = {
        'exp_1': experiment.exp_1,
        'exp_2': experiment.exp_2,
        'exp_3': experiment.exp_3,
        'exp_4': experiment.exp_4,
        'exp_5': experiment.exp_5,
        'exp_6': experiment.exp_6, # REINFORCE_Gaussian vs. FixedBidPolicy
        'exp_7': experiment.exp_7, # REINFORCE_Baseline_Gaussian vs. FixedBidPolicy
        'exp_8': experiment.exp_8, # AC_TD_Gaussian vs. FixedBidPolicy
        'exp_9': experiment.exp_9, # AC_Q_Gaussian vs. FixedBidPolicy
        'exp_10': experiment.exp_10, # REINFORCE_Uniform vs. FixedBidPolicy
        'exp_11': experiment.exp_11, # REINFORCE_Triangular vs. FixedBidPolicy
        'exp_12': experiment.exp_12, # AC_SARSA_Gaussian vs. FixedBidPolicy
        'exp_13': experiment.exp_13, # REINFORCE_Baseline_Triangular vs. FixedBidPolicy
        'exp_14': experiment.exp_14, # AC_TD_Triangular vs. FixedBidPolicy
        'exp_15': experiment.exp_15, # AC_Q_Triangular vs. FixedBidPolicy
        'exp_16': experiment.exp_16, # AC_SARSA_Triangular vs. FixedBidPolicy
        'exp_17': experiment.exp_17, # AC_SARSA_Baseline_V_Gaussian vs. FixedBidPolicy
        'exp_18': experiment.exp_18, # REINFORCE_Tabu_Gaussian (w/o rwd shaping) vs. FixedBidPolicy
        'exp_19': experiment.exp_19, # REINFORCE_Tabu_Gaussian (w rwd shaping) vs. FixedBidPolicy
        'exp_20': experiment.exp_20, # AC_Q_Fourier_Gaussian (w/o rwd shaping) vs. FixedBidPolicy
        'exp_21': experiment.exp_21, # REINFORCE_Tabu_Gaussian_v2 vs. FixedBidPolicy
        'exp_22': experiment.exp_22, # REINFORCE_Tabu_Gaussian_v3 vs. FixedBidPolicy
        'exp_100': experiment.exp_100, # REINFORCE_Gaussian vs. StepPolicy (increasing)
        'exp_101': experiment.exp_101, # REINFORCE_Baseline_Gaussian vs. StepPolicy (increasing)
        'exp_102': experiment.exp_102, # AC_TD_Gaussian vs. StepPolicy (increasing)
        'exp_103': experiment.exp_103, # AC_Q_Gaussian vs. StepPolicy (increasing)
        'exp_104': experiment.exp_104, # REINFORCE_Uniform vs. StepPolicy (increasing)
        'exp_105': experiment.exp_105, # REINFORCE_Triangular vs. StepPolicy (increasing)
        'exp_106': experiment.exp_106, # AC_SARSA_Gaussian vs. StepPolicy (increasing)
        'exp_107': experiment.exp_107, # REINFORCE_Baseline_Triangular vs. StepPolicy (increasing)
        'exp_108': experiment.exp_108, # AC_TD_Triangular vs. StepPolicy (increasing)
        'exp_109': experiment.exp_109, # AC_Q_Triangular vs. StepPolicy (increasing)
        'exp_110': experiment.exp_110, # AC_SARSA_Triangular vs. StepPolicy (increasing)
        'exp_111': experiment.exp_111, # REINFORCE_Gaussian vs. StepPolicy (decreasing)
        'exp_112': experiment.exp_112, # REINFORCE_Baseline_Gaussian vs. StepPolicy (decreasing)
        'exp_113': experiment.exp_113, # AC_TD_Gaussian vs. StepPolicy (decreasing)
        'exp_114': experiment.exp_114, # AC_Q_Gaussian vs. StepPolicy (decreasing)
        'exp_115': experiment.exp_115, # REINFORCE_Uniform vs. StepPolicy (decreasing)
        'exp_116': experiment.exp_116, # REINFORCE_Triangular vs. StepPolicy (decreasing)
        'exp_117': experiment.exp_117, # AC_SARSA_Gaussian vs. StepPolicy (decreasing)
        'exp_118': experiment.exp_118, # REINFORCE_Baseline_Triangular vs. StepPolicy (decreasing)
        'exp_119': experiment.exp_119, # AC_TD_Triangular vs. StepPolicy (decreasing)
        'exp_120': experiment.exp_120, # AC_Q_Triangular vs. StepPolicy (decreasing)
        'exp_121': experiment.exp_121, # AC_SARSA_Triangular vs. StepPolicy (decreasing)
        'exp_400': experiment.exp_400, # REINFORCE_v2_Gaussian (w/o rwd shaping) vs. FixedBidPolicy, env setup 2
        'exp_1000': experiment.exp_1000, # REINFORCE_v2_Gaussian (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1001': experiment.exp_1001, # REINFORCE_v2_Gaussian (w/ rwd shaping) vs. FixedBidPolicy
        'exp_1002': experiment.exp_1002, # REINFORCE_v3_Gaussian (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1003': experiment.exp_1003, # REINFORCE_v3_Gaussian (w/ rwd shaping) vs. FixedBidPolicy
        'exp_1004': experiment.exp_1004, # AC_Q_v2_Gaussian (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1005': experiment.exp_1005, # AC_Q_v2_Gaussian (w/ rwd shaping) vs. FixedBidPolicy
        'exp_1006': experiment.exp_1006, # REINFORCE_Baseline_Gaussian_v2 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1007': experiment.exp_1007, # REINFORCE_Baseline_Gaussian_v2 (w/ rwd shaping) vs. FixedBidPolicy
        'exp_1008': experiment.exp_1008, # REINFORCE_Baseline_Gaussian_v3 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1009': experiment.exp_1009, # REINFORCE_Baseline_Gaussian_v3 (w/ rwd shaping) vs. FixedBidPolicy
        'exp_1010': experiment.exp_1010, # AC_TD_Gaussian_v2 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1011': experiment.exp_1011, # AC_TD_Gaussian_v2 (w/ rwd shaping) vs. FixedBidPolicy
        'exp_1012': experiment.exp_1012, # AC_SARSA_Gaussian_v2 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1013': experiment.exp_1013, # AC_SARSA_Gaussian_v2 (w/ rwd shaping) vs. FixedBidPolicy
        'exp_1014': experiment.exp_1014, # REINFORCE_Gaussian_v4 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1015': experiment.exp_1015, # REINFORCE_Baseline_Gaussian_v4 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1016': experiment.exp_1016, # AC_TD_Gaussian_v3 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1017': experiment.exp_1017, # AC_TD_Gaussian_v4 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1018': experiment.exp_1018, # AC_Q_Gaussian_v3 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1019': experiment.exp_1019, # AC_Q_Gaussian_v4 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1020': experiment.exp_1020, # AC_SARSA_Baseline_V_Gaussian_v2 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1021': experiment.exp_1021, # AC_SARSA_Baseline_V_Gaussian_v3 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1022': experiment.exp_1022, # AC_SARSA_Baseline_V_Gaussian_v4 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1023': experiment.exp_1023, # AC_Q_Baseline_V_Gaussian_v2 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1024': experiment.exp_1024, # AC_Q_Baseline_V_Gaussian_v3 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1025': experiment.exp_1025, # REINFORCE_Baseline_LogNormal_v3_1 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_2000': experiment.exp_2000, # REINFORCE_v2_Gaussian (w/o rwd shaping) vs. StepPolicy (increasing)
        'exp_2001': experiment.exp_2001, # REINFORCE_v2_Gaussian (w/ rwd shaping) vs. StepPolicy (increasing)
        'exp_2002': experiment.exp_2002, # REINFORCE_v3_Gaussian (w/o rwd shaping) vs. StepPolicy (increasing)
        'exp_2003': experiment.exp_2003, # REINFORCE_v3_Gaussian (w/ rwd shaping) vs. StepPolicy (increasing)
        'exp_2004': experiment.exp_2004, # AC_Q_v2_Gaussian (w/o rwd shaping) vs. StepPolicy (increasing)
        'exp_2005': experiment.exp_2005, # AC_Q_v2_Gaussian (w/ rwd shaping) vs. StepPolicy (increasing)
        'exp_2006': experiment.exp_2006, # REINFORCE_Baseline_Gaussian_v2 (w/o rwd shaping) vs. StepPolicy (increasing)
        'exp_2007': experiment.exp_2007, # REINFORCE_Baseline_Gaussian_v2 (w/ rwd shaping) vs. StepPolicy (increasing)
        'exp_2008': experiment.exp_2008, # REINFORCE_Baseline_Gaussian_v3 (w/o rwd shaping) vs. StepPolicy (increasing)
        'exp_2009': experiment.exp_2009, # REINFORCE_Baseline_Gaussian_v3 (w/ rwd shaping) vs. StepPolicy (increasing)
        'exp_2010': experiment.exp_2010, # AC_TD_Gaussian_v2 (w/o rwd shaping) vs. StepPolicy (increasing)
        'exp_2011': experiment.exp_2011, # AC_TD_Gaussian_v2 (w/ rwd shaping) vs. StepPolicy (increasing)
        'exp_2012': experiment.exp_2012, # AC_SARSA_Gaussian_v2 (w/o rwd shaping) vs. StepPolicy (increasing)
        'exp_2013': experiment.exp_2013, # AC_SARSA_Gaussian_v2 (w/ rwd shaping) vs. StepPolicy (increasing)
        'exp_2014': experiment.exp_2014, # REINFORCE_Gaussian_v4 (w/o rwd shaping) vs. StepPolicy (increasing)
        'exp_2015': experiment.exp_2015, # REINFORCE_Baseline_Gaussian_v4 (w/o rwd shaping) vs. StepPolicy (increasing)
        'exp_3000': experiment.exp_3000, # REINFORCE_v2_Gaussian (w/o rwd shaping) vs. StepPolicy (decreasing)
        'exp_3001': experiment.exp_3001, # REINFORCE_v2_Gaussian (w/ rwd shaping) vs. StepPolicy (decreasing)
        'exp_3002': experiment.exp_3002, # REINFORCE_v3_Gaussian (w/o rwd shaping) vs. StepPolicy (decreasing)
        'exp_3003': experiment.exp_3003, # REINFORCE_v3_Gaussian (w/ rwd shaping) vs. StepPolicy (decreasing)
        'exp_3004': experiment.exp_3004, # AC_Q_v2_Gaussian (w/o rwd shaping) vs. StepPolicy (decreasing)
        'exp_3005': experiment.exp_3005, # AC_Q_v2_Gaussian (w/ rwd shaping) vs. StepPolicy (decreasing)
        'exp_3006': experiment.exp_3006, # REINFORCE_Baseline_Gaussian_v2 (w/o rwd shaping) vs. StepPolicy (decreasing)
        'exp_3007': experiment.exp_3007, # REINFORCE_Baseline_Gaussian_v2 (w/ rwd shaping) vs. StepPolicy (decreasing)
        'exp_3008': experiment.exp_3008, # REINFORCE_Baseline_Gaussian_v3 (w/o rwd shaping) vs. StepPolicy (decreasing)
        'exp_3009': experiment.exp_3009, # REINFORCE_Baseline_Gaussian_v3 (w/ rwd shaping) vs. StepPolicy (decreasing)
        'exp_3010': experiment.exp_3010, # AC_TD_Gaussian_v2 (w/o rwd shaping) vs. StepPolicy (decreasing)
        'exp_3011': experiment.exp_3011, # AC_TD_Gaussian_v2 (w/ rwd shaping) vs. StepPolicy (decreasing)
        'exp_3012': experiment.exp_3012, # AC_SARSA_Gaussian_v2 (w/o rwd shaping) vs. StepPolicy (decreasing)
        'exp_3013': experiment.exp_3013, # AC_SARSA_Gaussian_v2 (w/ rwd shaping) vs. StepPolicy (decreasing)
        'exp_3014': experiment.exp_3014, # REINFORCE_Gaussian_v4 (w/o rwd shaping) vs. StepPolicy (decreasing)
        'exp_3015': experiment.exp_3015, # REINFORCE_Baseline_Gaussian_v4 (w/o rwd shaping) vs. StepPolicy (decreasing)
        'exp_4000': experiment.exp_4000, # REINFORCE_v2_1_Gaussian (w/o rwd shaping) vs. FixedBidPolicy, env setup 2
        'exp_4001': experiment.exp_4001, # REINFORCE_v3_1_Gaussian (w/o rwd shaping) vs. FixedBidPolicy, env setup 2
        'exp_4002': experiment.exp_4002, # REINFORCE_Baseline_Gaussian_v3_1 (w/o rwd shaping) vs. FixedBidPolicy, env setup 2
        'exp_4003': experiment.exp_4003, # AC_TD_Gaussian_v3_1 (w/o rwd shaping) vs. FixedBidPolicy, env setup 2
        'exp_4004': experiment.exp_4004, # AC_SARSA_Baseline_V_Gaussian_v3_1 (w/o rwd shaping) vs. FixedBidPolicy, env setup 2
    }
    try:
        exp_func = function_mappings[exp]
    except KeyError:
        raise ValueError('invalid input')

    print("Running experiment {}".format(exp_func.__name__))
    states, actions, rewards = exp_func(num_days, num_trajs, num_epochs, lr, debug=debug)
    if debug:
        if states is not None:
            logger.debug("states shape: %s", states.shape)
            logger.debug("states:\n%s", states)
        if actions is not None:
            logger.debug("actions:\n%s", actions)
        if rewards is not None:
            logger.debug("rewards:\n%s", rewards)