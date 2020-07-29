import numpy as np
from derby.core.basic_structures import AuctionItemSpecification
from derby.core.ad_structures import Campaign
from derby.core.auctions import KthPriceAuction
from derby.core.pmfs import PMF
from derby.core.environments import train, generate_trajectories, OneCampaignNDaysEnv
from derby.core.agents import Agent
from derby.core.policies import FixedBidPolicy, BudgetPerReachPolicy, StepPolicy, \
DummyREINFORCE, REINFORCE_Gaussian_MarketEnv_Continuous, REINFORCE_Baseline_Gaussian_MarketEnv_Continuous, \
REINFORCE_Uniform_MarketEnv_Continuous, REINFORCE_Triangular_MarketEnv_Continuous, \
AC_TD_Gaussian_MarketEnv_Continuous, AC_Q_Gaussian_MarketEnv_Continuous, AC_SARSA_Gaussian_MarketEnv_Continuous, \
REINFORCE_Baseline_Triangular_MarketEnv_Continuous, AC_TD_Triangular_MarketEnv_Continuous, \
AC_Q_Triangular_MarketEnv_Continuous, AC_SARSA_Triangular_MarketEnv_Continuous, \
AC_SARSA_Baseline_V_Gaussian_MarketEnv_Continuous
from pprint import pprint
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
import sys
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


    def get_states_scaler_descaler(self, env):     
        # an array of shape [num_of_samples, num_of_agents, state_size]
        samples = env.get_states_samples(10000)
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

        return scale_states_func, descale_states_func


    def get_actions_scaler_descaler(self, env):
        # an array of shape [num_of_samples, num_of_agents, state_size]
        samples = env.get_states_samples(10000)
        samples_shape = samples.shape
        # reshape to [num_of_samples * num_of_agents, state_size]
        samples = samples.reshape(-1, samples_shape[-1])
        # an array of shape [num_of_samples * num_of_agents, 1], containing the sample budgets.
        budget_samples = samples[:,1:2]
        actions_scaler = MinMaxScaler()
        actions_scaler.fit(budget_samples)

        def descale_actions_func(scaled_actions):
            # Assuming scaled_actions is of shape [batch_size, episode_length, num_of_subactions, subactions_size],
            # where subactions_size is bid_vector_size (i.e. vector [auction_item_spec_id, bid_per_item, total_limit]).
            # Slice out the 1st field of the vector (i.e. the auction_item_spec_id field).
            sa_without_ais = scaled_actions[:,:,:,1:]
            # Reshaping from [batch_size, episode_length, num_of_subactions, subactions_size-1]
            # to [batch_size * episode_length * num_of_subactions, subactions_size-1]
            sa_without_ais_reshp = sa_without_ais.reshape(-1, sa_without_ais.shape[-1])
            # Descale the remaining fields of the bid vectors (i.e. [bid_per_item, total_limit]).
            descaled_actions_without_ais = actions_scaler.inverse_transform(sa_without_ais_reshp)
            # Reshape back to [batch_size, episode_length, num_of_subactions, subactions_size-1]
            descaled_actions_without_ais = descaled_actions_without_ais.reshape(sa_without_ais.shape)
            # Concatenate the sliced out 1st fields with the descaled other fields.
            descaled_actions = np.concatenate((scaled_actions[:,:,:,0:1], descaled_actions_without_ais), axis=3)
            return descaled_actions

        def scale_actions_func(descaled_actions):
            # Assuming scaled_actions is of shape [batch_size, episode_length, num_of_subactions, subactions_size],
            # where subactions_size is bid_vector_size (i.e. vector [auction_item_spec_id, bid_per_item, total_limit]).
            # Slice out the 1st field of the vector (i.e. the auction_item_spec_id field).
            da_without_ais = descaled_actions[:,:,:,1:]
            # Reshaping from [batch_size, episode_length, num_of_subactions, subactions_size-1]
            # to [batch_size * episode_length * num_of_subactions, subactions_size-1]
            da_without_ais_reshp = da_without_ais.reshape(-1, da_without_ais.shape[-1])
            # Scale the remaining fields of the bid vectors (i.e. [bid_per_item, total_limit]).
            scaled_actions_without_ais = actions_scaler.transform(da_without_ais_reshp)
            # Reshape back to [batch_size, episode_length, num_of_subactions, subactions_size-1]
            scaled_actions_without_ais = scaled_actions_without_ais.reshape(da_without_ais.shape)
            # Concatenate the sliced out 1st fields with the scaled other fields.
            scaled_actions = np.concatenate((descaled_actions[:,:,:,0:1], scaled_actions_without_ais), axis=3)
            return scaled_actions

        return scale_actions_func, descale_actions_func


    def exp_1(self, num_days, num_trajs, num_epochs, lr, debug=False):
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
        states, actions, rewards = generate_trajectories(env, num_of_trajs, horizon_cutoff, debug=debug)
        return states, actions, rewards


    def exp_2(self, num_days, num_trajs, num_epochs, lr, debug=False):
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
        states, actions, rewards = generate_trajectories(env, num_of_trajs, horizon_cutoff, debug=debug)
        return states, actions, rewards


    def exp_3(self, num_days, num_trajs, num_epochs, lr, debug=False):
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
        states, actions, rewards = generate_trajectories(env, num_of_trajs, horizon_cutoff, debug=debug)
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
            for c in campaigns:
                pprint(c)
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
            for c in campaigns:
                pprint(c)
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
                    Agent("agent2", FixedBidPolicy(5, 5))
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
   
        # Prints agent1's policy info
        state_size = 6
        agents[0].policy.build( (NUM_EPOCHS, num_of_trajs, len(agents) * state_size) )
        agents[0].policy.summary()

        print("agent policies: {}".format([agent.policy for agent in env.agents]))

        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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


    def exp_7(self, num_days, num_trajs, num_epochs, lr, debug=False):
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
                    Agent("agent2", FixedBidPolicy(5, 5))
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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


    def exp_8(self, num_days, num_trajs, num_epochs, lr, debug=False):
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
                    Agent("agent2", FixedBidPolicy(5, 5))
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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


    def exp_9(self, num_days, num_trajs, num_epochs, lr, debug=False):
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
                    Agent("agent2", FixedBidPolicy(5, 5))
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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


    def exp_10(self, num_days, num_trajs, num_epochs, lr, debug=False):
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
                    Agent("agent2", FixedBidPolicy(5, 5))
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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


    def exp_11(self, num_days, num_trajs, num_epochs, lr, debug=False):
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
                    Agent("agent2", FixedBidPolicy(5, 5))
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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


    def exp_12(self, num_days, num_trajs, num_epochs, lr, debug=False):
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
                    Agent("agent2", FixedBidPolicy(5, 5))
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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


    def exp_13(self, num_days, num_trajs, num_epochs, lr, debug=False):
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
                    Agent("agent2", FixedBidPolicy(5, 5))
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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


    def exp_14(self, num_days, num_trajs, num_epochs, lr, debug=False):
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
                    Agent("agent2", FixedBidPolicy(5, 5))
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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


    def exp_15(self, num_days, num_trajs, num_epochs, lr, debug=False):
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
                    Agent("agent2", FixedBidPolicy(5, 5))
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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


    def exp_16(self, num_days, num_trajs, num_epochs, lr, debug=False):
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
                    Agent("agent2", FixedBidPolicy(5, 5))
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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


    def exp_17(self, num_days, num_trajs, num_epochs, lr, debug=False):
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
                    Agent("agent1", AC_SARSA_Baseline_V_Gaussian_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", FixedBidPolicy(5, 5))
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
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


if __name__ == '__main__':
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
    except:
        debug = False
    experiment = Experiment()
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
        'exp_121': experiment.exp_121 # AC_SARSA_Triangular vs. StepPolicy (decreasing)
    }
    try:
        exp_func = function_mappings[exp]
    except KeyError:
        raise ValueError('invalid input')

    print("Running experiment {}".format(exp_func.__name__))
    states, actions, rewards = exp_func(num_days, num_trajs, num_epochs, lr, debug=debug)
    if debug:
        if states is not None:
            print("states shape: {}".format(states.shape))
            print("states:\n{}".format(states))
            print()
        if actions is not None:
            print("actions:\n{}".format(actions))
            print()
        if rewards is not None:
            print("rewards:\n{}".format(rewards))