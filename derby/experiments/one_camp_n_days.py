import numpy as np
from derby.core.basic_structures import AuctionItemSpecification
from derby.core.ad_structures import Campaign
from derby.core.auctions import KthPriceAuction
from derby.core.pmfs import PMF
from derby.core.environments import train, generate_trajectories, OneCampaignNDaysEnv
from derby.core.agents import Agent
from derby.core.policies import DummyPolicy1, DummyPolicy2, BudgetPerReachPolicy, DummyREINFORCE, REINFORCE_MarketEnv_Continuous
from pprint import pprint
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
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
        num_of_trajs = 20 # how many times to run the game from start to finish
        horizon_cutoff = 100
        agents = [
                    Agent("agent1", DummyREINFORCE(learning_rate=0.0002)), 
#                    Agent("agent2", DummyPolicy1(auction_item_specs[1], 0.5, 0.5))
        ]   
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
        num_of_trajs = 20 # how many times to run the game from start to finish
        horizon_cutoff = 100
        agents = [
                    Agent("agent1", DummyREINFORCE(learning_rate=0.0002)), 
        ]
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


    def exp_6(self, debug=False):
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
                    Agent("agent1", REINFORCE_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=0.000002)),
                    Agent("agent2", DummyPolicy1(auction_item_specs[1], 5, 5))
        ]
        
        num_of_days = 5 # how long the game lasts
        num_of_trajs = 20 # how many times to run the game
        NUM_EPOCHS = 200 # how many batches of trajs to run
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
        print("optimizer: {}, learning_rate: {}".format(agents[0].policy.optimizer, agents[0].policy.learning_rate))

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

        return None, None, None


if __name__ == '__main__':
    experiment = Experiment()
    exp_func = experiment.exp_6
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