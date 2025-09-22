from abc import ABC, abstractmethod
from typing import Set, List, Dict, Any, TypeVar, Iterable
import numpy as np
from derby.core.basic_structures import AuctionItemSpecification, AuctionItem, Bid
from derby.core.pmfs import PMF
from derby.core.auctions import AbstractAuction
from derby.core.ad_structures import Campaign
from derby.core.states import State, CampaignBidderState
from derby.core.markets import OneCampaignMarket, SequentialAuctionMarket
from derby.core.utils import flatten_2d
# cannot import this as it creates a circular import dependency
# environments -imports-> agents -imports-> policies -imports-> environments
# from derby.core.agents import Agent
import os
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def generate_trajectories(env, num_of_trajs, horizon_cutoff, scale_states_func=None,
                          debug=False, update_policies_after_every_step=False):
    """
    Generate trajectories for all agents in the environment.
    Ensures correct shapes for actions to keep bid_per_item <= total_limit.
    """
    if scale_states_func is None:
        scale_states_func = lambda states: states

    num_agents = len(env.agents)
    num_subactions = env.agents[0].policy.num_subactions
    subaction_size = 1 + env.agents[0].policy.num_dist_per_subaction  # auction_item_spec_id + bid + total_limit

    # Preallocate lists for efficiency
    all_traj_states_list = []
    all_traj_actions_lists = [[] for _ in range(num_agents)]
    all_traj_rewards_list = []

    for traj_idx in range(num_of_trajs):
        # Reset environment
        all_agents_states = env.reset()
        all_agents_states = scale_states_func(all_agents_states)
        all_agents_states = np.array(all_agents_states, dtype=np.float32)

        # Preallocate trajectory arrays
        traj_states = np.zeros((horizon_cutoff + 1, num_agents, all_agents_states.shape[-1]), dtype=np.float32)
        traj_states[0] = all_agents_states
        traj_rewards = np.zeros((horizon_cutoff, num_agents), dtype=np.float32)
        traj_actions = [np.zeros((horizon_cutoff, num_subactions, subaction_size), dtype=np.float32) 
                        for _ in range(num_agents)]

        done = False
        t = 0
        while not done and t < horizon_cutoff:
            actions = []
            for ai, agent in enumerate(env.agents):
                agent_states = env.get_folded_states(agent, traj_states[None, :t+1])
                act = agent.compute_action(agent_states)  # shape: [1, subaction_size]
                act = np.array(act, dtype=np.float32)
                actions.append(act)

                # Store the full action vector (not just the first element)
                traj_actions[ai][t, 0, :] = act[0]  # [subaction_size] -> [spec_id, bid_per_item, total_limit]
            # Pass full actions to env.step
            next_states, rewards, done = env.step(actions)
            next_states = scale_states_func(next_states)
            next_states = np.array(next_states, dtype=np.float32)

            traj_states[t+1] = next_states
            traj_rewards[t] = np.array(rewards, dtype=np.float32)

            t += 1


        # Append to epoch arrays
        all_traj_states_list.append(traj_states[:t+1][None])  # add batch dim
        all_traj_rewards_list.append(traj_rewards[:t][None])  # add batch dim
        for ai in range(num_agents):
            actions_to_append = traj_actions[ai][:t][None]  # add batch dim
            all_traj_actions_lists[ai].append(actions_to_append)

    # Concatenate once for efficiency
    all_traj_states = np.concatenate(all_traj_states_list, axis=0)  # [num_trajs, episode_len, num_agents, state_size]
    all_traj_rewards = np.concatenate(all_traj_rewards_list, axis=0)  # [num_trajs, episode_len, num_agents]
    all_traj_actions = [np.concatenate(per_agent_list, axis=0) for per_agent_list in all_traj_actions_lists]

    if debug:
        traj_lengths = [len(per_agent_list[0][0]) for per_agent_list in all_traj_actions_lists]
        print(f"DEBUG: trajectory lengths: {traj_lengths}")
        print("DEBUG: all_traj_actions[0] shape:", all_traj_actions[0].shape)
        print("DEBUG: all_traj_actions[0] sample:", all_traj_actions[0][0,0])

    return all_traj_states, all_traj_actions, all_traj_rewards


def train_epoch(env, num_of_trajs, horizon_cutoff, scale_states_func=None, discount=1.0, debug=False):
    """
    TF-native training epoch (vectorized)
    """
    states, actions, rewards = generate_trajectories(env, num_of_trajs, horizon_cutoff,
                                                     scale_states_func=scale_states_func,
                                                     debug=debug)
    agents = env.agents
    with tf.GradientTape() as tape:
        per_agent_info = []
        for agent in agents:
            agent_states = env.get_folded_states(agent, states)
            agent_actions = env.get_folded_actions(agent, actions)
            agent_rewards = env.get_folded_rewards(agent, rewards)
            loss = agent.compute_policy_loss(agent_states, agent_actions, agent_rewards)
            per_agent_info.append((agent, agent_states, agent_actions, agent_rewards, loss))

        if debug:
            print("losses:", [info[4] for info in per_agent_info])

    for agent, ag_states, ag_actions, ag_rewards, ag_loss in per_agent_info:
        agent.update_policy(ag_states, ag_actions, ag_rewards, ag_loss, tf_grad_tape=tape)
        agent.update_stats(ag_states, ag_actions, ag_rewards)

    # Summarize rewards
    summary = {}
    for agent in agents:
        if agent.cumulative_rewards is not None and len(agent.cumulative_rewards) > 0:
            mean = float(np.mean(agent.cumulative_rewards[-num_of_trajs:]))
            std = float(np.std(agent.cumulative_rewards[-num_of_trajs:]))
        else:
            mean, std = 0.0, 0.0
        summary[agent.name] = (mean, std)

    return summary


def train(env, num_of_trajs, horizon_cutoff, scale_states_func=None, debug=False, update_policies_after_every_step=False):
    with tf.GradientTape() as tape:
        states, actions, rewards = generate_trajectories(env, num_of_trajs, 
                                                                horizon_cutoff,
                                                                scale_states_func=scale_states_func,
                                                                debug=debug,
                                                                update_policies_after_every_step=update_policies_after_every_step)
# # DEBUG
#         print("actions:\n{}".format(actions))
# #
        # tuple (agent, states for agent, actions for agent, rewards for agent, agent's policy loss)
        # for evey agent in env.agents.
        sarl_per_agent = []
        for agent in env.agents:
            agent_states = env.get_folded_states(agent, states)
            agent_actions = env.get_folded_actions(agent, actions)
            agent_rewards = env.get_folded_rewards(agent, rewards)
            agent_loss = agent.compute_policy_loss(agent_states, agent_actions, agent_rewards)
            sarl_tup = (agent, agent_states, agent_actions, agent_rewards, agent_loss)
            sarl_per_agent.append(sarl_tup)

        if debug:
            print("losses:")
            print([agent_loss for _, _, _, _, agent_loss in sarl_per_agent])
    
    for ag, ag_states, ag_actions, ag_rewards, ag_loss in sarl_per_agent:
        ag.update_policy(ag_states, ag_actions, ag_rewards, ag_loss, tf_grad_tape=tape)
        ag.update_stats(ag_states, ag_actions, ag_rewards)


class AbstractEnvironment(ABC):

    # Types of ways an array of shape [..., num_of_agents, vector_size_per_agent,...]
    # can be folded/reshaped before being passed on.

    # no folding, i.e. all agents relevant. shape [..., num_of_agents, vector_size_per_agent,...]
    FOLD_TYPE_NONE = 0
    # fold all, i.e. all agents relevant. shape [..., num_of_agents * vector_size_per_agent,...]
    FOLD_TYPE_ALL = 1 
    # a single agent's slice. shape [..., vector_size_per_agent,...]
    FOLD_TYPE_SINGLE = 2 

    def __init__(self):
        super().__init__()
        self.agents = None
        self.horizon = 1
        self.done = False

    @abstractmethod
    def init(self, agents, horizon=1):
        self.agents = tuple(agents)
        for i in range(len(self.agents)):
            self.agents[i].agent_num = i
        self.horizon = horizon

    @abstractmethod
    def reset(self):
        '''
        :return: array of shape [num_of_agents] representing 
        the initial states of all agents.
        '''
        pass

    @abstractmethod
    def step(self, actions):
        '''
        :param actions: an array of shape [num_of_agents]
        :return: states, rewards, done. Where:
        states is an array of shape [num_of_agents].
        rewards is an array of shape [num_of_agents].
        done is a boolean specifying if the environment has reach it's last step.
        '''
        pass

    @abstractmethod
    def get_states_samples(self, num_of_samples=1):
        '''
        Returns an array of shape [num_of_samples, num_of_agents, state_size]
        containing samples of possible states of the environment.
        '''
        pass

    @abstractmethod
    def get_folded_states(self, agent, states, fold_type=None):
        pass

    @abstractmethod
    def get_folded_actions(self, agent, actions, fold_type=None):
        pass

    @abstractmethod
    def get_folded_rewards(self, agent, rewards, fold_type=None):
        pass


class MarketEnv(AbstractEnvironment):

    def __init__(self, vectorize=True):
        super().__init__()
        self.vectorize = vectorize
        self._market = None

    @abstractmethod
    def init(self, agents, horizon=1):
        super().init(agents, horizon=horizon)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, actions):
        pass

    @abstractmethod
    def get_states_samples(self, num_of_samples=1):
        pass

    def get_folded_states(self, agent, states, fold_type=None):
        '''
        Takes states and folds according to the fold type:
            1) Does no folding.
            2) Folds all agents' states into full joint states.
               (i.e. shape [batch_size, episode_length, num_of_agents * state_size]).
            3) Picks out only the states of the given agent out of all the agents states.
               (i.e. [batch_size, episode_length, state_size])
        Which case is true is based on whether the policy needs all agent's states or 
        only the given agent's states.
        Let new_state_size represent the new state size in each scenario.
        :param agent: the agent.
        :param states: an array of shape [batch_size, episode_length, num_of_agents, state_size].
        :return: an array of shape [batch_size, episode_length, new_state_size]. Note that 
        new_state_size is () if states are objects instead of vectors.
        '''
        if fold_type is None:
            fold_type = agent.policy.states_fold_type()

        if fold_type == AbstractEnvironment.FOLD_TYPE_NONE:
            return states

        elif fold_type == AbstractEnvironment.FOLD_TYPE_ALL:
            if self.vectorize:
                states_type = type(states)
                if states_type is np.ndarray:
                    rtn = states.reshape(*states.shape[:2], -1)
                elif tf.is_tensor(states):
                    st_shape = tf.shape(states)
                    rtn = tf.reshape(states, [*st_shape[:2]] + [tf.reduce_prod(st_shape[2:])])
                else:
                    raise Exception("states is of type {}, which this func does not know how to fold!".format(states_type))
                return rtn
            else:
                raise Exception("Do not know how to fold for fold type {} in non-vectorized case!".format(fold_type))

        elif fold_type == AbstractEnvironment.FOLD_TYPE_SINGLE:
            return states[:, :, agent.agent_num]
        
        else:
            raise Exception("Do not know how to fold for fold type {}!".format(fold_type))

    def get_folded_actions(self, agent, actions, fold_type=None):
        '''
        :param actions: a list of size num_of_agents, where each entry is 
        an array of shape [batch_size, episode_length, ...].
        '''
        if fold_type is None:
            fold_type = agent.policy.actions_fold_type()

        if fold_type == AbstractEnvironment.FOLD_TYPE_NONE:
            return actions

        elif fold_type == AbstractEnvironment.FOLD_TYPE_ALL:
            Exception("Do not know how to fold for fold type {}!".format(fold_type))
            
        elif fold_type == AbstractEnvironment.FOLD_TYPE_SINGLE:
            return actions[agent.agent_num]
        
        else:
            raise Exception("Do not know how to fold for fold type {}!".format(fold_type))

    def get_folded_rewards(self, agent, rewards, fold_type=None):
        '''
        :param rewards: an array of shape [batch_size, episode_length, num_of_agents].
        '''
        if fold_type is None:
            fold_type = agent.policy.rewards_fold_type()

        if fold_type == AbstractEnvironment.FOLD_TYPE_NONE:
            return rewards

        elif fold_type == AbstractEnvironment.FOLD_TYPE_ALL:
            Exception("Do not know how to fold for fold type {}!".format(fold_type))
            
        elif fold_type == AbstractEnvironment.FOLD_TYPE_SINGLE:
            return rewards[:, :, agent.agent_num]
        
        else:
            raise Exception("Do not know how to fold for fold type {}!".format(fold_type))

    def convert_from_actions_tensor(self, actions, agents, auction_item_specs_by_id):
        '''
        (if self.vectorize is True)
        Assume actions is a numpy array of shape [m, n, k]
        where:
            m = # of agents
            n = # of bids (per agent)
            k = # of fields of a bid vector
        Assuming bid vector is: [auction_item_spec_id, bid_per_item, total_limit].
        Assuming first dimension is in the same order as the agents list passed to init.
        output: 2D list of bid objects. bids[i] are agent i's bids, 
                bids[i][j] is the jth bid of agent i.
        '''
        if not self.vectorize:
            # Assume actions is a 2D array of bid objects, where
            # actions[i] is agent i's bids and actions[i][j] is the jth bid of agent i.
            return actions
        bids = []
        for i in range(len(actions)):
            agent_i_bids = []
            for j in range(len(actions[i])):
                bid_j_of_agent_i = actions[i][j]
                bidder = agents[i]
                auction_item_spec_id = bid_j_of_agent_i[0]
                auction_item_spec = auction_item_specs_by_id[auction_item_spec_id]
                if self.debug:
                    print("DEBUG: actions_to_convert:", bid_j_of_agent_i)
                    print("DEBUG shape:", tf.shape(bid_j_of_agent_i))
                    print(f"DEBUG bid vector for agent {bidder.uid}, auction_item_spec {auction_item_spec.uid}: {bid_j_of_agent_i}")
                    print("DEBUG bid_per_item:", bid_j_of_agent_i[1])
                    print("DEBUG total_limit:", bid_j_of_agent_i[2])
                bid_obj = Bid.from_vector(bid_j_of_agent_i, bidder, auction_item_spec)
                agent_i_bids.append(bid_obj)
            bids.append(agent_i_bids)
        return bids

    def convert_to_states_tensor(self, states):
        """
        Convert list of State objects to numpy array float32.
        """
        if not self.vectorize:
            return states
        states_tensor = np.array([s.to_vector() for s in states], dtype=np.float32)
        return states_tensor


class OneCampaignNDaysEnv(MarketEnv):

    def __init__(self, auction: AbstractAuction, auction_item_spec_pmf: PMF, campaign_pmf: PMF,
                 num_items_per_timestep_min: int, num_items_per_timestep_max: int, vectorize=True, debug=False):
        super().__init__(vectorize)
        self._auction = auction
        self._campaign_pmf = campaign_pmf
        # let this be half-open, i.e. [num_items_per_timestep_min, num_items_per_timestep_max)
        self._num_items_per_timestep_range = (num_items_per_timestep_min, num_items_per_timestep_max)
        self._auction_item_spec_pmf = auction_item_spec_pmf
        self._auction_item_specs_by_id = { spec.uid : spec for spec in self._auction_item_spec_pmf.items() }
        self.debug = debug

    def init(self, agents, horizon=1):
        super().init(agents, horizon=horizon)

    def reset(self):
        self.done = False
        bidder_states = []
        camps = self._campaign_pmf.draw_n(len(self.agents))
        for i in range(len(self.agents)):
            agent = self.agents[i]
            camp = camps[i]
            cbstate = CampaignBidderState(agent, camp)
            bidder_states.append(cbstate)
        self._market = OneCampaignMarket(self._auction, bidder_states, self._auction_item_spec_pmf)
        bidder_states = self.convert_to_states_tensor(bidder_states)
        return bidder_states

    def step(self, actions):
        '''
        input: actions
                (actions same form as in convert_from_actions_tensor func)
        output: states, rewards (as np array), done 
                (states as np array if self.vectorize is True)
        '''
        states = []
        rewards = []
        if self.debug:
            for ai, act in enumerate(actions):
                print(f"DEBUG step input action for agent {ai}:", act)
                print("DEBUG shape:", act.shape)
        
        if not self.done:
            # Convert actions to a 2D list of bid objects, where bids[i] is agent i's 
            # bids and bids[i][j] is the jth bid of agent i.
            bids = self.convert_from_actions_tensor(actions, self.agents, self._auction_item_specs_by_id)

            # Randomize the number of auction items available (i.e. users who show up) in this step
            pre_step_agent_spends = [ self._market.get_bidder_state(agent).spend for agent in self.agents ]
            num_items_min = self._num_items_per_timestep_range[0]
            num_items_max = self._num_items_per_timestep_range[1]
            self._market.num_of_items_per_timestep = np.random.randint(num_items_min, num_items_max)

            # Run the auction
            item_matches_bid_spec_func = lambda item, bid: AuctionItemSpecification.is_a_type_of(item.auction_item_spec, bid.auction_item_spec)
            results = self._market.run_auction(flatten_2d(bids), item_matches_bid_spec_func)
            self.done = (self.horizon != None) and (self._market.timestep == self.horizon)
            
            # Calculate each agent's reward
            for i in range(len(self.agents)):
                agent = self.agents[i]
                agent_bids = bids[i]
                cbstate = self._market.get_bidder_state(agent)
                states.append(cbstate)
                agent_expenditure = ( cbstate.spend - pre_step_agent_spends[i] ) # alternatively, use agent_bids
                agent_reward = -1 * agent_expenditure # i.e. negative reward
                if self.done:
                    # TODO: replace with actual formula
                    agent_reward += min(cbstate.campaign.budget, 
                                        (cbstate.impressions / (1.0 * cbstate.campaign.reach)) * cbstate.campaign.budget)
                rewards.append(agent_reward)

        states = self.convert_to_states_tensor(states)
        # numpy array of shape [m, ]
        rewards = np.array(rewards)
        return states, rewards, self.done

    def get_states_samples(self, num_of_samples=1):
        '''
        :return samples: an array of shape [num_of_samples, num_of_agents, state_size].
        # If agents have not been set, num_of_agents defaults to 1.
        '''
        samples = []
        if self.agents is None:
            temp_agents = [None]
        else:
            temp_agents = self.agents
        for i in range(num_of_samples):
            bidder_states = []
            camps = self._campaign_pmf.draw_n(len(temp_agents))
            for i in range(len(temp_agents)):
                agent = temp_agents[i]
                camp = camps[i]
                # choose between the "null" campaign and camp.
                # to allow for better extrapolation/normalization.
                camp = np.random.choice([None, camp])
                budget = 0 if camp is None else camp.budget
                reach = 0 if camp is None else camp.reach
                cbstate = CampaignBidderState(agent, camp)
                cbstate.spend = np.random.sample() * budget
                cbstate.impressions = np.random.randint(reach + 1)
                cbstate.timestep = np.random.randint(self.horizon + 1)
                bidder_states.append(cbstate)
            bidder_states = self.convert_to_states_tensor(bidder_states)
            samples.append(bidder_states)
        samples = np.array(samples)
        return samples


class SequentialAuctionEnv(MarketEnv):

    def __init__(self, auction: AbstractAuction, all_auction_items: List[AuctionItem], 
                 campaign_pmf: PMF, num_of_items_per_timestep: int = 1, vectorize=True):
        super().__init__(vectorize)
        self._auction = auction
        self._all_auction_items = all_auction_items
        self._campaign_pmf = campaign_pmf
        self._num_items_per_timestep = num_of_items_per_timestep
        self._auction_item_specs_by_id = { item.auction_item_spec.uid : item.auction_item_spec for item in self._all_auction_items }
        self.auction_items = []

    def init(self, agents, horizon=1):
        super().init(agents, horizon=horizon)

    def reset(self):
        self.done = False
        bidder_states = []
        camps = self._campaign_pmf.draw_n(len(self.agents))
        for i in range(len(self.agents)):
            agent = self.agents[i]
            camp = camps[i]
            cbstate = CampaignBidderState(agent, camp)
            bidder_states.append(cbstate)

        self.auction_items = self._all_auction_items[:]
        np.random.shuffle(self.auction_items)

        item_satisfies_campaign_func = lambda item, campaign: AuctionItemSpecification.is_item_type_match(item.auction_item_spec, campaign.target)
        self._market = SequentialAuctionMarket(self._auction, bidder_states, self.auction_items, 
                                   item_satisfies_campaign_func=item_satisfies_campaign_func, 
                                   num_of_items_per_timestep=self._num_items_per_timestep)

        bidder_states = self.convert_to_states_tensor(bidder_states)
        return bidder_states

    def step(self, actions):
        '''
        input: actions
                (actions same form as in convert_from_actions_tensor func)
        output: states, rewards (as np array), done 
                (states as np array if self.vectorize is True)
        '''
        states = []
        rewards = []
        
        if not self.done:
            # Convert actions to a 2D list of bid objects, where bids[i] is agent i's 
            # bids and bids[i][j] is the jth bid of agent i.
            bids = self.convert_from_actions_tensor(actions, self.agents, self._auction_item_specs_by_id)

            pre_step_agent_spends = [ self._market.get_bidder_state(agent).spend for agent in self.agents ]

            # Run the auction
            item_matches_bid_spec_func = lambda item, bid: AuctionItemSpecification.is_item_type_match(item.auction_item_spec, bid.auction_item_spec)
            results = self._market.run_auction(flatten_2d(bids), item_matches_bid_spec_func)
            self.done = (self.horizon != None) and (self._market.timestep == self.horizon)
            
            # Calculate each agent's reward
            for i in range(len(self.agents)):
                agent = self.agents[i]
                agent_bids = bids[i]
                cbstate = self._market.get_bidder_state(agent)
                states.append(cbstate)
                agent_expenditure = ( cbstate.spend - pre_step_agent_spends[i] ) # alternatively, use agent_bids
                agent_reward = -1 * agent_expenditure # i.e. negative reward
                if self.done:
                    # TODO: replace with actual formula
                    agent_reward += min(cbstate.campaign.budget, 
                                        (cbstate.impressions / (1.0 * cbstate.campaign.reach)) * cbstate.campaign.budget)
                rewards.append(agent_reward)
        
        states = self.convert_to_states_tensor(states)
        # numpy array of shape [m, ]
        rewards = np.array(rewards)
        return states, rewards, self.done