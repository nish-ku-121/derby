from abc import ABC, abstractmethod
from typing import List, Dict, Any, TypeVar, Iterable
import numpy as np
from derby.core.basic_structures import AuctionItemSpecification, AuctionItem, Bid
from derby.core.pmfs import PMF
from derby.core.auctions import AbstractAuction
from derby.core.ad_structures import Campaign
from derby.core.states import State, CampaignBidderState
from derby.core.markets import OneCampaignMarket, SequentialAuctionMarket
from derby.core.utils import flatten_2d
import os
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



_warned_legacy_kw = False
 # Legacy globals removed (model memory accounting disabled)

def train(env, num_of_trajs, horizon_cutoff, scale_states_func=None, update_policies_after_every_step=False, **_ignored_legacy_kwargs):
    global _warned_legacy_kw
    if _ignored_legacy_kwargs and not _warned_legacy_kw:
        logger.warning("Deprecated experiment kwarg(s) ignored: %s (legacy debug plumbing removed; set logging level instead)", list(_ignored_legacy_kwargs.keys()))
        _warned_legacy_kw = True
    # Generate trajectories OUTSIDE any GradientTape to avoid recording the entire rollout graph.
    # This significantly reduces tape memory usage when horizon * num_of_trajs is large.
    states, actions, rewards = generate_trajectories(
        env,
        num_of_trajs,
        horizon_cutoff,
        scale_states_func=scale_states_func,
        update_policies_after_every_step=update_policies_after_every_step,
        
    )

    # For each agent, compute loss under its own (short-lived) GradientTape if it is a TF policy.
    # This also fixes an implicit limitation of the previous single-tape approach where multiple
    # gradient() calls could be made on a non-persistent tape (only the first would succeed).
    sarl_per_agent = []  # (agent, states, actions, rewards, loss, tape_or_None)
    for agent in env.agents:
        agent_states = env.get_folded_states(agent, states)
        agent_actions = env.get_folded_actions(agent, actions)
        agent_rewards = env.get_folded_rewards(agent, rewards)
        # Shape normalization safeguards
        try:
            import numpy as _np  # noqa: WPS433
            if isinstance(agent_actions, _np.ndarray) and agent_actions.ndim == 1:
                agent_actions = agent_actions[None, :]
            if isinstance(agent_rewards, _np.ndarray) and agent_rewards.ndim == 1:
                agent_rewards = agent_rewards[None, :]
        except Exception:  # pragma: no cover
            pass
        try:
            if hasattr(agent_rewards, 'shape') and len(agent_rewards.shape) >= 3:
                if agent_rewards.shape[-1] == len(env.agents):
                    agent_rewards = agent_rewards[..., agent.agent_num]
                if agent_rewards.shape[-1] == 1:
                    import numpy as _np  # noqa: WPS433
                    if isinstance(agent_rewards, _np.ndarray):
                        agent_rewards = agent_rewards.squeeze(-1)
        except Exception:  # pragma: no cover
            pass
        if getattr(agent.policy, 'is_tensorflow', False):
            with tf.GradientTape() as agent_tape:
                agent_loss = agent.compute_policy_loss(agent_states, agent_actions, agent_rewards)
            sarl_per_agent.append((agent, agent_states, agent_actions, agent_rewards, agent_loss, agent_tape))
        else:
            agent_loss = agent.compute_policy_loss(agent_states, agent_actions, agent_rewards)
            sarl_per_agent.append((agent, agent_states, agent_actions, agent_rewards, agent_loss, None))


    logger.debug("losses:")
    logger.debug("%s", [agent_loss for _, _, _, _, agent_loss, _ in sarl_per_agent])

    for ag, ag_states, ag_actions, ag_rewards, ag_loss, ag_tape in sarl_per_agent:
        ag.update_policy(ag_states, ag_actions, ag_rewards, ag_loss, tf_grad_tape=ag_tape)
        ag.update_stats(ag_states, ag_actions, ag_rewards)

    return


def generate_trajectories(env, num_of_trajs, horizon_cutoff, scale_states_func=None,
                          update_policies_after_every_step=False, **_ignored):
    """Generate batched trajectories (states, actions, rewards) with float32 upstream.

    Returns:
        states: np.ndarray shape (B, T+1, num_agents, state_dim...)
        actions: list[np.ndarray] length num_agents, each (B, T, action_dim...)
        rewards: np.ndarray shape (B, T, num_agents)
    """
    if scale_states_func is None:
        scale_states_func = lambda s: s

    per_traj_states: List[np.ndarray] = []
    per_traj_actions_per_agent: List[List[np.ndarray]] = []
    per_traj_rewards: List[np.ndarray] = []

    for _ in range(num_of_trajs):
        step_state_slices: List[np.ndarray] = []
        step_rewards_slices: List[np.ndarray] = []
        step_actions_per_agent: List[List[np.ndarray]] = [[] for _ in env.agents]

        all_agents_states = env.reset()
        all_agents_states = scale_states_func(all_agents_states)
        all_agents_states = np.array(all_agents_states, dtype=np.float32)[None, None, :]
        step_state_slices.append(all_agents_states[0])

        for _step in range(horizon_cutoff):
            raw_actions = []
            for agent in env.agents:
                agent_states = env.get_folded_states(agent, all_agents_states[:, -1:])
                raw_actions.append(agent.compute_action(agent_states))
            all_agents_states, rewards, done = env.step(raw_actions)
            all_agents_states = scale_states_func(all_agents_states)
            all_agents_states = np.array(all_agents_states, dtype=np.float32)[None, None, :]
            wrapped_actions = [np.array(a, dtype=np.float32)[None, None] for a in raw_actions]
            rewards = np.array(rewards, dtype=np.float32)[None, None, :]
            if env.vectorize:
                step_state_slices.append(all_agents_states[0])
                step_rewards_slices.append(rewards[0])
                for ai, a in enumerate(wrapped_actions):
                    step_actions_per_agent[ai].append(a)
            if done:
                break

        if env.vectorize:
            traj_states = np.concatenate(step_state_slices, axis=0)[None, ...].astype(np.float32)
            traj_rewards = (np.concatenate(step_rewards_slices, axis=0)[None, ...].astype(np.float32)
                            if step_rewards_slices else np.zeros((1, 0, len(env.agents)), dtype=np.float32))
            traj_actions: List[np.ndarray] = []
            for lst in step_actions_per_agent:
                if lst:
                    traj_actions.append(np.concatenate(lst, axis=1).astype(np.float32))
                else:
                    traj_actions.append(np.zeros((1, 0), dtype=np.float32))
            per_traj_states.append(traj_states)
            per_traj_rewards.append(traj_rewards)
            per_traj_actions_per_agent.append(traj_actions)

    if env.vectorize and per_traj_states:
        all_states = np.concatenate(per_traj_states, axis=0).astype(np.float32)
        all_rewards = np.concatenate(per_traj_rewards, axis=0).astype(np.float32)
        num_agents = len(env.agents)
        all_actions = []
        for a_i in range(num_agents):
            stacked = np.concatenate([ta[a_i] for ta in per_traj_actions_per_agent], axis=0).astype(np.float32)
            all_actions.append(stacked)
    else:
        all_states = all_rewards = None
        all_actions = None
    return all_states, all_actions, all_rewards


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
                # The first column is auction_item_spec_id; ensure it's an int key
                # because some policies produce float/NumPy scalar IDs (e.g., 1.0),
                # which can miss int keys in dict lookups due to hashing differences.
                try:
                    auction_item_spec_id = int(bid_j_of_agent_i[0])
                except Exception:
                    # Fallback: attempt to extract Python scalar first, then cast
                    try:
                        auction_item_spec_id = int(float(bid_j_of_agent_i[0]))
                    except Exception:
                        raise KeyError(f"Invalid auction_item_spec_id type: {type(bid_j_of_agent_i[0])} value={bid_j_of_agent_i[0]}")

                try:
                    auction_item_spec = auction_item_specs_by_id[auction_item_spec_id]
                except KeyError as e:
                    available = sorted(list(auction_item_specs_by_id.keys()))
                    raise KeyError(
                        f"Unknown auction_item_spec_id {auction_item_spec_id}; available IDs: {available}"
                    ) from e
                bid_obj = Bid.from_vector(bid_j_of_agent_i, bidder, auction_item_spec)
                agent_i_bids.append(bid_obj)
            bids.append(agent_i_bids)
        return bids

    def convert_to_states_tensor(self, states: Iterable[State]):
        '''
        Assuming states is of the form:
        [
            state 1,
            ...,
            state n,
            
        ]
        (The canonical use of this func is for n = m, where m is 
         the # of agents)
        '''
        if not self.vectorize:
            return states
        states_tensor = []
        for state in states:
            vec = state.to_vector()
            states_tensor.append(vec)
        # numpy array of shape [n, s]
        # where:
        #   n = # of states
        #   s = # of fields of a state vector
        return np.array(states_tensor)


class OneCampaignNDaysEnv(MarketEnv):

    def __init__(self, auction: AbstractAuction, auction_item_spec_pmf: PMF, campaign_pmf: PMF,
                 num_items_per_timestep_min: int, num_items_per_timestep_max: int, vectorize=True):
        super().__init__(vectorize)
        self._auction = auction
        self._campaign_pmf = campaign_pmf
        # let this be half-open, i.e. [num_items_per_timestep_min, num_items_per_timestep_max)
        self._num_items_per_timestep_range = (num_items_per_timestep_min, num_items_per_timestep_max)
        self._auction_item_spec_pmf = auction_item_spec_pmf
        self._auction_item_specs_by_id = { spec.uid : spec for spec in self._auction_item_spec_pmf.items() }

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
