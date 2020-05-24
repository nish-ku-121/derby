from typing import Set, List, Dict
import itertools
from derby.core.policies import AbstractPolicy



class Agent:
    _uid_generator = itertools.count(1)

    uid: int
    name: str
    policy: AbstractPolicy
    agent_num: int # assuming this is set by the environment/game

    def __init__(self, name: str, policy: AbstractPolicy):
        self.uid = next(type(self)._uid_generator)
        self.name = name
        self.policy = policy
        self.policy.agent = self

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.uid == other.uid

    def __hash__(self):
        return hash(self.__class__.__name__ + str(self.uid))

    def __repr__(self):
        return "{}(uid: {}, name: {}, policy: {})".format(self.__class__.__name__, 
                                                          self.uid, self.name, self.policy)

    def compute_action(self, states):
        '''
        :param states: an np array of shape [num_of_agents, state_size].
        :return: the action to take.
        '''
        states = states[None, None, :]
        if self.policy.is_partial:
            states = states[:, :, self.agent_num]
        return self.policy.choose_actions(self.policy.call(states))[0,0]

    def update_policy(self, states, actions, rewards):
        '''
        :param states: an array of shape [batch_size, episode_length, num_of_agents, state_size].
        :param actions: an array of shape [batch_size, episode_length].
        :param rewards: an array of shape [batch_size, episode_length].
        '''