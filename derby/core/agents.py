from typing import Set, List, Dict
import itertools
from derby.core.policies import AbstractPolicy



class Agent:
    _uid_generator = itertools.count(1)

    uid: int
    name: str
    policy: AbstractPolicy
    agent_num: int # assuming this is set by the environment/game
    observes_all_agents_states: bool

    def __init__(self, name: str, policy: AbstractPolicy, observes_all_agents_states: bool = True):
        self.uid = next(type(self)._uid_generator)
        self.name = name
        self.policy = policy
        self.observes_all_agents_states = observes_all_agents_states

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.uid == other.uid

    def __hash__(self):
        return hash(self.__class__.__name__ + str(self.uid))

    def __repr__(self):
        return "{}(uid: {}, name: {}, policy: {})".format(self.__class__.__name__, 
                                                          self.uid, self.name, self.policy)

    def compute_action(self, states):
        '''
        :param states: an array of shape [episode/trajectory length, num of agents, batch size].
        '''
        return self.policy.call(states)

    def update_policy(self, states, actions, rewards):
        '''
        :param states: an array of shape [episode/trajectory length, num of agents, batch size].
        :param actions: an array of shape [episode/trajectory length, num of agents, batch size].
        :param rewards: an array of shape [episode/trajectory length, num of agents, batch size].
        '''
        return self.policy.update_policy(states, actions, rewards)