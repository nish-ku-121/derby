from typing import Set, List, Dict
import itertools
from derby.core.policies import AbstractPolicy



class Agent:
    _uid_generator = itertools.count(1)

    uid: int
    name: str
    policy: AbstractPolicy

    def __init__(self, name: str, policy: AbstractPolicy):
        self.uid = next(type(self)._uid_generator)
        self.name = name
        self.policy = policy

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.uid == other.uid

    def __hash__(self):
        return hash(self.__class__.__name__ + str(self.uid))

    def __repr__(self):
        return "{}(uid: {}, name: {}, policy: {})".format(self.__class__.__name__, 
                                                          self.uid, self.name, self.policy)

    def compute_action(self, states):
        return self.policy.call(states)