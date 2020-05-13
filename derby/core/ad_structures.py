from dataclasses import dataclass
from typing import Set, List, Dict, OrderedDict
import uuid
from derby.core.basic_structures import AuctionItem



class Campaign:
    """
    Represents a campaign
    """
    uid: int
    _reach: int
    _budget: float
    _target: AuctionItem ## Which auction item to target

    def __init__(self, reach, budget, target):
        self.uid = uuid.uuid4().int
        self._reach = reach
        self._budget = budget
        self._target = target

    @property
    def reach(self):
        return self._reach

    @property
    def budget(self):
        return self._budget

    @property
    def target(self):
        return self._target

    def __repr__(self):
        return "{}(uid: {}, reach: {}, budget: {}, target: {})".format(self.__class__.__name__,
                                                                       self.uid, self.reach, 
                                                                       self.budget, self.target)

    def __lt__(self, other):
        return (self.budget / self.reach) <= (other.budget / other.reach)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.uid == other.uid

    def __hash__(self):
        return hash(self.__class__.__name__ + str(self.uid))