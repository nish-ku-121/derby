from dataclasses import dataclass
from typing import Set, List, Dict, OrderedDict
import uuid
from derby.core.basic_structures import AuctionItem



@dataclass
class Campaign:
    """
    Represents a campaign
    """
    uid: int
    reach: int
    budget: float
    target: AuctionItem ## Which auction item to target

    def __init__(self, reach, budget, target):
        self.uid = uuid.uuid4().int
        self.reach = reach
        self.budget = budget
        self.target = target

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