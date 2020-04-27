from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import choice
from typing import Set, List, Dict, OrderedDict
import uuid



class State(ABC):
    uid: int

    def __init__(self):
        self.uid = uuid.uuid4().int

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.uid == other.uid

    def __hash__(self):
        return hash(self.uid)


class BidderState(State):
    bidder: 'typing.Any'

    def __init__(self, bidder):
        super().__init__()
        self.bidder = bidder
