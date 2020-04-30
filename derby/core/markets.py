from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import choice
from typing import Set, List, Dict, Any, TypeVar
import uuid
from derby.core.basic_structures import AuctionItem, Bid, AuctionResults
from derby.core.pmf import PMF
from derby.core.auctions import AbstractAuction
from derby.core.states import BidderState



T = TypeVar('T')

class AbstractMarket(ABC):
    auction: AbstractAuction
    bidder_states: Dict[T, BidderState]
    auction_items_pmf: PMF
    __auction_items: List[AuctionItem]
    timestep: int

    def __init__(self, auction: AbstractAuction, bidder_states: List[BidderState], auction_items_pmf: PMF, timestep: int = 0):
        super().__init__()
        self.auction = auction
        self.auction_items_pmf = auction_items_pmf
        self.__auction_items = []
        self.timestep = timestep
        for state in bidder_states:
            self.bidder_states[bidder_states.bidder] = state

    @abstractmethod
    def run_auction(self, bids, items_to_bids_mapping_func=None) -> AuctionResults:
        self.update_auction_items()
        auction_results = self.auction.run_auction(bids, self.__auction_items, items_to_bids_mapping_func)
        self.update_bidder_states(auction_results)
        self.update_timestep()
        return auction_results

    @abstractmethod
    def update_auction_items(self, num_of_items: int = 0):
        pass

    @abstractmethod
    def update_bidder_states(self, auction_results: AuctionResults):
        pass

    @abstractmethod
    def update_timestep(self):
        pass