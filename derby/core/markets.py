from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import choice
from typing import Set, List, Dict, Any, TypeVar, Iterable
import uuid
from derby.core.basic_structures import AuctionItem, Bid, AuctionResults
from derby.core.pmf import AuctionItemPMF
from derby.core.auctions import AbstractAuction
from derby.core.states import BidderState, CampaignBidderState



T = TypeVar('T')

class AbstractMarket(ABC):
    auction: AbstractAuction
    _bidder_states_by_bidder: Dict[T, BidderState]
    auction_items_pmf: AuctionItemPMF
    _auction_items: List[AuctionItem]
    timestep: int

    def __init__(self, auction: AbstractAuction, bidder_states: Iterable[BidderState], auction_items_pmf: AuctionItemPMF, timestep: int = 0):
        super().__init__()
        self.auction = auction
        self.auction_items_pmf = auction_items_pmf
        self._auction_items = []
        self.timestep = timestep
        self._bidder_states_by_bidder = {}
        for bstate in bidder_states:
            self._bidder_states_by_bidder[bstate.bidder] = bstate

    @abstractmethod
    def run_auction(self, bids, items_to_bids_mapping_func=None) -> AuctionResults:
        self.update_auction_items()
        auction_results = self.auction.run_auction(bids, self._auction_items, items_to_bids_mapping_func)
        self.update_timestep() ## timestep update needs to come right after the auction is run
        self.update_bidder_states(auction_results) 
        return auction_results

    def get_bidder_state(self, bidder=None):
        """
        returns all bidder states if given argument None
        """
        if bidder == None:
            return self._bidder_states_by_bidder.values()
        else:
            return self._bidder_states_by_bidder[bidder]

    @abstractmethod
    def update_auction_items(self):
        pass

    @abstractmethod
    def update_bidder_states(self, auction_results: AuctionResults):
        pass

    @abstractmethod
    def update_timestep(self):
        pass


class OneCampaignMarket(AbstractMarket):
    num_of_items_per_timestep: int

    def __init__(self, auction: AbstractAuction, bidder_states: Iterable[CampaignBidderState], auction_items_pmf: AuctionItemPMF, timestep: int = 0, num_of_items_per_timestep: int = 1):
        super().__init__(auction, bidder_states, auction_items_pmf, timestep)
        self.num_of_items_per_timestep = num_of_items_per_timestep

    def run_auction(self, bids, items_to_bids_mapping_func=None):
        return super().run_auction(bids, items_to_bids_mapping_func)

    def update_auction_items(self):
        self._auction_items = self.auction_items_pmf.draw_n(self.num_of_items_per_timestep, replace=True)

    def update_bidder_states(self, auction_results: AuctionResults):
        for bid in auction_results:
            bidder = bid.bidder
            if (bidder != None):
                bstate = self.get_bidder_state(bidder)
                allocs = auction_results.get_allocations(bid)
                expenditure = auction_results.get_total_expenditure(bid)
                bstate.spend += expenditure
                bstate.impressions += len(allocs)

    def update_timestep(self):
        self.timestep += 1
        bidder_states = self.get_bidder_state()
        for bstate in bidder_states:
            bstate.timestep = self.timestep
