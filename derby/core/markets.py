from abc import ABC, abstractmethod
from typing import Set, List, Dict, Any, TypeVar, Iterable, Callable
from derby.core.basic_structures import AuctionItemSpecification, AuctionItem, Bid, AuctionResults
from derby.core.pmfs import PMF
from derby.core.auctions import AbstractAuction
from derby.core.states import BidderState, CampaignBidderState



T = TypeVar('T')

class AbstractMarket(ABC):
    _auction: AbstractAuction
    _bidder_states_by_bidder: Dict[T, BidderState]
    _auction_items: List[AuctionItem]
    timestep: int

    def __init__(self, auction: AbstractAuction, bidder_states: Iterable[BidderState], timestep: int = 0):
        super().__init__()
        self._auction = auction
        self._auction_items = []
        self.timestep = timestep
        self._bidder_states_by_bidder = {}
        for bstate in bidder_states:
            self._bidder_states_by_bidder[bstate.bidder] = bstate

    @abstractmethod
    def run_auction(self, bids, item_matches_bid_spec_func=None) -> AuctionResults:
        self.update_auction_items()
        auction_results = self._auction.run_auction(bids, self._auction_items, item_matches_bid_spec_func)
        # The timestep update should preferably come before call to 
        # update_bidder_states, so that the updated timestep is available 
        # to update_bidder_states.
        # More over, if bidder states need to have a timestep field updated,
        # then preferably do it in update_timestep instead of update_bidder_states.
        # Mainly, though, don't double-update a bidder state timestep field
        # in both update_timestep and update_bidder_states.
        self.update_timestep()
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
    _auction_item_spec_pmf: PMF
    num_of_items_per_timestep: int

    def __init__(self, auction: AbstractAuction, bidder_states: Iterable[CampaignBidderState], 
                    auction_item_spec_pmf: PMF, timestep: int = 0, num_of_items_per_timestep: int = 1):
        super().__init__(auction, bidder_states, timestep)
        self._auction_item_spec_pmf = auction_item_spec_pmf
        self.num_of_items_per_timestep = num_of_items_per_timestep

    def run_auction(self, bids, item_matches_bid_spec_func=None):
        return super().run_auction(bids, item_matches_bid_spec_func)

    def update_auction_items(self):
        self._auction_items = []
        specs = self._auction_item_spec_pmf.draw_n(self.num_of_items_per_timestep, replace=True)
        for spec in specs:
            self._auction_items.append(AuctionItem(spec))

    def update_bidder_states(self, auction_results: AuctionResults):
        for bid in auction_results:
            bidder = bid.bidder
            if (bidder != None):
                cbstate = self.get_bidder_state(bidder)
                allocs = auction_results.get_allocations(bid)
                expenditure = auction_results.get_total_expenditure(bid)
                cbstate.spend += expenditure
                cbstate.impressions += len(list(filter(
                        lambda item: AuctionItemSpecification.is_a_type_of(item.auction_item_spec, cbstate.campaign.target), 
                        allocs)))

    def update_timestep(self):
        self.timestep += 1
        bidder_states = self.get_bidder_state()
        for cbstate in bidder_states:
            cbstate.timestep = self.timestep


class SequentialAuctionMarket(AbstractMarket):
    _all_auction_items: List[AuctionItem]
    _all_auction_items_index: int
    _item_satisfies_campaign_func: Callable
    num_of_items_per_timestep: int

    def __init__(self, auction: AbstractAuction, bidder_states: Iterable[CampaignBidderState],
                 all_auction_items: List[AuctionItem], item_satisfies_campaign_func: Callable = None, 
                 timestep: int = 0, num_of_items_per_timestep: int = 1):
        super().__init__(auction, bidder_states, timestep)
        self._all_auction_items = all_auction_items
        self._all_auction_items_index = 0
        if item_satisfies_campaign_func == None:
            self._item_satisfies_campaign_func = lambda item, campaign: AuctionItemSpecification.is_item_type_match(item.auction_item_spec, campaign.target)
        else:
            self._item_satisfies_campaign_func = item_satisfies_campaign_func
        self.num_of_items_per_timestep = num_of_items_per_timestep

    def run_auction(self, bids, item_matches_bid_spec_func=None):
        return super().run_auction(bids, item_matches_bid_spec_func)

    def update_auction_items(self):
        self._auction_items = []
        start_idx = self._all_auction_items_index
        end_idx = min(start_idx + self.num_of_items_per_timestep, len(self._all_auction_items))
        for i in range(start_idx, end_idx):
            self._auction_items.append(self._all_auction_items[i])
            self._all_auction_items_index += 1

    def update_bidder_states(self, auction_results):
        for bid in auction_results:
            bidder = bid.bidder
            if (bidder != None):
                cbstate = self.get_bidder_state(bidder)
                allocs = auction_results.get_allocations(bid)
                expenditure = auction_results.get_total_expenditure(bid)
                cbstate.spend += expenditure
                cbstate.impressions += len(list(filter(
                        lambda item: self._item_satisfies_campaign_func(item, cbstate.campaign), 
                        allocs)))

    def update_timestep(self):
        self.timestep += 1
        bidder_states = self.get_bidder_state()
        for cbstate in bidder_states:
            cbstate.timestep = self.timestep