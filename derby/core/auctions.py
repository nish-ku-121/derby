from abc import ABC, abstractmethod
from random import choice
from typing import Set, List, Dict, OrderedDict
import uuid
from derby.core.basic_structures import AuctionItemSpecification, AuctionItem, Bid, AuctionResults
from derby.core.utils import kth_largest



class AbstractAuction(ABC):

    @abstractmethod
    def run_auction(self, bids, auction_items, items_to_bids_mapping_func=None) -> AuctionResults:
        pass

    @staticmethod
    def dedup_by_bidder(bids: List[Bid]) -> List[Bid]:
        """
        returns a list containing only the max bid of each bidder.
        """
        if (bids == None):
            return None
        if (len(bids) == 0):
            return []
        uniques = {}
        rtn = []
        for bid in bids:
            if bid.bidder != None:
                temp_key = bid.bidder
                if not temp_key in uniques:
                    uniques[temp_key] = bid
                else:
                    best_bid = uniques[temp_key]
                    if bid.bid_per_item > best_bid.bid_per_item:
                        uniques[temp_key] = bid
            else:
                raise Exception('bid {} is not owned by any bidder!'.format(bid.uid))
        return list(uniques.values())


    @staticmethod
    def items_to_bids_filter(bids, auction_items, item_matches_bid_spec_func):
        '''
        item_matches_bid_spec_func: 
            function which takes inputs (item, bid) and outputs True/False
            if item matches the bid's spec (based on some notion of "match").
        '''
        rtn = {}
        for item in auction_items:
            if not (item in rtn):
                rtn[item] = []
            temp_list = rtn[item]
            for bid in bids:
                if item_matches_bid_spec_func(item, bid):
                    temp_list.append(bid)
        return rtn


class KthPriceAuction(AbstractAuction):
    k: int

    def __init__(self, k):
        super().__init__()
        self.k = k

    def run_auction(self, bids, auction_items, item_matches_bid_spec_func=None) -> AuctionResults:
        # Set the default item_matches_bid_spec_func
        if (item_matches_bid_spec_func == None):
            item_matches_bid_spec_func = lambda item, bid: AuctionItemSpecification.is_exact_match(item.auction_item_spec, bid.auction_item_spec)
        # Initialize auction results with all bids being allocated nothing
        results = AuctionResults()
        for bid in bids:
            results.set_result(bid)
        # Get relevant bids for each item
        items_to_bids_mapping = type(self).items_to_bids_filter(bids, auction_items, item_matches_bid_spec_func)
        for item in auction_items:
            relevant_bids = items_to_bids_mapping[item]
            # Allow only one bid per bidder
            relevant_bids = type(self).dedup_by_bidder(relevant_bids)
            # If a bid exceeds it's total limit, then ignore it
            relevant_bids = list(filter(lambda b: b.bid_per_item <= b.total_limit, relevant_bids))
            # If there are no bids for the item, then add it as unallocated
            if (len(relevant_bids) == 0):
                results.set_unallocated(item)
                continue
            # Compute the price, which is defined as the kth largest of the relevant bids.
            # Note the price of the kth largest bid is zero if there is no such bid.
            price = kth_largest([bid.bid_per_item for bid in relevant_bids], self.k, 0.0)
            # Get all the winning bids of the relevant bids.
            winning_bids = type(self).get_winning_bids(relevant_bids) 
            # Select a random bid among all winning bids,
            # allocate and price the item to the winner,
            # finally deplete the winner's total limit.
            bid_that_won = choice(winning_bids)
            results.set_result(bid_that_won, item, price)
            bid_that_won.deduct_limit(price)
        return results

    @staticmethod
    def get_winning_bids(bids) -> List[Bid]:
        if (bids == None):
            return None
        if (len(bids) == 0):
            return []
        # Get the winning bid's value
        max_bid_value = max([bid.bid_per_item for bid in bids])
        # Get all the bids that are at least as big as the winning bid.
        winning_bids = list(filter(lambda b: b.bid_per_item >= max_bid_value, bids))
        return winning_bids