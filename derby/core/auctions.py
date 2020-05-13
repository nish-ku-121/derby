from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import choice
from typing import Set, List, Dict, OrderedDict
import uuid
from derby.core.basic_structures import AuctionItem, Bid, AuctionResults
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
    def items_to_bids_by_item(bids, auction_items):
        bids_by_item = {item: [] for item in auction_items}
        for bid in bids:
            bid_item = bid.auction_item
            if (bid_item != None):
                if bid_item in bids_by_item:
                        bids_by_item[bid_item].append(bid)
        return bids_by_item

    @staticmethod
    def items_to_bids_by_item_type(bids, auction_items):
        rtn = {item: [] for item in auction_items}
        
        items_by_item_type = {}
        for item in auction_items:
            if item.item_type != None:
                temp_key = frozenset(item.item_type)
            else:
                temp_key = 'None'
            if temp_key in items_by_item_type:
                items_by_item_type[temp_key].append(item)
            else:
                items_by_item_type[temp_key] = [item]

        for bid in bids:
            if (bid.auction_item != None):
                bid_item_type = bid.auction_item.item_type
                if (bid_item_type != None):
                    bid_item_type = frozenset(bid_item_type)
                else:
                    bid_item_type = 'None'
                if bid_item_type in items_by_item_type:
                    relevant_items = items_by_item_type[bid_item_type]
                    for item in relevant_items:
                        rtn[item].append(bid)
        return rtn

    @staticmethod
    def items_to_bids_by_item_type_submatch(bids, auction_items):
        rtn = {}
        for item in auction_items:
            if not (item in rtn):
                rtn[item] = []
            temp_list = rtn[item]
            for bid in bids:
                if bid.auction_item.item_type_submatches(item):
                    temp_list.append(bid)
        return rtn


class KthPriceAuction(AbstractAuction):
    k: int

    def __init__(self, k):
        super().__init__()
        self.k = k

    def run_auction(self, bids, auction_items, items_to_bids_mapping_func=None) -> AuctionResults:
        if (items_to_bids_mapping_func == None):
            items_to_bids_mapping_func = self.items_to_bids_by_item
        # Initialize auction results with all bids being allocated nothing
        results = AuctionResults()
        for bid in bids:
            results.set_result(bid)
        # Get relevant bids for each item
        items_to_bids_mapping = items_to_bids_mapping_func(bids, auction_items)
        for item in auction_items:
            relevant_bids = items_to_bids_mapping[item]
            # Allow only one bid per bidder
            relevant_bids = self.dedup_by_bidder(relevant_bids)
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
            winning_bids = self.get_winning_bids(relevant_bids) 
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