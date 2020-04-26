from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Set, List, Dict, Any
import uuid



@dataclass
class AuctionItem:
    item_id: int # TODO: should this be unique? suppose bidder A gets half of quantity and bidder B gets other half.
    name: str
    item_type: Set[str]
    owner: 'typing.Any'
    #quantity: int

    def __init__(self, name: str = None, item_type: Set[str] = {}, owner=None):
        self.item_id = uuid.uuid4().int # item_id
        self.name = name
        self.item_type = item_type
        self.owner = owner
        # self.quantity = quantity

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.item_id == other.item_id

    def __hash__(self):
        return hash(self.item_id)

    def __repr__(self):
        return "{}(item_id: {}, name: {})".format(self.__class__.__name__, self.item_id, self.name)

    def copy(self):
        c = AuctionItem(self.name, self.item_type, self.owner)
        return c

    def item_type_submatches(self, auction_item):
        return self.item_type <= auction_item.item_type
    '''
    @staticmethod
    def split_quantity(auction_item:AuctionItem, orig_item_new_quantity: int, copy_item_new_quantity: int):
        c = auction_item.copy()
        auction_item.quantity = orig_item_new_quantity
        c.quantity = copy_item_new_quantity
        return c

    @staticmethod
    def combine_quantity(auction_item_1:AuctionItem, auction_item_2:AuctionItem, in_place=False):
        if in_place:
            c = auction_item_1
            auction_item_1.quantity += auction_item_2.quantity
            auction_item_2.quantity = 0
        else:
            c = auction_item_1.copy()
            c.quantity = auction_item_1.quantity + auction_item_2.quantity
        return c
    '''

@dataclass
class Bid:
    uid: int
    bidder: 'typing.Any'
    auction_item: AuctionItem # Doesn't have to be a fully-specified AuctionItem, e.g. can contain only item_type
    bid_per_item: float
    total_limit: float # same units as bid_per_item (e.g. dollars)

    def __init__(self, bidder, auction_item: AuctionItem, bid_per_item: float = 0.0, total_limit: float = 0.0):
        self.uid = uuid.uuid4().int
        self.bidder = bidder
        self.auction_item = auction_item
        self.bid_per_item = bid_per_item
        self.total_limit = total_limit
        
        # A bid cannot be negative
        assert self.bid_per_item >= 0
        # A limit cannot be non-positive ## TODO: why? ask enrique
        assert self.total_limit > 0
        # A bid cannot be bigger than its limit, since in the worst case a bidder could end up paying a price arbitrarily close to its bid.
        assert self.bid_per_item <= self.total_limit

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.uid == other.uid

    def __hash__(self):
        return hash(self.uid)

    def __repr__(self):
        return "{}(uid: {}, bidder: {}, auction_item: {})".format(self.__class__.__name__, self.uid, self.bidder, self.auction_item.item_id)

    def deduct_limit(self, price: float):
        self.total_limit -= price


@dataclass
class AuctionResults:
    allocations_and_expenditures: Dict[Bid, Dict[AuctionItem, float]]
    __UNALLOC_KEY: Bid = None

    def __init__(self, allocations_and_expenditures: Dict[Bid, Dict[AuctionItem, float]] = None):
        if allocations_and_expenditures == None:
            self.allocations_and_expenditures = dict()
        else:
            self.allocations_and_expenditures = allocations_and_expenditures
        # Add an __UNALLOC_KEY to represent any items that go unallocated
        self.set_result(self.__UNALLOC_KEY)

    def set_result(self, bid: Bid, auction_item: AuctionItem = None, expenditure: float = 0.0):
        if not (bid in self.allocations_and_expenditures):
            self.allocations_and_expenditures[bid] = {}
        temp_dict = self.allocations_and_expenditures[bid]
        # so override auction_item's expenditure if it's already in the dict
        if auction_item != None:
            temp_dict[auction_item] = expenditure

    def get_result(self, bid: Bid):
        return self.allocations_and_expenditures[bid]

    def set_unallocated(self, auction_item: AuctionItem):
        self.set_result(self.__UNALLOC_KEY, auction_item)

    def get_unallocated(self):
        return self.get_result(self.__UNALLOC_KEY)

    def get_allocations(self, bid: Bid):
        return self.allocations_and_expenditures[bid].keys()

    def get_expenditures(self, bid: Bid):
        return self.allocations_and_expenditures[bid].values()

    def get_item_expenditure(self, bid: Bid, auction_item: AuctionItem):
        return self.allocations_and_expenditures[bid][auction_item]

    def get_total_expenditure(self, bid: Bid):
        return sum(self.get_expenditures(bid))