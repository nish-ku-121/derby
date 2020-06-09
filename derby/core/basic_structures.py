from typing import Set, List, Dict, Any
import itertools



class AuctionItemSpecification:
    _uid_generator = itertools.count(1)

    uid: int
    _name: str
    _item_type: Set[str]

    def __init__(self, name: str = None, item_type: Set[str] = {}):
        self.uid = next(type(self)._uid_generator)
        self._name = name
        self._item_type = item_type

    @property
    def name(self):
        return self._name

    @property
    def item_type(self):
        return self._item_type
    
    
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.uid == other.uid

    def __hash__(self):
        return hash(self.__class__.__name__ + str(self.uid))

    def __repr__(self):
        return "{}(uid: {}, name: {}, item_type: {})".format(self.__class__.__name__, 
                                                             self.uid, self.name, self.item_type)

    @staticmethod
    def is_exact_match(spec, other):
        if spec != None and other != None:
            return spec._name == other._name and spec._item_type == other._item_type
        else:
            return spec == other # True only if both are None

    @staticmethod
    def is_item_type_match(spec, other):
        if spec != None and other != None:
            return spec._item_type == other._item_type
        else:
            return spec == other # True only if both are None

    @staticmethod
    def is_a_type_of(spec, other):
        # e.g. {male, young} is a type of {male} <=> {male, young} >= {male}
        if spec != None and other != None:
            return spec._item_type >= other._item_type
        else:
            return spec == other # True only if both are None


class AuctionItem:
    _uid_generator = itertools.count(1)

    uid: int
    owner: 'typing.Any'
    auction_item_spec: AuctionItemSpecification

    def __init__(self, auction_item_spec: AuctionItemSpecification, owner=None):
        self.uid = next(type(self)._uid_generator)
        self.owner = owner
        self.auction_item_spec = auction_item_spec

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.uid == other.uid

    def __hash__(self):
        return hash(self.__class__.__name__ + str(self.uid))

    def __repr__(self):
        return "{}(uid: {}, owner: {}, auction_item_spec: {})".format(self.__class__.__name__, 
                                                         self.uid, self.owner, self.auction_item_spec.uid)


class Bid:
    _uid_generator = itertools.count(1)
    
    uid: int
    bidder: 'typing.Any'
    auction_item_spec: AuctionItemSpecification
    bid_per_item: float
    total_limit: float # same units as bid_per_item (e.g. dollars)

    def __init__(self, bidder, auction_item_spec: AuctionItemSpecification, 
                       bid_per_item: float = 0.0, total_limit: float = 0.0):
        self.uid = next(type(self)._uid_generator)
        self.bidder = bidder
        self.auction_item_spec = auction_item_spec
        self.bid_per_item = bid_per_item
        self.total_limit = total_limit
        
        # A bid cannot be negative
        assert self.bid_per_item >= 0
        # A limit cannot be non-positive 
        # TODO: why? ask enrique
        # changed to >=0 instead of >0.
        assert self.total_limit >= 0
        # A bid cannot be bigger than its limit, since in the worst case a bidder could end up paying a price arbitrarily close to its bid.
        assert self.bid_per_item <= self.total_limit

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.uid == other.uid

    def __hash__(self):
        return hash(self.__class__.__name__ + str(self.uid))

    def __repr__(self):
        return "{}(uid: {}, bidder: {}, auction_item_spec: {})".format(self.__class__.__name__, 
                                                                       self.uid, self.bidder, 
                                                                       self.auction_item_spec.uid)

    def deduct_limit(self, price: float):
        self.total_limit -= price

    @classmethod
    def from_vector(cls, bid_vec, bidder, auction_item_spec):
        '''
        Assuming bid vector is: [auction_item_spec_id, bid_per_item, total_limit].
        '''
        bid_per_item = bid_vec[1]
        total_limit = bid_vec[2]
        return cls(bidder, auction_item_spec, bid_per_item, total_limit)

    def to_vector(self):
        return [self.auction_item_spec.uid, self.bid_per_item, self.total_limit]


class AuctionResults:
    allocations_and_expenditures: Dict[Bid, Dict[AuctionItem, float]]
    _UNALLOC_KEY = None

    def __init__(self, allocations_and_expenditures: Dict[Bid, Dict[AuctionItem, float]] = None):
        if allocations_and_expenditures == None:
            self.allocations_and_expenditures = dict()
        else:
            self.allocations_and_expenditures = allocations_and_expenditures
        # Add an _UNALLOC_KEY to represent any items that go unallocated
        self.set_result(type(self)._UNALLOC_KEY)

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
        self.set_result(type(self)._UNALLOC_KEY, auction_item)

    def get_unallocated(self):
        return self.get_result(type(self)._UNALLOC_KEY)

    def get_allocations(self, bid: Bid):
        return self.allocations_and_expenditures[bid].keys()

    def get_expenditures(self, bid: Bid):
        return self.allocations_and_expenditures[bid].values()

    def get_item_expenditure(self, bid: Bid, auction_item: AuctionItem):
        return self.allocations_and_expenditures[bid][auction_item]

    def get_total_expenditure(self, bid: Bid):
        return sum(self.get_expenditures(bid))

    def __repr__(self):
        return str(self.allocations_and_expenditures)

    def __iter__(self):
        return AuctionResultsIterator(self)


class AuctionResultsIterator:
    def __init__(self, auction_results):
        self._auction_results_bids = list(auction_results.allocations_and_expenditures.keys())
        self._auction_results_bids.remove(type(auction_results)._UNALLOC_KEY)
        self._index = 0

    def __next__(self):
        if (self._index >= len(self._auction_results_bids)):
            raise StopIteration
        elem = self._auction_results_bids[self._index]
        self._index += 1
        return elem
