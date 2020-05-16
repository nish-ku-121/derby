from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import choice
from typing import Set, List, Dict, OrderedDict
import uuid
from derby.core.ad_structures import Campaign



class State(ABC):
    uid: int

    def __init__(self):
        self.uid = uuid.uuid4().int

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.uid == other.uid

    def __hash__(self):
        return hash(self.__class__.__name__ + str(self.uid))


class BidderState(State):
    bidder: 'typing.Any'

    def __init__(self, bidder):
        super().__init__()
        self.bidder = bidder


class CampaignBidderState(BidderState):
    campaign: Campaign
    spend: float # How much has been spent so far on the campaign (e.g. via bids won)
    impressions: int # How many impressions have been acquired
    timestep: int # e.g. day of the adx game

    def __init__(self, bidder, campaign: Campaign, spend=0.0, impressions=0, timestep=0):
        super().__init__(bidder)
        self.campaign = campaign
        self.spend = spend
        self.impressions = impressions
        self.timestep = timestep

    def __repr__(self):
        return "{}(campaign: {}, spend: {}, impressions: {}, timestep: {})".format(
                                                                            self.__class__.__name__, 
                                                                            self.campaign, self.spend, 
                                                                            self.impressions, self.timestep
                                                                        )