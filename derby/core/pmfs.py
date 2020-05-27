from typing import Dict, TypeVar
from derby.core.basic_structures import AuctionItem
import numpy as np



T = TypeVar('T')

class PMF:
    _pmf: Dict[T, int]
    _total_weight: int

    def __init__(self, items_with_weights: Dict[T, int] = None):
        self._pmf = dict()
        self._total_weight = 0
        if items_with_weights != None:
            self.add_items(items_with_weights)

    def items(self):
        return self._pmf.keys()

    def add_items(self, items_with_weights: Dict[T, int], update_existing=False):
        for key in items_with_weights:
            temp_weight = items_with_weights[key]
            if (not update_existing) and (key in self._pmf):
                raise KeyError("item {} already exists in PMF!".format(key))
            else:
                self._total_weight -= self._pmf.get(key, 0.0)
                self._pmf[key] = temp_weight
                self._total_weight += temp_weight

    def delete_item(self, item: T):
        if item in self._pmf:
            temp_weight = self._pmf[item]
            del self._pmf[item]
            self._total_weight -= temp_weight

    def draw_n(self, n: int, replace=True):
        items = []
        probs = []
        total_weight = float(self._total_weight)
        for key in self._pmf:
            items.append(key)
            probs.append((self._pmf[key] / total_weight))
        return np.random.choice(items, n, replace=replace, p=probs)

    def get_total_weight(self):
        return self._total_weight