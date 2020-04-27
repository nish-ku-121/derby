from dataclasses import dataclass
from typing import Dict, TypeVar
import numpy as np



T = TypeVar('T')

@dataclass
class PMF:
    __pmf: Dict[T, int]
    __total_weight: int

    def __init__(self, items_with_weights: Dict[T, int] = None):
        self.__pmf = dict()
        self.__total_weight = 0
        if items_with_weights != None:
            self.add_items(items_with_weights)   

    def add_items(self, items_with_weights: Dict[T, int], update_existing=False):
        for key in items_with_weights:
            temp_weight = items_with_weights[key]
            if (not update_existing) and (key in self.__pmf):
                raise KeyError("item {} already exists in PMF!".format(key))
            else:
                self.__total_weight -= self.__pmf.get(key, 0.0)
                self.__pmf[key] = temp_weight
                self.__total_weight += temp_weight

    def delete_item(self, item: T):
        if item in self.__pmf:
            temp_weight = self.__pmf[item]
            del self.__pmf[item]
            self.__total_weight -= temp_weight

    def draw_n(self, n: int, replace=False):
        items = []
        probs = []
        total_weight = float(self.__total_weight)
        for key in self.__pmf:
            items.append(key)
            probs.append((self.__pmf[key] / total_weight))
        return np.random.choice(items, n, replace=replace, p=probs)

    def get_total_weight(self):
        return self.__total_weight