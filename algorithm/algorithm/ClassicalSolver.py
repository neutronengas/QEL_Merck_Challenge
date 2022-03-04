"""#!/usr/bin/env python 3"""
# coding: utf-8

import numpy as np
from typing import List, Tuple

# class for defining Knapsack items
class Item(object):

    def __init__(self, id, profit, weight):
        self.id = id
        self.profit = profit
        self.weight = weight

    def __str__(self) -> str:
        first_line = "Object {}:".format(self.id)
        second_line = "   - profit: {}".format(self.profit)
        third_line = "   - weight: {}".format(self.weight)
        return "\n".join([first_line, second_line, third_line])

# returns optimal Knapsack solution for given Knapsack items
# items are instances of the class Item which is defined in QAOASolver.py
def dynamic_programming(items: List[Item], max_weight: int) -> int:
    profits = [item.profit for item in items]
    weights = [item.weight for item in items]
    R = np.zeros((len(profits) + 1, max_weight + 1))
    for i in range(1, len(profits) + 1):
        for j in range(1, max_weight + 1):
            if weights[i - 1] <= j:
                R[i][j] = max(profits[i - 1] + R[i - 1][j - weights[i - 1]], R[i - 1][j])
            else:
                R[i][j] = R[i - 1][j]
    return R[len(profits)][max_weight]

# returns lazy greedy Knapsack solution and index of last taken item for given Knapsack items
# items are instances of the class Item which is defined in QAOASolver.py
def lazy_greedy(items: List[Item], max_weight: int) -> Tuple[int, float]:
    profits = [item.profit for item in items]
    weights = [item.weight for item in items]
    new_items = [[i, profits[i], weights[i], profits[i] / weights[i], 0] for i in range(len(profits))]
    new_items.sort(key=(lambda x: x[3]), reverse=True)
    total_weight = 0
    r_stop = new_items[-1][-1]
    for item in new_items:
        if total_weight + item[2] <= max_weight:
            total_weight += item[2]
            item[-1] = 1
        else:
            r_stop = item[3] if r_stop == new_items[-1][-1] else r_stop
    new_items.sort(key=(lambda x: x[0]))
    return [i[-1] for i in new_items], r_stop