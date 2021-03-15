import random
from typing import List

import numpy as np


class Route(list):

    def __init__(self, values: List[int], cost_matrix: np.ndarray):
        super().__init__(values)
        self.cost_matrix = cost_matrix
        self._update_cost()

    def mutate(self):
        i, j = random.sample([k + 1 for k in range(len(self) - 2)], 2)
        self._swap(i, j)
        self._update_cost()

    def _update_cost(self):
        cost = 0
        for i in range(len(self) - 1):
            from_index = self[i]
            to_index = self[i + 1]
            cost += self.cost_matrix[from_index][to_index]
        self.cost = cost

    def _swap(self, i, j):
        value_i = self[i]
        value_j = self[j]
        self[i] = value_j
        self[j] = value_i
