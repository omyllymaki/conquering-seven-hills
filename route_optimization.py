import copy
import logging
import random
from typing import List, Callable

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def exp_schedule(t: float, max_temperature: float = 1.0, decay_constant: float = 0.005):
    return max_temperature * np.exp(-decay_constant * t)


def random_swap(route: List[int], mutation_probability: float = 0.2):
    for k in range(len(route)):
        if random.random() < mutation_probability:
            i, j = random.sample([k + 1 for k in range(len(route) - 2)], 2)
            value_i = route[i]
            value_j = route[j]
            route[i] = value_j
            route[j] = value_i
    return route


class Optimizer:

    def __init__(self, cost_matrix: np.ndarray,
                 mutation_probability: float = 0.1,
                 max_iter: int = 1000,
                 min_temperature: float = 1e-12,
                 cost_threshold: float = -np.inf,
                 schedule_function: Callable = exp_schedule,
                 mutation_function: Callable = random_swap):

        self.cost_matrix = cost_matrix
        self.mutation_probability = mutation_probability
        self.max_iter = max_iter
        self.min_temperature = min_temperature
        self.cost_threshold = cost_threshold
        self.schedule_function = schedule_function
        self.mutation_function = mutation_function

        self.temperatures = []
        self.costs = []
        self.probabilities = []
        self.delta_costs = []

    def run(self, init_route: List[int]):
        current_route = init_route.copy()
        best_route = current_route.copy()

        current_cost = self._calculate_cost(current_route)
        best_cost = self._calculate_cost(best_route)

        probability, delta_cost = 0, 0

        for t in range(self.max_iter):

            temperature = self.schedule_function(t)

            if temperature < self.min_temperature:
                logger.info("Minimum temperature reached. Return solution")
                return best_route, best_cost

            self.temperatures.append(temperature)
            self.costs.append(current_cost)
            self.probabilities.append(probability)
            self.delta_costs.append(delta_cost)

            mutated_route = self.mutation_function(current_route.copy())
            mutated_route_cost = self._calculate_cost(mutated_route)
            logger.debug(f"Mutated route: {mutated_route}; cost {mutated_route_cost}")
            delta_cost = mutated_route_cost - current_cost

            if delta_cost < 0:
                current_route = mutated_route.copy()
                current_cost = mutated_route_cost
                if current_cost < best_cost:
                    best_route = current_route.copy()
                    best_cost = current_cost
                    logger.info(f"Found better solution; round {t}; cost {best_cost}")
                    if best_cost < self.cost_threshold:
                        logger.info("Cost reached required threshold value. Return solution.")
                        return best_route, best_cost
            else:
                probability = np.exp(-delta_cost / temperature)
                if probability > random.uniform(0.0, 1.0):
                    current_route = mutated_route.copy()
                    current_cost = mutated_route_cost

            logger.debug(f"Round {t}: temperature {temperature}; cost {current_cost}")

        logger.info("Max iteration number reached. Return solution.")
        return best_route, best_cost

    def plot_solution(self):
        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.cla()
        plt.plot(self.costs)
        plt.title("Cost")
        plt.subplot(3, 2, 2)
        plt.cla()
        plt.plot(self.temperatures)
        plt.title("Temperature")
        plt.subplot(3, 2, 4)
        plt.cla()
        plt.plot(self.probabilities)
        plt.title("Probability")
        plt.subplot(3, 2, 6)
        plt.cla()
        plt.plot(self.delta_costs)
        plt.title("Cost change")
        plt.tight_layout()
        plt.show()

    def _calculate_cost(self, route: List[int]) -> float:
        cost = 0
        for i in range(len(route) - 1):
            from_index = route[i]
            to_index = route[i + 1]
            cost += self.cost_matrix[from_index][to_index]
        return cost
