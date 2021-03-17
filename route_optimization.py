import logging
import random
from typing import List, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)


def exp_schedule(t: float, max_temperature: float = 1.0, decay_constant: float = 0.005) -> float:
    return max_temperature * np.exp(-decay_constant * t)


def random_swap(route: List[int], mutation_probability: float = 0.2) -> List[int]:
    for k in range(len(route)):
        if random.random() < mutation_probability:
            i, j = random.sample([k + 1 for k in range(len(route) - 2)], 2)
            value_i = route[i]
            value_j = route[j]
            route[i] = value_j
            route[j] = value_i
    return route


def energy_probability(delta_cost: float, temperature: float, k: float = 1) -> float:
    return np.exp(-delta_cost / (k * temperature))


class SARouteOptimizer:

    def __init__(self,
                 cost_matrix: np.ndarray,
                 mutation_probability: float = 0.1,
                 max_iter: int = 1000,
                 min_temperature: float = 1e-12,
                 cost_threshold: float = -np.inf,
                 schedule_function: Callable = exp_schedule,
                 mutation_function: Callable = random_swap,
                 probability_function: Callable = energy_probability):

        self.cost_matrix = cost_matrix
        self.mutation_probability = mutation_probability
        self.max_iter = max_iter
        self.min_temperature = min_temperature
        self.cost_threshold = cost_threshold
        self.schedule_function = schedule_function
        self.mutation_function = mutation_function
        self.probability_function = probability_function

        self.temperatures = []
        self.costs = []
        self.probabilities = []
        self.delta_costs = []
        self.is_accepted = []

    def run(self, init_route: List[int]) -> Tuple[List[int], float]:
        current_route = init_route.copy()
        best_route = current_route.copy()

        current_cost = self._calculate_cost(current_route)
        best_cost = self._calculate_cost(best_route)

        probability, delta_cost = 0, 0
        is_accepted = True

        for t in range(self.max_iter):

            temperature = self.schedule_function(t)

            if temperature < self.min_temperature:
                logger.info("Minimum temperature reached. Return solution")
                return best_route, best_cost

            self.temperatures.append(temperature)
            self.costs.append(current_cost)
            self.probabilities.append(probability)
            self.delta_costs.append(delta_cost)
            self.is_accepted.append(is_accepted)

            mutated_route = self.mutation_function(current_route.copy())
            mutated_route_cost = self._calculate_cost(mutated_route)
            logger.debug(f"Mutated route: {mutated_route}; cost {mutated_route_cost}")
            delta_cost = mutated_route_cost - current_cost

            is_accepted = False
            if delta_cost < 0:
                is_accepted = True
                probability = 1
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
                probability = self.probability_function(delta_cost, temperature)
                if probability > random.uniform(0.0, 1.0):
                    is_accepted = True
                    current_route = mutated_route.copy()
                    current_cost = mutated_route_cost

            logger.debug(f"Round {t}: temperature {temperature}; cost {current_cost}")

        logger.info("Max iteration number reached. Return solution.")
        return best_route, best_cost

    def plot_solution(self):
        plt.figure(1)

        ax1 = plt.subplot(1, 2, 1)

        color = 'tab:blue'
        ax1.plot(self.costs, color=color)
        ax1.set_ylabel('Cost', color=color)

        color = 'tab:red'
        ax2 = ax1.twinx()
        ax2.plot(self.temperatures, color=color)
        ax2.set_ylabel('Temperature', color=color)

        ax1.set_xlabel("Iteration")
        plt.title("Cost & temperature")

        plt.subplot(1, 2, 2)
        sc = plt.scatter(self.temperatures,
                         self.delta_costs,
                         c=np.array(self.probabilities) + 0.001,
                         norm=colors.LogNorm(),
                         edgecolors="k",
                         cmap=LinearSegmentedColormap.from_list("MyCmapName", ["b", "r"]))
        plt.colorbar(sc)
        plt.gca().invert_xaxis()
        plt.plot(np.array(self.temperatures)[self.is_accepted],
                 np.array(self.delta_costs)[self.is_accepted],
                 "kx",
                 markersize=3)

        plt.xlabel("Temperature")
        plt.ylabel("Cost change")
        plt.title("Probability")

        plt.tight_layout()
        plt.show()

    def _calculate_cost(self, route: List[int]) -> float:
        cost = 0
        for i in range(len(route) - 1):
            from_index = route[i]
            to_index = route[i + 1]
            cost += self.cost_matrix[from_index][to_index]
        return cost
