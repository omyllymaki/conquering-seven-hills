import logging
import random
from abc import abstractmethod
from typing import List, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)


class SAModel:
    """
    Use case specific SA model that needs to be inherited by user of SARouteOptimizer.

    Contains methods used in SARouteOptimizer to
    - calculate cost for route
    - mutate route
    - calculate temperature
    - calculate probability from cost change and temperature
    """

    @abstractmethod
    def cost(self, route: List[int]) -> float:
        """
        Calculate cost for route.
        """
        raise NotImplementedError

    @staticmethod
    def schedule(t: int, max_temperature: float = 1.0, decay_constant: float = 0.005) -> float:
        """
        Calculate current temperature from iteration round t.
        """
        return max_temperature * np.exp(-decay_constant * t)

    @staticmethod
    def mutate(route: List[int], mutation_probability: float = 0.2) -> List[int]:
        """
        Mutate (modify) given route. This mutated route will be solution candidate that will be accepted or not, based
        on calculated probability.
        """
        for k in range(len(route)):
            if random.random() < mutation_probability:
                i, j = random.sample([k + 1 for k in range(len(route) - 2)], 2)
                value_i = route[i]
                value_j = route[j]
                route[i] = value_j
                route[j] = value_i
        return route

    @staticmethod
    def probability(delta_cost: float, temperature: float, k: float = 1) -> float:
        """
        Calculate acceptance probability for mutated route, based on cost change (vs. current solution) and temperature.
        """
        if delta_cost < 0:
            return 1
        else:
            return np.exp(-delta_cost / (k * temperature))


class SARouteOptimizer:
    """
    Simulated annealing route optimizer. With give model and termination criteria, finds optimal route that minimizes
    cost function defined by model.
    """

    def __init__(self,
                 model: SAModel,
                 max_iter: int = 10000,
                 max_iter_without_improvement: int = 2000,
                 min_temperature: float = 1e-12,
                 cost_threshold: float = -np.inf,
                 ):

        self.model = model
        self.max_iter = max_iter
        self.max_iter_without_improvement = max_iter_without_improvement
        self.min_temperature = min_temperature
        self.cost_threshold = cost_threshold

        self.temperatures = []
        self.costs = []
        self.probabilities = []
        self.delta_costs = []
        self.is_accepted = []

    def run(self, init_route: List[int]) -> Tuple[List[int], float]:
        """
        Find optimal route.

        :param init_route: Init guess for route.
        :return: optimal route, route cost
        """

        current_route = init_route.copy()
        best_route = current_route.copy()

        current_cost = self.model.cost(current_route)
        best_cost = self.model.cost(best_route)

        probability, delta_cost = 1, 0
        is_accepted = True
        no_improvement_counter = 0

        for t in range(self.max_iter):

            no_improvement_counter += 1
            temperature = self.model.schedule(t)

            if temperature < self.min_temperature:
                logger.info("Minimum temperature reached. Return solution")
                return best_route, best_cost

            self.temperatures.append(temperature)
            self.costs.append(current_cost)
            self.probabilities.append(probability)
            self.delta_costs.append(delta_cost)
            self.is_accepted.append(is_accepted)

            mutated_route = self.model.mutate(current_route.copy())
            mutated_route_cost = self.model.cost(mutated_route)
            logger.debug(f"Mutated route: {mutated_route}; cost {mutated_route_cost}")
            delta_cost = mutated_route_cost - current_cost

            is_accepted = False
            probability = self.model.probability(delta_cost, temperature)
            if probability >= random.uniform(0.0, 1.0):
                is_accepted = True
                current_route = mutated_route.copy()
                current_cost = mutated_route_cost
                if current_cost < best_cost:
                    best_route = current_route.copy()
                    best_cost = current_cost
                    logger.info(f"Found better solution; round {t}; cost {best_cost}")
                    no_improvement_counter = 0
                    if best_cost < self.cost_threshold:
                        logger.info("Cost reached required threshold value. Return solution.")
                        return best_route, best_cost

            logger.debug(f"Round {t}: temperature {temperature}; cost {current_cost}")

            if no_improvement_counter > self.max_iter_without_improvement:
                logger.info("Max iteration number without improvement reached. Return solution.")
                return best_route, best_cost

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
