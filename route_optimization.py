import copy
import logging
import random
from typing import List, Callable

import matplotlib.pyplot as plt
import numpy as np

from route import Route

logger = logging.getLogger(__name__)


def exp_schedule(t, max_temperature=1.0, decay_constant=0.005):
    return max_temperature * np.exp(-decay_constant * t)


class Optimizer:

    def __init__(self, cost_matrix: np.ndarray,
                 mutation_probability: float = 0.1,
                 max_iter: int = 1000,
                 min_temperature: float = 1e-12,
                 cost_threshold: float = -np.inf,
                 schedule: Callable = exp_schedule,
                 debug_plot: bool = False):

        self.cost_matrix = cost_matrix
        self.mutation_probability = mutation_probability
        self.max_iter = max_iter
        self.min_temperature = min_temperature
        self.cost_threshold = cost_threshold
        self.schedule = schedule
        self.debug_plot = debug_plot

        self.temperatures = []
        self.costs = []
        self.probabilities = []
        self.delta_costs = []

    def run(self, initial_indices: List[int]):
        current_route = Route(initial_indices, self.cost_matrix)
        best_route = copy.deepcopy(current_route)

        p, delta_cost = 0, 0

        for t in range(self.max_iter):

            temperature = self.schedule(t)

            if temperature < self.min_temperature:
                logger.info("Minimum temperature reached. Return solution")
                self.plot()
                return best_route

            self.temperatures.append(temperature)
            self.costs.append(current_route.cost)
            self.probabilities.append(p)
            self.delta_costs.append(delta_cost)

            mutated_route = copy.deepcopy(current_route)
            for k in range(len(mutated_route)):
                if random.random() < self.mutation_probability:
                    mutated_route.mutate()

            logger.debug(f"Mutated route: {mutated_route}, {mutated_route.cost}")
            delta_cost = mutated_route.cost - current_route.cost

            if delta_cost < 0:
                current_route = copy.deepcopy(mutated_route)
                if current_route.cost < best_route.cost:
                    best_route = copy.deepcopy(current_route)
                    logger.info(f"Found better solution; round {t}; cost {best_route.cost}")
                    if best_route.cost < self.cost_threshold:
                        logger.info("Cost reached required threshold value. Return solution.")
                        self.plot()
                        return best_route
            else:
                p = np.exp(-delta_cost / temperature)
                if p > random.uniform(0.0, 1.0):
                    current_route = copy.deepcopy(mutated_route)

            logger.debug(f"Round {t}: temperature {temperature}; cost {current_route.cost}")

        logger.info("Max iteration number reached. Return solution.")
        self.plot()
        return best_route

    def plot(self):
        if self.debug_plot:
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
