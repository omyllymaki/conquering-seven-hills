import copy
import logging
import random
from typing import List, Callable

import matplotlib.pyplot as plt
import numpy as np

from route import Route

logger = logging.getLogger(__name__)


def exp_schedule(t, max_temperature=1, decay_constant=0.005):
    return max_temperature * np.exp(-decay_constant * t)


def simulated_annealing(cost_matrix: np.ndarray,
                        initial_indices: List[int],
                        mutation_probability: float = 0.1,
                        max_iter: int = 1000,
                        schedule: Callable = exp_schedule,
                        debug_plot: bool = False):
    current_route = Route(initial_indices, cost_matrix)
    best_route = copy.deepcopy(current_route)

    temperatures, costs, ps, delta_costs = [], [], [], []
    p, delta_cost = 0, 0

    for t in range(max_iter):

        temperature = schedule(t)

        if temperature < 1e-12:
            return best_route

        if debug_plot:
            temperatures.append(temperature)
            costs.append(current_route.cost)
            ps.append(p)
            delta_costs.append(delta_cost)
            plt.figure(1)
            plt.subplot(1, 2, 1)
            plt.cla()
            plt.plot(costs)
            plt.subplot(3, 2, 2)
            plt.cla()
            plt.plot(temperatures)
            plt.subplot(3, 2, 4)
            plt.cla()
            plt.plot(ps)
            plt.subplot(3, 2, 6)
            plt.cla()
            plt.plot(delta_costs)
            plt.pause(0.00001)

        mutated_route = copy.deepcopy(current_route)
        for k in range(len(mutated_route)):
            if random.random() < mutation_probability:
                mutated_route.mutate()

        logger.debug(f"Mutated route: {mutated_route}, {mutated_route.cost}")
        delta_cost = mutated_route.cost - current_route.cost

        if delta_cost < 0:
            current_route = copy.deepcopy(mutated_route)
            if current_route.cost < best_route.cost:
                best_route = copy.deepcopy(current_route)
                logger.info(f"Found better solution; cost {best_route.cost}")
        else:
            p = np.exp(-delta_cost / temperature)
            if p > random.uniform(0.0, 1.0):
                current_route = copy.deepcopy(mutated_route)

        logger.debug(f"Round {t}: temperature {temperature}; cost {current_route.cost}")

    return best_route
