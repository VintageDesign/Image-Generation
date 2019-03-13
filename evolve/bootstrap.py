import numpy as np

from .utils import CircleDtype


class BootstrapAlgorithm:
    """A quick approximation algorithm built one circle at a time.

    Approximate an image one circle at a time. For each circle desired, initialize a population
    of random circles. Then for some number of generations, mutate, recombine, and select that
    population.

    At the end of the generational cycle, add the best circle from the population to the current
    approximation and move on to the next desired circle.

    This algorithm is intended to kickstart another, more traditional, evolutionary algorithm using
    the overall approximation this algorithm outputs as its own individuals to breed and select.
    """

    def __init__(self, target, circles, pop_size, generations):
        """Initialize the bootstrap algorithm.

        :param target: The desired target image.
        :type target: A 2D array of floats.
        :param circles: The number of circles the use in the approximation.
        :param pop_size: The number of circles in the population.
        :param generations: The number of generations used to find the best circle.
        """
        self.height, self.width = target.shape
        self.target = target
        self.circles = circles
        self.pop_size = pop_size
        self.generations = generations

        self.population = np.zeros(pop_size, dtype=CircleDtype)
        self.mutations = np.zeros_like(self.population, dtype=CircleDtype)

        # The image approximated with the best circle from each iteration.
        self.approximation = np.zeros_like(target, dtype="float32")
        # We need a destructible copy of the image to test each circle in each generation.
        self.temp_image = np.zeros_like(target, dtype="float32")
        # The individual this algorithm is building.
        self.individual = np.zeros(circles, dtype=CircleDtype)

    def run(self):
        """Run the bootstrap algorithm."""
        for circle in self.individual:
            pass

        return self.individual, self.approximation
