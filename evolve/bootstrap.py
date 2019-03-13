import numpy as np

from .utils import CircleDtype, fitness


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
        # The fitnesses of the general population.
        self.general_fitnesses = np.zeros(pop_size)
        self.mutations = np.zeros(pop_size, dtype=CircleDtype)
        # The fitnesses of the mutations.
        self.mutation_fitnesses = np.zeros(pop_size)

        # The image approximated with the best circle from each iteration.
        self.approximation = np.zeros_like(target, dtype="float32")
        # We need a destructible copy of the image to test each circle in each generation.
        self.temp_image = np.zeros_like(target, dtype="float32")
        # The individual this algorithm is building.
        self.individual = np.zeros(circles, dtype=CircleDtype)

    def init_pop(self):
        """Randomly initialize the population of circles."""
        for circle in self.population:
            self.init_circle(circle)

    def init_circle(self, circle):
        """Randomly initialize the given circle.

        :param circle: The circle to initialize.
        :type circle: A single CircleDtype object.
        """
        circle["color"] = np.random.randint(0, 256)
        # TODO: What should the bounds on the circle radii be?
        circle["radius"] = np.random.randint(10, max(self.height, self.width) / 4)
        circle["center"]["x"] = np.random.randint(0, self.width)
        circle["center"]["y"] = np.random.randint(0, self.height)

    def mutate(self):
        """Perform random mutations in the population."""
        raise NotImplementedError

    def evaluate(self):
        """Evaluate the fitnesses of the population."""
        raise NotImplementedError

    def select(self):
        """Perform selection on the combined general and mutation populations."""
        raise NotImplementedError

    def update_approximation(self, circle):
        """Add the given circle to the approximation image."""
        raise NotImplementedError

    def run(self):
        """Run the bootstrap algorithm."""
        for i in range(self.circles):
            self.init_pop()
            for generation in range(self.generations):
                # self.mutate()
                # Update the fitnesses so that selection is possible.
                # self.evaluate()
                # self.select()
                pass

            best = self.population[np.argmin(self.general_fitnesses)]
            self.individual[i] = best
            # self.update_approximation(best)

        return self.individual, self.approximation
