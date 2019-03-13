import itertools
from multiprocessing import Pool

import numpy as np

from evolve import BootstrapAlgorithm

from .utils import CircleDtype, fitness, pairwise


class CombinedAlgorithm:
    """A combined evolutionary approach to approximate a given image using circles.

    Use the bootstrap method to initialize the population with decent approximations. The run an
    evolutionary algorithm on this population, implementing recombination, mutation, and selection.

    Each individual is a collection of circles.

    TODO: I think we might have good success if we *add* new circles to each of the individuals,
    then run both recombination and mutation on them.

    TODO: It might also work to run just recombination, and possibly return several individuals
    superimposed on top of each other.
    """

    def __init__(self, target, circles, pop_size, generations):
        """Initialize the combined algorithm.

        :param target: The desired target image.
        :type target: A 2D array of floats.
        :param circles: The number of circles to use in the approximation.
        :param pop_size: The number of individuals in the population.
        :param generations: The number of generations to select and breed the population.
        """
        self.height, self.width = target.shape
        self.target = target
        self.circles = circles
        self.pop_size = pop_size
        self.generations = generations

        self.population = np.zeros((pop_size, circles), dtype=CircleDtype)
        self.children = np.zeros((pop_size, circles), dtype=CircleDtype)

        self.fitnesses = np.zeros(pop_size)
        self.child_fitnesses = np.zeros(pop_size)
        # Leave the population unsorted, but keep track of the indices that would sort by fitness
        self.sorted_indices = np.arange(pop_size)

        self.approximation = np.zeros_like(target, dtype="float32")

        self.proc_pool = Pool()

    @staticmethod
    def worker(args):
        """Find an approximation in a separate process."""
        target, circles, pop_size, generations = args
        ba = BootstrapAlgorithm(target, circles, pop_size, generations)
        return ba.run()

    def init_pop(self, pop_size, generations):
        """Use the bootstrap method to initialize a population of individuals.

        Performs the initialization in parallel using however many cores are available.

        :param pop_size: The bootstrap population size. (Number of circles)
        :param generations: The bootstrap generation length. (Number of iterations)
        """
        print("initializing population... 0.00%", end="", flush=True)
        for i, result in enumerate(
            self.proc_pool.imap_unordered(
                self.worker,
                itertools.repeat(
                    (self.target, self.circles, pop_size, generations), times=self.pop_size // 2
                ),
            )
        ):
            print(f"\rinitializing population... {i / (self.pop_size // 2):.2f}%", end="")
            individual, _ = result
            self.population[i] = individual

        for i, (mom, dad) in enumerate(pairwise(self.population[: self.pop_size // 2])):
            children = self.crossover(mom, dad)
            self.population[self.pop_size // 2 + i] = children[np.random.randint(2)]

        print(" done.")

    def crossover(self, mom, dad):
        """Breed two individuals via the traditional crossover.

        Take the first half of mom and splice with dad, and vice versa.
        """
        kid1 = np.concatenate((mom[: self.circles // 2], dad[self.circles // 2 :]))
        kid2 = np.concatenate((mom[self.circles // 2 :], dad[: self.circles // 2]))
        return kid1, kid2

    @staticmethod
    def compute_image(image, individual, fill_color):
        """Compute the image represented by the given individual."""
        image.fill(fill_color)
        height, width = image.shape
        for circle in individual:
            cx, cy, r = int(circle["center"]["x"]), int(circle["center"]["y"]), circle["radius"]
            # NOTE: ogrid is much cheaper than meshgrid.
            x, y = np.ogrid[-cy : height - cy, -cx : width - cx]
            image[x ** 2 + y ** 2 <= r ** 2] += circle["color"]

    def evaluate(self):
        """Evaluate the population."""
        for i, individual in enumerate(self.population):
            self.compute_image(self.approximation, individual, fill_color=255)
            self.fitnesses[i] = fitness(self.approximation, self.target)

        self.sorted_indices = np.argsort(self.fitnesses)

    def recombine(self):
        """Recombine the population."""
        for i, (mom, dad) in enumerate(pairwise(self.population)):
            children = self.crossover(mom, dad)
            self.children[i] = children[np.random.randint(2)]
        # Keep the best third of the population, and replace the rest with 2/3 of the new children.
        self.population[self.sorted_indices][self.pop_size // 3 :] = self.children[
            self.pop_size // 3 :
        ]

    def run(self):
        """Run the combined evolutionary algorithm.

        This function returns a tuple (individuals, fitnesses) of the best individuals and their
        fitness from each generation.
        """
        self.init_pop(pop_size=10, generations=100)

        # The best individual of each generation.
        best_individuals = np.zeros((self.generations, self.circles), dtype=CircleDtype)
        best_fitnesses = np.zeros(self.generations)

        for gen in range(self.generations):
            self.recombine()
            # TODO: Determine if the initial population is good enough that we don't need mutations.
            self.evaluate()

            best = np.argmin(self.fitnesses)
            best_individuals[gen] = self.population[best]
            best_fitnesses[gen] = self.fitnesses[best]
            print(
                f"\rgeneration: {gen} best fit: {self.fitnesses.min()} worst fit: {self.fitnesses.max()}",
                end="",
            )

        print("")
        return best_individuals, best_fitnesses
