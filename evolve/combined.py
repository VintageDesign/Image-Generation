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
        self.fitnesses = np.zeros(pop_size)

        self.approximation = np.zeros_like(target, dtype="float32")

        self.proc_pool = Pool()

    @staticmethod
    def worker(args):
        """Find an approximation in a separate process."""
        seed, (target, circles, pop_size, generations) = args
        ba = BootstrapAlgorithm(target, circles, pop_size, generations, seed)
        return ba.run()

    def init_pop(self, pop_size, generations):
        """Use the bootstrap method to initialize a population of individuals.

        Performs the initialization in parallel using however many cores are available.

        :param pop_size: The bootstrap population size. (Number of circles)
        :param generations: The bootstrap generation length. (Number of iterations)
        """
        # Use a different seed for each process to avoid results exactly the same as each other.
        seeds = np.random.randint(low=np.iinfo(np.uint32).max, size=self.pop_size)
        print("initializing population... 0%", end="", flush=True)
        for i, result in enumerate(
            self.proc_pool.imap_unordered(
                self.worker,
                zip(
                    seeds,
                    itertools.repeat(
                        (self.target, self.circles, pop_size, generations), times=self.pop_size
                    ),
                ),
            )
        ):
            print(f"\rinitializing population... {100 * i // self.pop_size}%", end="")
            individual, _ = result
            self.population[i] = individual
        print(" done.")

    @staticmethod
    def average(circle1, circle2):
        """Average the two given circles."""
        result = np.zeros_like(circle1, dtype=CircleDtype)
        result["color"] = (circle1["color"] + circle2["color"]) / 2
        result["radius"] = (circle1["radius"] + circle2["radius"]) / 2
        result["center"]["x"] = (circle1["center"]["x"] + circle2["center"]["x"]) / 2
        result["center"]["y"] = (circle1["center"]["y"] + circle2["center"]["y"]) / 2

        return result

    def crossover(self, mom, dad):
        """Breed two individuals via the traditional crossover.

        Take the first half of mom and splice with dad.
        """
        # return np.concatenate((mom[: self.circles // 2], dad[self.circles // 2 :]))
        child = np.zeros_like(mom, dtype=CircleDtype)
        for i in range(self.circles):
            child[i] = self.average(mom[i], dad[i])

        return child

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

        self.population = self.population[np.argsort(self.fitnesses)]

    def perturb_radius(self, circle, scale):
        """Perturb the radius of the given circle."""
        dr = np.random.normal(scale=scale)
        circle["radius"] = max(dr * circle["radius"] + circle["radius"], 1)

    def perturb_color(self, circle, scale):
        """Perturb the color of the given circle."""
        dc = np.random.normal(scale=scale)
        circle["color"] = max(min(dc * circle["color"] + circle["color"], 255), -255)

    def perturb_center(self, circle, scale):
        """Perturb the center of the given circle."""
        dx, dy = np.random.normal(scale=scale, size=2)
        circle["center"]["x"] = max(
            min(dx * circle["center"]["x"] + circle["center"]["x"], self.width), 0
        )
        circle["center"]["y"] = max(
            min(dy * circle["center"]["y"] + circle["center"]["y"], self.height), 0
        )

    def mutate_individual(self, individual, scale):
        """Mutate the given individual in place."""
        for circle in individual:
            self.perturb_radius(circle, scale)
            self.perturb_color(circle, scale)
            self.perturb_center(circle, scale)

    def mutate(self, population, scale):
        """Mutate each individual in the population."""
        for mutant in population:
            self.mutate_individual(mutant, scale)

    def breed(self):
        """Recombine and mutate the population."""
        children = np.zeros((self.pop_size - 1, self.circles), dtype=CircleDtype)

        # Pairwise breed and mutate everyone
        for i, (mom, dad) in enumerate(pairwise(np.random.permutation(self.population))):
            child = self.crossover(mom, dad)
            children[i] = child

        mutants = children.copy()

        self.mutate(mutants, scale=0.2)

        elite, offspring = (int(0.1 * self.pop_size), int(0.6 * self.pop_size))

        self.population[elite:offspring] = children[: offspring - elite]
        self.population[offspring:] = mutants[: len(self.population[offspring:])]

    def run(self):
        """Run the combined evolutionary algorithm.

        This function returns a tuple (individuals, fitnesses) of the best individuals and their
        fitness from each generation.
        """
        self.init_pop(pop_size=10, generations=20)
        self.evaluate()

        # The best individual of each generation.
        best_individuals = np.zeros((self.generations, self.circles), dtype=CircleDtype)
        best_fitnesses = np.zeros(self.generations)

        for gen in range(self.generations):
            # Handle recombination, mutation, and selection.
            self.breed()
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
