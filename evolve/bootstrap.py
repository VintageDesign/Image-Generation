import numpy as np

from .utils import CircleDtype, fitness


class BootstrapAlgorithm:
    """A quick approximation algorithm built one circle at a time.

    Approximate an image one circle at a time. For each circle desired, initialize a population
    of random circles. Then for some number of generations, mutate, recombine, and select that
    population.

    Each individual is a single circle.

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
        self.approximation.fill(255)
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
        circle["color"] = np.random.randint(-255, 256)
        # TODO: What should the bounds on the circle radii be?
        circle["radius"] = np.random.randint(10, max(self.height, self.width) / 4)
        circle["center"]["x"] = np.random.randint(0, self.width)
        circle["center"]["y"] = np.random.randint(0, self.height)

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

    def mutate(self, scale):
        """Perform random mutations in the population."""
        np.copyto(self.mutations, self.population)
        for mutant in self.mutations:
            self.perturb_radius(mutant, scale)
            self.perturb_color(mutant, scale)
            self.perturb_center(mutant, scale)

    def evaluate(self):
        """Evaluate the fitnesses of the population."""
        for i, (general, mutation) in enumerate(zip(self.population, self.mutations)):
            cx, cy, r = int(general["center"]["x"]), int(general["center"]["y"]), general["radius"]
            x, y = np.ogrid[-cy : self.height - cy, -cx : self.width - cx]
            mask = x ** 2 + y ** 2 <= r ** 2
            approx = self.approximation + mask * general["color"]

            self.general_fitnesses[i] = fitness(self.target, approx)

            cx, cy, r = (
                int(mutation["center"]["x"]),
                int(mutation["center"]["y"]),
                mutation["radius"],
            )
            x, y = np.ogrid[-cy : self.height - cy, -cx : self.width - cx]
            mask = x ** 2 + y ** 2 <= r ** 2
            approx = self.approximation + mask * mutation["color"]
            self.mutation_fitnesses[i] = fitness(self.target, approx)

    def select(self):
        """Perform selection on the combined general and mutation populations."""
        joint = np.concatenate((self.population, self.mutations))
        fitnesses = np.concatenate((self.general_fitnesses, self.mutation_fitnesses))
        indices = np.argsort(fitnesses)
        self.population = joint[indices][: self.pop_size]
        self.general_fitnesses = fitnesses[indices][: self.pop_size]

    def add_to_image(self, image, circle):
        """Add the given circle to the given image."""
        cx, cy, r = int(circle["center"]["x"]), int(circle["center"]["y"]), circle["radius"]
        x, y = np.ogrid[-cy : self.height - cy, -cx : self.width - cx]
        image[x ** 2 + y ** 2 <= r ** 2] += circle["color"]

    def run(self):
        """Run the bootstrap algorithm."""
        for i in range(self.circles):
            self.init_pop()
            for generation in range(self.generations):
                self.mutate(scale=1.0)
                # Update the fitnesses so that selection is possible.
                self.evaluate()
                self.select()

            best = self.population[np.argmin(self.general_fitnesses)]
            self.individual[i] = best
            self.add_to_image(self.approximation, best)

        return self.individual, self.approximation
