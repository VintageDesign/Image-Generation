import numpy as np

from evolve import fitness
from evolve.shapes import NumpyCircleArray


class EvolutionaryAlgorithm:
    """Implements an EA to approximate a given image."""

    def __init__(self, image, pop_size, ind_size):
        """Initialize the EA.

        :param image: The image to approximate
        :type image: A 2d array of uint8's
        :param pop_size: The number of individuals in the population.
        :param ind_size: The number of circles composing an individual.
        """
        self.target = image
        self.height, self.width = image.shape
        self.pop_size = pop_size
        self.ind_size = ind_size

        self.population = np.zeros((pop_size, ind_size), dtype=NumpyCircleArray.CircleDtype)
        self.fitnesses = np.zeros(pop_size)

        self.mutations = np.zeros_like(self.population)
        self.children = None

        # TODO: If we ever attempt to perform parallel computation, each process will need its own
        # image represented by a single individual.
        self.approx = np.zeros(image.shape, dtype="uint8")

    def init_pop(self):
        """Randomly initialize the population."""
        for individual in self.population:
            self.init_individual(individual)

    def init_individual(self, individual):
        """Randomly initialize the given individual.

        :param individual: The individual to initialize.
        :type individual: An array of NumpyCircleArray.CircleDtype objects.
        """
        for circle in individual:
            self.init_circle(circle)

    def init_circle(self, circle):
        """Randomly initialize the given circle.

        :param circle: The circle to initialize.
        :type circle: A single NumpyCircleArray.CircleDtype object.
        """
        circle["color"] = np.random.randint(0, 256, dtype="uint8")
        # TODO: What should the bounds on the circle radii be?
        circle["radius"] = np.random.randint(5, max(self.height, self.width), dtype="uint16")
        circle["center"]["x"] = np.random.randint(0, self.width, dtype="uint16")
        circle["center"]["y"] = np.random.randint(0, self.height, dtype="uint16")

    @staticmethod
    def compute_image(image, individual, fill_color=0):
        """Compute the image represented by the given individual.

        :param image: The preallocated image to fill
        :type image: a (height, width) numpy array of uint8s
        :param individual: The individual representing the image
        :type individual: An array of NumpyCircleArray.CircleDtype objects
        :param fill_color: The image background color.
        :type fill_color: A uint8 between 0 and 255.
        """
        image.fill(fill_color)
        height, width = image.shape
        for circle in individual:
            # NOTE: The type of circle["center"]["x"] is a numpy uint16. However, for whatever
            # reason, a numpy uint16 does *not* play nicely with np.ogrid unless it's the value 0.
            cx, cy, r = int(circle["center"]["x"]), int(circle["center"]["y"]), circle["radius"]

            x, y = np.ogrid[-cy : height - cy, -cx : width - cx]
            mask = x ** 2 + y ** 2 <= r ** 2
            image[mask] = (image[mask] + circle["color"]) / 2

    def evaluate(self):
        """Update the population fitnesses."""
        # TODO: This is a good candidate for parallelism.
        for i in range(self.pop_size):
            individual = self.population[i]
            # TODO: Fill with the mean color of the target image?
            self.compute_image(self.approx, individual, fill_color=self.target.mean())
            self.fitnesses[i] = fitness(self.approx, self.target)

    def perturb_radius(self, circle, scale=5):
        """Perturb the radius of the given circle."""
        dr = np.random.normal(scale=scale)
        # Avoid overflows.
        if dr < 0 and abs(dr) > circle["radius"]:
            dr = -dr
        circle["radius"] = circle["radius"] + dr

    def perturb_color(self, circle, scale=5):
        """Perturb the color of the given circle."""
        dc = np.random.normal(scale=scale)
        # Avoid overflows.
        if dc < 0 and -dc > circle["color"]:
            dc = -dc
        circle["color"] += dc

    def perturb_center(self, circle, scale=5):
        """Perturb the center of the given circle."""
        dx, dy = np.random.normal(scale=scale, size=2)
        circle["center"]["x"] = (circle["center"]["x"] + dx) % self.width
        circle["center"]["y"] = (circle["center"]["y"] + dy) % self.height

    def mutate_individual(self, individual):
        """Mutate the given individual in place."""
        for circle in individual:
            self.perturb_radius(circle)
            self.perturb_color(circle)
            self.perturb_center(circle)

    def mutate(self):
        """Mutate each individual in the population."""
        np.copyto(self.mutations, self.population)
        for mutant in self.mutations:
            self.mutate_individual(mutant)

    def reproduce(self):
        """Reproduce the individuals in the population."""
        # TODO: Should the population be sorted by fitness?
        raise NotImplementedError

    def select(self):
        """Select the individuals who survive."""
        raise NotImplementedError

    def run(self, generations=100):
        """Run the EA.

        :param generations: The number of generations to run the EA for
        :returns: The best individuals over the total runtime of the algorithm.
        :rtype: An array of EvolutionaryAlgorithm.HistoryDtype objects.
        """
        self.init_pop()
        self.evaluate()

        fitnesses = np.zeros(generations)
        individuals = np.zeros((generations, self.ind_size), dtype=NumpyCircleArray.CircleDtype)
        for gen in range(generations):
            # self.reproduce()
            self.mutate()
            # self.select()

            self.evaluate()
            best = np.argmax(self.fitnesses)
            fitnesses[gen] = self.fitnesses[best]
            individuals[gen] = self.population[best]

        return fitnesses, individuals
