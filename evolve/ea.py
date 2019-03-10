import numpy as np

from evolve.shapes import NumpyCircleArray


class EvolutionaryAlgorithm:
    """Implements an EA to approximate a given image."""

    def __init__(self, shape, pop_size, ind_size):
        """Initialize the EA.

        :param number: The number of circles to use
        :param shape: The shape of the approximated image
        :type shape: a (height, width) tuple of ints
        :param pop_size: The number of individuals in the population.
        :param ind_size: The number of circles composing an individual.
        """
        self.height, self.width = shape
        self.pop_size = pop_size
        self.ind_size = ind_size

        self.population = np.zeros((pop_size, ind_size), dtype=NumpyCircleArray.CircleDtype)
        # TODO: If we ever attempt to perform parallel computation, each process will need its own
        # image represented by a single individual.
        self.image = np.zeros(shape, dtype="uint8")

        self.init_pop()

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
            mask = x**2 + y**2 <= r**2
            image[mask] = (image[mask] + circle["color"]) / 2
