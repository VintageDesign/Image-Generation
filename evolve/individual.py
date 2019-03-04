import numpy as np


class Individual:
    """A collection of Shapes, making up an image."""

    def __init__(self, number, ctor, size):
        """Initialize an Individual randomly.

        :param number: The number of shapes to use.
        :param ctor: The constructor for the shapes.
        :type ctor: An constructor for an evolve.shapes.Shape
        :param size: The image size to approximate.
        :type size: A (width, height) tuple.
        """
        self.shapes = [ctor(size) for _ in range(number)]
        self.image = np.zeros(size)

        # Add each shape to the represented image.
        for shape in self.shapes:
            shape.add_to_image(self.image)

    def mutate(self):
        """Mutate this individual."""
        raise NotImplementedError

    def recombine(self, other: "Individual") -> "Individual":
        """Recombine with another individual."""
        raise NotImplementedError

    def fitness(self, image):
        """Determine how closely this individual approximates the given image."""
        raise NotImplementedError

    def to_image(self):
        """Convert this individual to an image array."""
        raise NotImplementedError
