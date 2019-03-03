from collections.abc import Iterable

import numpy as np

from evolve.shapes import Shape


class Circle(Shape):
    """A grayscale circle."""

    def __init__(self, center, radius, color, alpha):
        """Initialize the circle.

        :param center: The center of the circle.
        :type center: a 2-tuple
        :param radius: The radius of the circle.
        :type radius: float
        :param color: The color of the circle.
        :type color: uint8_t
        :param alpha: The circle transparency
        :type alpha: float from 0 to 1
        """
        super().__init__()

        if not isinstance(center, Iterable):
            raise ValueError("The center must be an iterable.")
        elif len(center) != 2:
            raise ValueError("The center must be an iterable of length 2.")
        # TODO: This may not be the right choice. If we end up implementing evolution with triangles
        # this would be a list of the vertices, but for a circle I don't think the size should be
        # considered a part of the "position".
        self.position = (center, radius)
        self.color = color
        self.alpha = alpha

    def mutate(self) -> "Circle":
        """Get a mutated copy of this circle."""
        raise NotImplementedError("Mutation has not been implemented yet.")

    def recombine(self, other: "Circle") -> "Circle":
        """Recombine two circles."""
        # Hmmmm, kinky...
        raise NotImplementedError("Recombination has not been implemented yet.")

    def add_to_image(self, image: np.ndarray):
        """Add this circle to the given image."""
        # Get the circle center and radius
        a, b = self.position[0]
        r = self.position[1]

        # Produce a mask the same size as the image.
        height, width = image.shape
        y, x = np.ogrid[-a : height - a, -b : width - b]
        # Draw a circle in the mask.
        mask = x ** 2 + y ** 2 <= r ** 2

        # Average the circle color with the existing image.
        image[mask] = (image[mask] + self.color) / 2

    def __repr__(self):
        """Official string representation of a Circle."""
        c, r = self.position
        return f"Circle(center={c}, radius={r}, color={self.color}, alpha={self.alpha})"
