from collections.abc import Iterable

import numpy as np

from evolve.shapes import Shape


class Circle(Shape):
    """A grayscale circle."""

    def __init__(self, size, center=None, radius=None, color=None):
        """Initialize the circle.

        :param size: The dimensions of the approximated image.
        :type size: A 2-tuple of ints
        :param center: The center of the circle. If None, the center is randomly generated.
        :param center: A 2-tuple of ints, optional
        :param radius: The radius of the circle. If None, the radius is randomly generated.
        :param radius: int, optional
        :param color: The color of the circle. If None, the color is randomly generated.
        :param color: An integer from 0 to 255, optional
        """
        super().__init__(size)

        if center is not None and not isinstance(center, Iterable):
            raise ValueError("The center must be an iterable.")
        if center is not None and len(center) != 2:
            raise ValueError("The center must be an iterable of length 2.")

        width, height = size

        if radius is None:
            radius = np.random.randint(2, max(width, height) / 2)
        if center is None:
            w = np.random.randint(0, width + 1)
            h = np.random.randint(0, height + 1)
            center = (w, h)
        if color is None:
            color = np.random.randint(0, 256)

        self.position = (center, radius)
        self.color = color

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
        return f"Circle(center={c}, radius={r}, color={self.color})"
