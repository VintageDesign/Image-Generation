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

        self.__center = 0, 0
        self.__radius = 0

        self.center = center
        self.radius = radius
        self.color = color

    @property
    def center(self):
        """Get the circle's center."""
        return self.__center

    @center.setter
    def center(self, value):
        """Set the circle's center."""
        a, b = value
        x, y = self.__center
        width, height = self.size
        if 0 < a < width:
            x = a
        if 0 < b < height:
            y = b
        self.__center = x, y

    @property
    def radius(self):
        """Get the circle's radius."""
        return self.__radius

    @radius.setter
    def radius(self, value):
        """Set the circle's radius."""
        width, height = self.size
        if 0 < value < min(width, height) / 2:
            self.__radius = value

    def add_to_image(self, image: np.ndarray):
        """Add this circle to the given image."""
        # Get the circle center and radius
        a, b = self.center

        # Produce a mask the same size as the image.
        height, width = image.shape
        y, x = np.ogrid[-a : height - a, -b : width - b]
        # Draw a circle in the mask.
        mask = x ** 2 + y ** 2 <= self.radius ** 2

        # Average the circle color with the existing image.
        image[mask] = (image[mask] + self.color) / 2

    def perturb_center(self):
        """Perturb this circle's center."""
        dx, dy = np.random.normal(scale=min(*self.size) / 10, size=2)
        self.center = self.center[0] + int(dx), self.center[1] + int(dy)

    def perturb_color(self):
        """Perturb this circle's color."""
        self.color += int(np.random.normal(scale=20))

    def perturb_radius(self):
        """Perturb this circle's radius."""
        self.radius += int(np.random.normal(scale=min(*self.size) / 10))

    def perturb(self):
        """Perturb this circle's position and color."""
        self.perturb_center()
        self.perturb_radius()
        self.perturb_color()

    def __repr__(self):
        """Official string representation of a Circle."""
        return f"Circle(center={self.center}, radius={self.radius}, color={self.color})"
