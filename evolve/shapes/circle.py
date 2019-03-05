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

        height, width = size

        if radius is None:
            radius = np.random.randint(2, max(width, height) / 2)
        if center is None:
            w = np.random.randint(0, width + 1)
            h = np.random.randint(0, height + 1)
            center = (h, w)
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
        height, width = self.size
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
        height, width = self.size
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


class NumpyCircleArray:
    """A thin wrapper around an array of numpy CircleDtype objects.

    This class should be used because it attempts to use numpy arrays for storage
    rather than normal Python lists of ordinary objects. Thus being more efficiency-aware.
    """

    # Use uint16_t's for the centers, because we won't be working that *that* large of images.
    CircleCenterDtype = np.dtype([("x", "uint16"), ("y", "uint16")])
    CircleDtype = np.dtype(
        [("radius", "uint16"), ("color", "uint8"), ("center", CircleCenterDtype)]
    )

    def __init__(self, number, shape, random=False):
        """Create an array of circles.

        :param number: The number of circles to create
        :type number: int
        :param shape: The shape of the approximated image
        :type shape: A 2-tuple of ints
        :param random: Generate random circles, defaults to False
        :type random: bool, optional
        """
        self.number = number
        self.height, self.width = shape

        # Allocate all the memory.
        self.circles = np.zeros(number, dtype=self.CircleDtype)
        self.image = np.zeros(shape, dtype="uint8")

        if random:
            self.init_circles()

        self.update_image()

    def init_circle(self, circle):
        """Randomly initialize the given circle."""
        circle["color"] = np.random.randint(0, 256, dtype="uint8")
        circle["radius"] = np.random.randint(0, min(self.height, self.width) / 4, dtype="uint16")
        circle["center"]["x"] = np.random.randint(0, self.width, dtype="uint16")
        circle["center"]["y"] = np.random.randint(0, self.height, dtype="uint16")

    def init_circles(self):
        """Randomly initialize the given array of circles."""
        for circle in self.circles:
            self.init_circle(circle)

    def add_circle(self, image, circle):
        """Add the given circle to the given image."""
        # TODO: For some stupid reason, swapping x and y below has no effect,
        # but swapping a and b does... Figure out why.
        b, a = int(circle["center"]["x"]), int(circle["center"]["y"])
        # Produce a circular mask and set its value.
        x, y = np.ogrid[-a : self.width - a, -b : self.height - b]
        mask = x ** 2 + y ** 2 <= circle["radius"] ** 2

        # Average the existing color and the new circle's color
        image[mask] = (image[mask] + circle["color"]) / 2

    def update_image(self):
        """Add the given circles to the given image."""
        self.image.fill(255)
        for circle in self.circles:
            self.add_circle(self.image, circle)

    def mutate(self):
        """Mutate the collection of circles."""
        for circle in self.circles:
            self.mutate_circle(circle)

    def mutate_circle(self, circle):
        """Mutate the given circle."""
        self.mutate_center(circle)
        self.mutate_radius(circle)
        self.mutate_color(circle)

    def mutate_center(self, circle):
        """Mutate the given circle's center."""
        # TODO: Tweak the scale.
        dx, dy = np.random.normal(scale=5, size=2)
        circle["center"]["x"] = (circle["center"]["x"] + dx) % self.width
        circle["center"]["y"] = (circle["center"]["y"] + dy) % self.width

    @staticmethod
    def mutate_radius(circle):
        """Mutate the given circle's radius."""
        # TODO: Tweak the scale.
        dr = np.random.normal(scale=5)

        # Avoid overflows.
        if dr < 0 and abs(dr) > circle["radius"]:
            dr = -dr

        circle["radius"] = circle["radius"] + dr

    @staticmethod
    def mutate_color(circle):
        """Mutate the given circle's color."""
        # TODO: Tweak the scale.
        dc = np.random.normal(scale=10)
        if dc < 0 and abs(dc) > circle["color"]:
            dc = -dc

        circle["color"] = circle["color"] + dc
