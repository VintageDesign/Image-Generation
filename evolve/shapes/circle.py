import numpy as np


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
        a, b = int(circle["center"]["x"]), int(circle["center"]["y"])
        # Produce a circular mask and set its value.
        x, y = np.ogrid[-b : self.height - b, -a : self.width - a]
        mask = x ** 2 + y ** 2 <= int(circle["radius"]) ** 2

        # Average the existing color and the new circle's color
        image[mask] = (image[mask] + int(circle["color"])) / 2

    def update_image(self):
        """Add the given circles to the given image."""
        self.image.fill(self.image.mean())
        for circle in self.circles:
            self.add_circle(self.image, circle)

    def mutate(self) -> "NumpyCircleArray":
        """Mutate the collection of circles."""
        mutation = NumpyCircleArray(self.number, (self.height, self.width))
        np.copyto(mutation.circles, self.circles, casting="no")
        for circle in mutation.circles:
            mutation.mutate_circle(circle)

        return mutation

    def mutate_circle(self, circle):
        """Mutate the given circle."""
        self.mutate_center(circle)
        self.mutate_radius(circle)
        self.mutate_color(circle)

    def mutate_center(self, circle):
        """Mutate the given circle's center."""
        # TODO: Tweak the scale.
        dx, dy = np.random.normal(scale=3, size=2)
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
        dc = np.random.normal(scale=5)
        if dc < 0 and abs(dc) > circle["color"]:
            dc = -dc

        circle["color"] = circle["color"] + dc
