from abc import ABC, abstractmethod
from collections.abc import Iterable


class Shape(ABC):
    """An abstract shape class to build images out of."""

    def __init__(self):
        """Initialize a shape."""
        self.__color = None
        self.__position = None
        self.__alpha = None

    @property
    def color(self):
        """Get the color for this shape."""
        return self.__color

    @color.setter
    def color(self, value):
        """Set the color for this shape."""
        if not isinstance(value, int):
            raise ValueError("The color must be an integer.")
        elif value < 0 or value > 255:
            raise ValueError("The color must be between 0 and 255.")
        self.__color = value

    @property
    def alpha(self):
        """Get the alpha for this shape."""
        return self.__alpha

    @alpha.setter
    def alpha(self, value):
        """Set the alpha for this shape."""
        if value < 0 or value > 1:
            raise ValueError("The transparency must be between 0 and 1.")
        self.__alpha = value

    @property
    def position(self):
        """Get the position for this shape."""
        return self.__position

    @position.setter
    def position(self, value):
        """Set the position for this shape."""
        # The position is either an iterable of points, or a point and a size.
        if not isinstance(value, Iterable):
            raise ValueError("The position must be an iterable.")
        self.__position = value

    @abstractmethod
    def mutate(self) -> "Shape":
        """Get a mutated copy of this shape."""

    @abstractmethod
    def recombine(self, other: "Shape") -> "Shape":
        """Recombine two shapes."""

    @abstractmethod
    def add_to_image(self, image):
        """Add this shape to the given image."""

    def __add__(self, other: "Shape") -> "Shape":
        """Recombine two shapes."""
        return self.recombine(other)

    def __invert__(self) -> "Shape":
        """Get a mutated copy of this shape."""
        return self.mutate()

    @abstractmethod
    def __repr__(self):
        """Official string representation of a Shape."""
