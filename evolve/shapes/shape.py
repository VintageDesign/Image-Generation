from abc import ABC, abstractmethod
from collections.abc import Iterable


class Shape(ABC):
    """An abstract shape class to build images out of."""

    def __init__(self):
        """Initialize a shape."""
        self.__color = None
        self.__position = None

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
    def add_to_image(self, image):
        """Add this shape to the given image."""

    @abstractmethod
    def __repr__(self):
        """Official string representation of a Shape."""
