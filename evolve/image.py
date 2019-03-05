import numpy as np


def from_file(filename):
    """Get a grayscale image from a file."""
    # TODO: Look up how to handle image filetypes...
    raise NotImplementedError


def save_image(image, filename):
    """Save a grayscale image to a file."""
    # TODO: Look up how to handle image filetypes...
    raise NotImplementedError


def fitness(image1, image2):
    """Determine how close two images are."""
    assert image1.shape == image2.shape
    height, width = image1.shape
    # TODO: Is the normalization by the area necessary?
    return np.sum((image1 - image2) ** 2) / (height * width)
