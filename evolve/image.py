import numpy as np


def fitness(image1, image2):
    """Determine how close two images are."""
    assert image1.shape == image2.shape
    # TODO: Is normalization by the area necessary?
    return np.sum((image1 - image2) ** 2)
