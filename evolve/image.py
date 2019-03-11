import numpy as np


def fitness(image1, image2):
    """Determine how close two images are."""
    assert image1.shape == image2.shape
    height, width = image1.shape
    return 1 / (np.sum((image1 - image2) ** 2) / (height * width))
