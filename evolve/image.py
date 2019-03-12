import numpy as np


def fitness(image1, image2):
    """Determine how close two images are."""
    assert image1.shape == image2.shape
    height, width = image1.shape

    # Potential overflow issue if the images are uints
    return np.sum(np.abs(image1 - image2.astype(float))) / (height * width)
