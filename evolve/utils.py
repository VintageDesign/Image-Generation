import itertools

import numpy as np


# Use float32's because it caused a massive speedup vs uint16's.
CircleCenterDtype = np.dtype([("x", "float32"), ("y", "float32")])
CircleDtype = np.dtype([("radius", "float32"), ("color", "float32"), ("center", CircleCenterDtype)])


def fitness(image1, image2):
    """Determine how close two images are."""
    assert image1.shape == image2.shape
    height, width = image1.shape

    # Potential overflow issue if the images are uints
    return np.sum(np.abs(image1 - image2)) / (height * width)


def pairwise(iterable):
    """Iterate over the given iterable in pairs."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
