import itertools

import numpy as np


# Use float32's because it caused a massive speedup vs uint16's.
CircleCenterDtype = np.dtype([("x", "float32"), ("y", "float32")])
CircleDtype = np.dtype([("radius", "float32"), ("color", "float32"), ("center", CircleCenterDtype)])


def pairwise(iterable):
    """Iterate over the given iterable in pairs."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
