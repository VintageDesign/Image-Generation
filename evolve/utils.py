import itertools


def pairwise(iterable):
    """Iterate over the given iterable in pairs."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
