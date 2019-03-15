#!/usr/bin/env python3
import argparse
import itertools
from multiprocessing import Pool

import imageio
import matplotlib.pyplot as plt
import numpy as np

from evolve import BootstrapAlgorithm


def worker(args):
    """Find an approximation in a separate process."""
    seed, (target, circles, pop_size, generations) = args
    ba = BootstrapAlgorithm(target, circles, pop_size, generations, seed)
    return ba.run()


def average(filename, circles, layers, pop_size, generations):
    """Average multiple runs of the BootstrapAlgorithm to form a composite image.

    :param filename: The image filename to approximate.
    :param circles: The number of circles to use in each layer.
    :param layers: The number of layers to use.
    :param pop_size: The population size to use.
    :param generations: The number of generations to use.
    """
    target = imageio.imread(filename).astype("float32")
    approximate = np.zeros_like(target, dtype="float32")

    # Use a different seed for each process to avoid results exactly the same as each other.
    seeds = np.random.randint(low=np.iinfo(np.uint32).max, size=layers)
    with Pool() as pool:
        results = pool.map(
            worker,
            zip(seeds, itertools.repeat((target, circles, pop_size, generations), times=layers)),
        )

    for _, image in results:
        approximate = approximate + image

    # Need to normalize because we're summing all of the images together.
    approximate = approximate / len(results)

    _, axes = plt.subplots(1, 3)
    axes[0].set_title("Best Approximation")
    axes[0].imshow(approximate, cmap="gray", vmin=0, vmax=255)
    axes[0].axis("off")
    axes[1].set_title("Target Image")
    axes[1].imshow(target, cmap="gray", vmin=0, vmax=255)
    axes[1].axis("off")
    axes[2].set_title("Diff")
    axes[2].imshow(np.abs(target - approximate), cmap="gray", vmin=0, vmax=255)
    axes[2].axis("off")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Form a composite image using the Bootstrap method"
    )
    parser.add_argument("image", help="The image to approximate")
    parser.add_argument("--layers", "-l", type=int, help="The number of layers to use")
    parser.add_argument("--circles", "-c", type=int, help="The number of circles to use")
    parser.add_argument("--population", "-p", type=int, help="The population size to use")
    parser.add_argument("--generations", "-g", type=int, help="The number of generations to use")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    average(args.image, args.circles, args.layers, args.population, args.generations)
