#!/usr/bin/env python3
import argparse

import imageio
import matplotlib.pyplot as plt
import numpy as np

from evolve import EvolutionaryAlgorithm


def parse_args():
    parser = argparse.ArgumentParser(description="Approximate the given image using an EA.")

    parser.add_argument("--quiet", action="store_true", default=False, help="Quiet all output.")
    parser.add_argument("image", help="The image to approximate.")
    parser.add_argument("--population", "-p", type=int, default=10, help="The population size.")
    parser.add_argument("--individual", "-i", type=int, default=20, help="The individual size.")
    parser.add_argument(
        "--generations", "-g", type=int, default=100, help="The number of generations."
    )
    # TODO: Save the result to an output file.

    return parser.parse_args()


def main(args):
    image = imageio.imread(args.image)

    ea = EvolutionaryAlgorithm(image, pop_size=args.population, ind_size=args.individual)

    fitnesses, individuals = ea.run(generations=args.generations, verbose=not args.quiet)

    if not args.quiet:
        solution = individuals[np.argmin(fitnesses)]

        approximation = np.zeros(image.shape, dtype="uint8")
        ea.compute_image(approximation, solution, 255)

        _, axes = plt.subplots(1, 3)
        axes[0].set_title("Best Approximation")
        axes[0].imshow(approximation, cmap="gray", vmin=0, vmax=255)
        axes[0].axis("off")
        axes[1].set_title("Target Image")
        axes[1].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[1].axis("off")
        axes[2].set_title("Diff")
        axes[2].imshow(image - approximation.as_type("int16"), cmap="gray", vmin=0, vmax=255)
        axes[2].axis("off")
        plt.show()


if __name__ == "__main__":
    main(parse_args())
