#!/usr/bin/env python3
import argparse

import imageio
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from evolve import BootstrapAlgorithm, CombinedAlgorithm, EvolutionaryAlgorithm


def parse_args():
    parser = argparse.ArgumentParser(description="Approximate the given image using an EA.")

    parser.add_argument("--quiet", action="store_true", default=False, help="Quiet all output.")
    parser.add_argument("--population", "-p", type=int, default=10, help="The population size.")
    parser.add_argument(
        "--circles",
        "-c",
        type=int,
        default=20,
        help="The number of circles used to approximate the image.",
    )
    parser.add_argument(
        "--generations", "-g", type=int, default=100, help="The number of generations."
    )
    parser.add_argument("image", help="The image to approximate.")
    parser.add_argument("output", nargs="?", default="", help="The filename to save the output to.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--bootstrap",
        action="store_true",
        default=False,
        help="Use the bootstrap method to build a quick approximation.",
    )
    group.add_argument(
        "--ea",
        action="store_true",
        default=False,
        help="The default traditional, slow EA approach.",
    )
    group.add_argument(
        "--combined", action="store_true", default=False, help="A combined approach."
    )

    return parser.parse_args()


def main(args):
    target = np.asarray(Image.open(args.image).convert('L')).astype("float32")
    approximation = np.zeros_like(target, dtype="float32")


    if args.ea:
        ea = EvolutionaryAlgorithm(target, pop_size=args.population, ind_size=args.circles)
        fitnesses, individuals = ea.run(generations=args.generations, verbose=not args.quiet)
        solution = individuals[np.argmin(fitnesses)]
        print("best solution:", solution)
        ea.compute_image(approximation, solution, 255)

        plt.title("The best individual's fitness over time.")
        plt.plot(fitnesses)
        if args.output:
            plt.savefig(args.output + ".fitness.png")
        if not args.quiet:
            plt.show()

    elif args.bootstrap:
        ba = BootstrapAlgorithm(target, args.circles, args.population, args.generations)
        solution, approximation = ba.run()
        print("best solution:", solution)
    elif args.combined:
        ca = CombinedAlgorithm(target, args.circles, args.population, args.generations)
        individuals, fitnesses = ca.run()
        solution = individuals[np.argmin(fitnesses)]
        print("best solution:", solution)
        ca.compute_image(approximation, solution, fill_color=255)

        plt.title("The best individual's fitness over time.")
        plt.plot(fitnesses)
        if args.output:
            plt.savefig(args.output + ".fitness.png")
        if not args.quiet:
            plt.show()

    _, axes = plt.subplots(1, 3)
    axes[0].set_title("Best Approximation")
    axes[0].imshow(approximation, cmap="gray", vmin=0, vmax=255)
    axes[0].axis("off")
    axes[1].set_title("Target Image")
    axes[1].imshow(target, cmap="gray", vmin=0, vmax=255)
    axes[1].axis("off")
    axes[2].set_title("Diff")
    axes[2].imshow(np.abs(target - approximation), cmap="gray", vmin=0, vmax=255)
    axes[2].axis("off")

    if args.output:
        plt.savefig(args.output)

    if not args.quiet:
        plt.show()


if __name__ == "__main__":
    main(parse_args())
