#!/usr/bin/env python3
import imageio
import matplotlib.pyplot as plt
import numpy as np

from evolve import EvolutionaryAlgorithm


def main():
    image = imageio.imread("images/test.png")

    ea = EvolutionaryAlgorithm(image, 5, 20)

    fitnesses, individuals = ea.run(generations=100, verbose=True)

    solution = individuals[np.argmax(fitnesses)]

    approximation = np.zeros(image.shape, dtype="uint8")
    ea.compute_image(approximation, solution, fill_color=image.mean())

    fig, axes = plt.subplots(1, 2)
    axes[0].set_title("Best Approximation")
    axes[0].imshow(approximation, cmap="gray", vmin=0, vmax=255)
    axes[0].axis("off")
    axes[1].set_title("Target Image")
    axes[1].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[1].axis("off")
    plt.show()


if __name__ == "__main__":
    main()
