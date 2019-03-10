#!/usr/bin/env python3
import imageio
import matplotlib.pyplot as plt
import numpy as np

from evolve import EvolutionaryAlgorithm


def main():
    image = imageio.imread("images/test.png")

    ea = EvolutionaryAlgorithm(image, 5, 20)

    fitnesses, individuals = ea.run(generations=100)

    solution = individuals[np.argmax(fitnesses)]

    approximation = np.zeros(image.shape, dtype="uint8")
    ea.compute_image(approximation, solution, fill_color=image.mean())

    plt.title("Final Approximated Image")
    plt.axis("off")
    plt.imshow(approximation, cmap="gray", vmin=0, vmax=255)
    plt.show()


if __name__ == "__main__":
    main()
