#!/usr/bin/env python3
import imageio
import matplotlib.pyplot as plt

from evolve import EvolutionaryAlgorithm


def main():
    image = imageio.imread("images/test.png")

    ea = EvolutionaryAlgorithm(image.shape, 5, 20)

    EvolutionaryAlgorithm.compute_image(ea.image, ea.population[0])

    plt.title("Final Approximated Image")
    plt.axis("off")
    plt.imshow(ea.image, cmap="gray", vmin=0, vmax=255)
    plt.show()


if __name__ == "__main__":
    main()
