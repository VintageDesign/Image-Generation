#!/usr/bin/env python3
import imageio
import matplotlib.pyplot as plt

from evolve import simulated_annealing


def main():
    image = imageio.imread("images/test.png")

    # TODO: We may need to explore algorithms other than SA :(
    approximation = simulated_annealing(image, circles=20)

    plt.title("Final Approximated Image")
    plt.axis("off")
    plt.imshow(approximation.image, cmap="gray", vmin=0, vmax=255)
    plt.show()


if __name__ == "__main__":
    main()
