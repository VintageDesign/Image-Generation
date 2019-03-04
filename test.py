#!/usr/bin/env python3
import matplotlib.pyplot as plt
from skimage import data

from evolve.shapes import Circle
from evolve import Individual


def main():
    # Use an existing image with lots of circles in it to test with.
    coins = data.coins()

    x = Individual(10, Circle, coins.shape)

    plt.imshow(x.image, cmap="gray", interpolation="nearest")
    plt.show()


if __name__ == "__main__":
    main()
