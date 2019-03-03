#!/usr/bin/env python3
import matplotlib.pyplot as plt
from skimage import data

from evolve.shapes import Circle


def main():
    c = Circle((100, 200), 50, 3, 1)
    # Use an existing image with lots of circles in it to test with.
    coins = data.coins()

    # Arrays are passed by reference and modifiable.
    c.add_to_image(coins)

    plt.imshow(coins, cmap='gray', interpolation='nearest')
    plt.show()


if __name__ == "__main__":
    main()
