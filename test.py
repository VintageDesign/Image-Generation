#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

from evolve.shapes import Circle


def main():
    image = np.ones((100, 100)) * 255

    c = Circle(image.shape)

    plt.ion()
    for _ in range(20):
        image = np.ones((100, 100)) * 255
        c.perturb()
        c.add_to_image(image)

        plt.imshow(image, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
        plt.pause(0.01)

    plt.show()


if __name__ == "__main__":
    main()
