#!/usr/bin/env python3
import matplotlib.pyplot as plt

from evolve.shapes import NumpyCircleArray


def main():
    circles = NumpyCircleArray(2, (100, 100), True)
    plt.ion()
    for _ in range(20):
        circles.mutate()
        circles.update_image()
        plt.imshow(circles.image, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
        plt.pause(0.05)

    plt.show()


if __name__ == "__main__":
    main()
