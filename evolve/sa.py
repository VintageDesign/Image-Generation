import matplotlib.pyplot as plt
import numpy as np

from evolve.image import fitness
from evolve.shapes import NumpyCircleArray


# TODO: Pass in all the tweakable parameters.
def simulated_annealing(image, circles, temp=2000, cooling_factor=0.0001):
    """Produce the best approximation of the given image."""
    fig, axes = plt.subplots(1, 2)
    axes[0].axis("off")
    axes[1].axis("off")
    plt.ion()

    x = NumpyCircleArray(circles, image.shape, random=True)
    x.update_image()
    fx = fitness(image, x.image)

    iteration = 0

    # TODO: I want the fitness to be a part of the convergence criterion...
    while temp > 0.0001:
        # TODO: Pass tweakable parameters.
        x_new = x.mutate()
        x_new.update_image()
        fx_new = fitness(image, x_new.image)

        if np.random.random() < np.exp((fx - fx_new) / temp):
            x = x_new
            fx = fx_new

        temp *= 1 - cooling_factor
        iteration += 1

        if iteration % 100 == 0:
            axes[0].set_title(f"Iteration {iteration}")
            axes[0].imshow(x.image, cmap="gray", vmin=0, vmax=255)
            axes[1].set_title("Desired Image")
            axes[1].imshow(image, cmap="gray", vmin=0, vmax=255)
            plt.pause(0.00001)

    return x
