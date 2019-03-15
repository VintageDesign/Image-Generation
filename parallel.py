import itertools
import imageio
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from evolve import BootstrapAlgorithm
from evolve import CircleDtype

def worker(args):
    """Find an approximation in a separate process."""
    seed, (target, circles, pop_size, generations) = args
    ba = BootstrapAlgorithm(target, circles, pop_size, generations, seed)
    return ba.run()
def average(image, circles,  pop_size, generations):
    """Use the bootstrap method to initialize a population of individuals.
        Performs the initialization in parallel using however many cores are available.
        :param pop_size: The bootstrap population size. (Number of circles)
        :param generations: The bootstrap generation length. (Number of iterations)
    """
    target = imageio.imread(image).astype("float32")
    approximate = np.zeros_like(target, dtype="float32")
    proc_pool = Pool()
    population = []
    # Use a different seed for each process to avoid results exactly the same as each other.
    seeds = np.random.randint(low=np.iinfo(np.uint32).max, size=pop_size)
    with Pool() as pool:
        results = pool.map(worker, zip(seeds, itertools.repeat((target, circles, pop_size, generations), times=pop_size)))

    for _, ind in results:
        approximate = approximate + ind


    approximate = approximate / len(results)





    print(" done.")
    _, axes = plt.subplots(1, 3)
    axes[0].set_title("Best Approximation")
    axes[0].imshow(approximate, cmap="gray", vmin=0, vmax=255)
    axes[0].axis("off")
    axes[1].set_title("Target Image")
    axes[1].imshow(target, cmap="gray", vmin=0, vmax=255)
    axes[1].axis("off")
    axes[2].set_title("Diff")
    axes[2].imshow(np.abs(target - approximate), cmap="gray", vmin=0, vmax=255)
    axes[2].axis("off")
    plt.show()


average('images/mona_lisa.png',400, 32, 50)
