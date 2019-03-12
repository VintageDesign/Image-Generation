import numpy as np

# Use uint16_t's for the centers, because we won't be working that *that* large of images.
CircleCenterDtype = np.dtype([("x", "float32"), ("y", "float32")])
CircleDtype = np.dtype([("radius", "float32"), ("color", "float32"), ("center", CircleCenterDtype)])


class EvolutionaryAlgorithm:
    """Implements an EA to approximate a given image."""

    def __init__(self, image, pop_size, ind_size):
        """Initialize the EA.

        :param image: The image to approximate
        :type image: A 2d array of uint8's
        :param pop_size: The number of individuals in the population.
        :param ind_size: The number of circles composing an individual.
        """
        self.target = image.astype("float32")
        self.height, self.width = image.shape
        self.pop_size = pop_size
        self.ind_size = ind_size

        self.population = np.zeros((pop_size, ind_size), dtype=CircleDtype)
        self.fitnesses = np.zeros(len(self.population))

        self.mutations = np.zeros_like(self.population)
        self.mutation_fitnesses = np.zeros(len(self.mutations))

        self.children = []
        self.children_fitnesses = np.zeros(len(self.children))

        # TODO: If we ever attempt to perform parallel computation, each process will need its own
        # image represented by a single individual.
        self.approx = np.zeros(image.shape, dtype="float32")

    def init_pop(self):
        """Randomly initialize the population."""
        for individual in self.population:
            self.init_individual(individual)

    def init_individual(self, individual):
        """Randomly initialize the given individual.

        :param individual: The individual to initialize.
        :type individual: An array of CircleDtype objects.
        """
        for circle in individual:
            self.init_circle(circle)

    def init_circle(self, circle):
        """Randomly initialize the given circle.

        :param circle: The circle to initialize.
        :type circle: A single CircleDtype object.
        """
        circle["color"] = np.random.randint(0, 256)
        # TODO: What should the bounds on the circle radii be?
        circle["radius"] = np.random.randint(5, max(self.height, self.width) / 8)
        circle["center"]["x"] = np.random.randint(0, self.width)
        circle["center"]["y"] = np.random.randint(0, self.height)

    @staticmethod
    def fitness(image1, image2):
        """Determine how close two images are."""
        assert image1.shape == image2.shape
        height, width = image1.shape

        # Potential overflow issue if the images are uints
        return np.sum(np.abs(image1 - image2)) / (height * width)

    # TODO: 99% Of the EA runtime is spent in this function. Make it better.
    @staticmethod
    def compute_image(image, individual, fill_color=255):
        """Compute the image represented by the given individual.

        :param image: The preallocated image to fill
        :type image: a (height, width) numpy array of uint8s
        :param individual: The individual representing the image
        :type individual: An array of CircleDtype objects
        :param fill_color: The image background color.
        :type fill_color: A uint8 between 0 and 255.
        """
        image.fill(fill_color)
        height, width = image.shape
        for circle in individual:
            # NOTE: The type of circle["center"]["x"] is a numpy uint16. However, for whatever
            # reason, a numpy uint16 does *not* play nicely with np.ogrid unless it's the value 0.
            cx, cy, r = int(circle["center"]["x"]), int(circle["center"]["y"]), circle["radius"]

            x, y = np.ogrid[-cy : height - cy, -cx : width - cx]
            mask = x ** 2 + y ** 2 <= r ** 2
            # TODO: This is not the right way to blend the colors.
            image[mask] += circle["color"]

    def update_fitnesses(self, population, fitnesses):
        """Update the fitnesses for the given population."""
        # TODO: This is a good candidate for parallelism.
        for i, individual in enumerate(population):
            # TODO: Fill with the mean color of the target image?
            self.compute_image(self.approx, individual)
            fitnesses[i] = self.fitness(self.approx, self.target)

    def evaluate(self, population="general"):
        """Update the population fitnesses.

        :param population: Which population to evaluate. One of 'all', 'general', 'mutations', or
        'children'.
        """
        if population == "general":
            self.update_fitnesses(self.population, self.fitnesses)
        elif population == "mutations":
            self.update_fitnesses(self.mutations, self.mutation_fitnesses)
        elif population == "children":
            self.update_fitnesses(self.children, self.children_fitnesses)
        elif population == "all":
            self.update_fitnesses(self.population, self.fitnesses)
            self.update_fitnesses(self.mutations, self.mutation_fitnesses)
            self.update_fitnesses(self.children, self.children_fitnesses)

    def perturb_radius(self, circle, scale):
        """Perturb the radius of the given circle."""
        dr = int(np.random.normal(scale=scale))
        # if circle["radius"] + dr < 2:
        #     dr = -dr

        circle["radius"] += dr

    def perturb_color(self, circle, scale):
        """Perturb the color of the given circle."""
        dc = int(np.random.normal(scale=scale))
        circle["color"] += dc

    def perturb_center(self, circle, scale):
        """Perturb the center of the given circle."""
        dx, dy = np.random.normal(scale=scale, size=2).astype(int)
        # TODO: Use min/max rather than mod.
        circle["center"]["x"] = (circle["center"]["x"] + dx) % self.width
        circle["center"]["y"] = (circle["center"]["y"] + dy) % self.height

    def mutate_individual(self, individual, scale):
        """Mutate the given individual in place."""
        for circle in individual:
            choice = np.random.randint(0, 3)
            if choice == 0:
                self.perturb_radius(circle, scale)
            elif choice == 1:
                self.perturb_color(circle, scale)
            elif choice == 2:
                self.perturb_center(circle, scale)

    def mutate(self, scale):
        """Mutate each individual in the population."""
        np.copyto(self.mutations, self.population)
        for mutant in self.mutations:
            self.mutate_individual(mutant, scale)

    def reproduce(self):
        """Reproduce the individuals in the population."""
        # TODO: Should the population be sorted by fitness?
        # TODO: Should the best be intentionally reproduced with the worst?
        raise NotImplementedError

    def select(self):
        """Select the top 50% of the population based on fitness."""
        joint = np.concatenate((self.population, self.mutations))
        fit = np.concatenate((self.fitnesses, self.mutation_fitnesses))

        indices = np.argsort(fit)
        joint = joint[indices]
        fit = fit[indices]

        self.population = joint[: self.pop_size]
        self.fitnesses = fit[: self.pop_size]

    # TODO: Display the current best individual as the EA progresses.
    def run(self, generations, verbose=False):
        """Run the EA.

        :param generations: The number of generations to run the EA for
        :param verbose: Output useful diagnostic information
        :returns: The best individuals over the total runtime of the algorithm.
        :rtype: An array of EvolutionaryAlgorithm.HistoryDtype objects.
        """
        self.init_pop()
        self.evaluate(population="general")

        fitnesses = np.zeros(generations)
        individuals = np.zeros((generations, self.ind_size), dtype=CircleDtype)
        for gen in range(generations):
            # self.reproduce()
            self.mutate(scale=6)
            # TODO: This is 95+% of the runtime of the run() function call.
            # And compute_image is 95+% of this call.
            self.evaluate(population="general")
            self.evaluate(population="mutations")

            self.select()

            best = np.argmin(self.fitnesses)
            if verbose:
                print("best fitness:", self.fitnesses[best])
                print("best:", self.population[best])
            fitnesses[gen] = self.fitnesses[best]
            individuals[gen] = self.population[best]

        return fitnesses, individuals
