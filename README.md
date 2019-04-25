# natural-computing-hw2

Evolving portraits using geometric shapes.

---

Results can be found in [`paper/output`](paper/output).
The paper itself can be build with the included [makefile](paper/Makefile).

The [`test.py`](test.py) script has the usage

```shell
$ ./test.py --help
usage: test.py [-h] [--quiet] [--population POPULATION] [--circles CIRCLES]
               [--generations GENERATIONS] [--bootstrap | --ea | --combined]
               image [output]

Approximate the given image using an EA.

positional arguments:
  image                 The image to approximate.
  output                The filename to save the output to.

optional arguments:
  -h, --help            show this help message and exit
  --quiet               Quiet all output.
  --population POPULATION, -p POPULATION
                        The population size.
  --circles CIRCLES, -c CIRCLES
                        The number of circles used to approximate the image.
  --generations GENERATIONS, -g GENERATIONS
                        The number of generations.
  --bootstrap           Use the bootstrap method to build a quick
                        approximation.
  --ea                  The default traditional, slow EA approach.
  --combined            A combined approach.
```
