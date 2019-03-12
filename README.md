# natural-computing-hw2

Evolving portraits using geometric shapes.

---

## Profiling

Profiling the `test.py` script is quite easy.

```shell
# Generate the profile info
python3 -m cProfile -o stats.txt test.py --quiet images/test.png
# View the top 25 functions by cumulative time.
./stats.py stats.txt
```

Moving from integers to floats made the fitness function an order of magnitude faster, but only moderately improved the image computation function.

```text
(natural) nots@abyss ~/Documents/school/natural-computing/homework/hw2 $ ./stats.py ints.txt
Mon Mar 11 21:10:04 2019    ints.txt

         3117393 function calls (3107727 primitive calls) in 26.154 seconds

   Ordered by: cumulative time
   List reduced from 3610 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    398/1    0.008    0.000   26.155   26.155 {built-in method builtins.exec}
        1    0.000    0.000   26.155   26.155 test.py:2(<module>)
        1    0.000    0.000   25.841   25.841 test.py:26(main)
        1    0.001    0.001   25.834   25.834 ea.py:177(run)
      201    0.000    0.000   25.614    0.127 ea.py:104(evaluate)
      201    0.117    0.001   25.614    0.127 ea.py:96(update_fitnesses)
    40200   11.004    0.000   14.832    0.000 ea.py:63(fitness)
    40200    9.634    0.000   10.665    0.000 ea.py:73(compute_image)
    40512    2.078    0.000    2.262    0.000 {method 'astype' of 'numpy.ndarray' objects}
    40200    0.058    0.000    1.268    0.000 fromnumeric.py:1966(sum)


Mon Mar 11 21:10:04 2019    ints.txt

         3117393 function calls (3107727 primitive calls) in 26.154 seconds

   Ordered by: internal time
   List reduced from 3610 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    40200   11.004    0.000   14.832    0.000 ea.py:63(fitness)
    40200    9.634    0.000   10.665    0.000 ea.py:73(compute_image)
    40512    2.078    0.000    2.262    0.000 {method 'astype' of 'numpy.ndarray' objects}
    40282    0.983    0.000    1.034    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    40200    0.627    0.000    0.908    0.000 index_tricks.py:147(__getitem__)
   120602    0.202    0.000    0.298    0.000 util.py:158(_copy_meta)
    80461    0.161    0.000    0.161    0.000 {built-in method numpy.arange}
    40200    0.122    0.000    0.122    0.000 {method 'fill' of 'numpy.ndarray' objects}
      201    0.117    0.001   25.614    0.127 ea.py:96(update_fitnesses)
   120601    0.102    0.000    0.432    0.000 util.py:173(__array_finalize__)
```

```text
(natural) nots@abyss ~/Documents/school/natural-computing/homework/hw2 $ ./stats.py float32.txt
Mon Mar 11 21:09:22 2019    float32.txt

         2762298 function calls (2752632 primitive calls) in 12.948 seconds

   Ordered by: cumulative time
   List reduced from 3610 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    398/1    0.008    0.000   12.948   12.948 {built-in method builtins.exec}
        1    0.000    0.000   12.948   12.948 test.py:2(<module>)
        1    0.000    0.000   12.630   12.630 test.py:26(main)
        1    0.001    0.001   12.623   12.623 ea.py:177(run)
      201    0.000    0.000   12.414    0.062 ea.py:104(evaluate)
      201    0.096    0.000   12.414    0.062 ea.py:96(update_fitnesses)
    40200    8.175    0.000    9.287    0.000 ea.py:73(compute_image)
    40200    1.608    0.000    3.032    0.000 ea.py:63(fitness)
    40200    0.052    0.000    1.112    0.000 fromnumeric.py:1966(sum)
    40263    0.074    0.000    1.052    0.000 fromnumeric.py:69(_wrapreduction)


Mon Mar 11 21:09:22 2019    float32.txt

         2762298 function calls (2752632 primitive calls) in 12.948 seconds

   Ordered by: internal time
   List reduced from 3610 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    40200    8.175    0.000    9.287    0.000 ea.py:73(compute_image)
    40200    1.608    0.000    3.032    0.000 ea.py:63(fitness)
    40282    0.867    0.000    0.910    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    40200    0.537    0.000    0.777    0.000 index_tricks.py:147(__getitem__)
    40200    0.335    0.000    0.335    0.000 {method 'fill' of 'numpy.ndarray' objects}
    80461    0.136    0.000    0.136    0.000 {built-in method numpy.arange}
    80403    0.125    0.000    0.185    0.000 util.py:158(_copy_meta)
      201    0.096    0.000   12.414    0.062 ea.py:96(update_fitnesses)
   120600    0.085    0.000    0.085    0.000 util.py:182(__array_wrap__)
    40263    0.074    0.000    1.052    0.000 fromnumeric.py:69(_wrapreduction)
```

## TODO

* Clean up
* Parallelize fitness evaluation.
* Think about the right way to compute and display the difference between the two images.
* Get started on the report
* Find ways to be innovative
* Try more complicated images...
* Attempt recombination

## Strategy

Here's what I want to try after several hours of talking with Kyle. This is a simplified version of
his algorithm.

```python
def gen_individual(approximation):
    """Generate a decent approximation of the target image."""
    individual = []
    # An individual is a single circle. I want to make this as small as possible, maybe even one.
    population = init_pop()
    for _ in range(generations):
        mutate(population)
        # Might be tough with just a single circle.
        recombine(population)
        select(population)
    approximation = add_to_image(approximation, get_best(population))
    individual.append(get_best(population))

    return individual
```

This does a pretty good job of approximating the image, and quite quickly. I want to use this algorithm
to bootstrap an evolutionary algorithm.

```python
# Trivially parallelisable.
population = [gen_individual() for _ in range(pop_size)]
for _ in range(generations):
    # This would be fine-tuning the already decent approximations found above.
    mutate(population)
    recombine(population)
    select(population)
```
