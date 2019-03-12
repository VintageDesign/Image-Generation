# natural-computing-hw2

Evolving portraits using geometric shapes.

---

## Profiling

Profiling the `test.py` script is quite easy.

```shell
# Generate the profile info
python3 -m cProfile -o stats.txt test.py --quiet images/test.png
# View the top 25 functions by cumulative time.
./stats.py
```

## TODO

* Clean up
* Parallelize fitness evaluation.
* Experiment with floats and/or int16s.
* Think about the right way to compute and display the difference between the two images.
* Get started on the report
* Find ways to be innovative
* Try more complicated images...
* Make `stats.py` use commandline arguments for the stats file.
* Attempt recombination
