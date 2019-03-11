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
