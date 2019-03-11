#!/usr/bin/env python3
import pstats

p = pstats.Stats("stats.txt")

p.strip_dirs().sort_stats("cumulative").print_stats(10)
p.strip_dirs().sort_stats("tottime").print_stats(10)
