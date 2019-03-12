#!/usr/bin/env python3
import argparse
import pstats


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze the profiler results.")
    parser.add_argument("statsfile", help="The profiler output file.")

    return parser.parse_args()


def main(args):
    p = pstats.Stats(args.statsfile)

    p.strip_dirs().sort_stats("cumulative").print_stats(10)
    p.strip_dirs().sort_stats("tottime").print_stats(10)


if __name__ == "__main__":
    main(parse_args())
