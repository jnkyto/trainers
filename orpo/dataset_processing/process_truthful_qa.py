#!/usr/bin/env python3
# MIT ⓒ2024 Joona Kytöniemi
# Process truthful_qa dataset to ORPO format

import os
import sys
import jsonlines

from datasets import load_dataset
from argparse import ArgumentParser, FileType


def argparser():
    ap = ArgumentParser()
    ap.add_argument("--output", type=FileType('w+'), metavar='F', required=True)
    # TODO: Need to implement args in a different way where they can be freely accessed within every method
    # ap.add_argument("--prefix", type=str, default=default_prefix)
    # ap.add_argument("--suffix", type=str, default=default_suffix)
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])
    ds = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]

    samples = []
    for sample in ds:
        samples.append({
            "prompt": sample["question"],
            "chosen": sample["best_answer"],
            "rejected": sample["incorrect_answers"][0]
        })

    writer = jsonlines.Writer(args.output)
    writer.write_all(samples)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
