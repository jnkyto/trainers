#!/usr/bin/env python3
# MIT ⓒ2024 Joona Kytöniemi
# Viewer for ORPO-formatted jsonl's

import sys
from argparse import ArgumentParser, FileType

import jsonlines


def argparse():
    ap = ArgumentParser()
    ap.add_argument("--input", type=FileType('r'), metavar='F', required=True)
    return ap


def main(argv):
    args = argparse().parse_args(argv[1:])
    reader = jsonlines.Reader(args.input)
    for i, obj in enumerate(reader):
        print(f"-------------------- msg no. {i} --------------------\n")
        print(f'Prompt:\n\t{obj["prompt"]}\n\n')
        print(f'Chosen:\n\t{obj["chosen"]}\n\n')
        print(f'Rejected:\n\t{obj["rejected"]}\n')


if __name__ == "__main__":
    sys.exit(main(sys.argv))
