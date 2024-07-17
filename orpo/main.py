# ⓒ 2024 Joona Kytöniemi
# ORPO Trainer for instructional datasets

import os
import sys
import json
import random

from datetime import datetime
from argparse import ArgumentParser
from trl import ORPOConfig, ORPOTrainer


def argparser():
    ap = ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])



if __name__ == "__main__":
    sys.exit(main(sys.argv))
