#!/usr/bin/env python3
# MIT ⓒ2024 Joona Kytöniemi

import os
import sys
import pandas as pd
from argparse import ArgumentParser

_root_data_directory = "../../../data/en_fi"
_file_paths = {
    "elrc-fi_info_en-fi": [
        "ELRC-Finnish_Information.en-fi.en",
        "ELRC-Finnish_Information.en-fi.fi"
    ],
    "elrc-norden_en-fi": [
        "ELRC-www.norden.org.en-fi.en",
        "ELRC-www.norden.org.en-fi.fi"
    ],
    "opensubtitles-2016_en-fi": [
        "OpenSubtitles.en-fi.en",
        "OpenSubtitles.en-fi.fi"
    ],
    "ted2020_en-fi": [
        "TED2020.en-fi.en",
        "TED2020.en-fi.fi"
    ]
}


def argparser():
    ap = ArgumentParser()
    return ap


def check_input_files_present(root_dir, file_paths):
    assert os.path.exists(root_dir)
    for directory in file_paths:
        assert os.path.exists(f"{root_dir}/{directory}")
        for file in file_paths[directory]:
            assert os.path.isfile(f"{root_dir}/{directory}/{file}")
    print("Files validated successfully.")


def main(args):
    check_input_files_present(_root_data_directory, _file_paths)
    all_samples = []
    for directory in _file_paths:
        for i, file in enumerate(_file_paths[directory]):
            samples = []
            with open(f"{_root_data_directory}/{directory}/{file}", 'r') as f:
                for line in f:
                    key = "en" if i == 0 else "fi"
                    samples.append({key: line})
            all_samples.append(samples)

    print(all_samples[0:10])


if __name__ == "__main__":
    args = argparser().parse_args(sys.argv[1:])
    sys.exit(main(args))
