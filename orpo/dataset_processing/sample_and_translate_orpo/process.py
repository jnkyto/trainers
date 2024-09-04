#!/usr/bin/env python3
# MIT ⓒ2024 Joona Kytöniemi

# NOTE:
# This script is currently at an unfinished state

import os
import sys
import pandas as pd
from argparse import ArgumentParser
from sampler import sampler

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
    print("File presence validated successfully.")


def main(args):
    check_input_files_present(_root_data_directory, _file_paths)
    print("Gathering all of the samples. This might take a while.")
    all_df = pd.DataFrame()
    for directory in _file_paths:
        dataset_samples = []
        for i, file in enumerate(_file_paths[directory]):
            with open(f"{_root_data_directory}/{directory}/{file}", 'r') as f:
                for line in f:
                    key = "en" if i == 0 else "fi"
                    if i != 0:
                        dataset_samples[i][key] = line.rstrip("\n")
                    else:
                        dataset_samples.append({key: line})
        temp_df = pd.DataFrame.from_dict(dataset_samples)
        temp_df = sampler(dframe=temp_df, bands=10, per=20, thold=0.1)
        all_df.append(temp_df)

    # TODO: Make everything above work and export sampled data to a single jsonl-file
    # The data structure is invalid currently



if __name__ == "__main__":
    args = argparser().parse_args(sys.argv[1:])
    sys.exit(main(args))
