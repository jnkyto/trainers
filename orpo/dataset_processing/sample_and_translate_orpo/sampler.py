#!/usr/bin/env python3
# MIT ©2024 Joona Kytöniemi

import sys
import random
import logging
from time import sleep
import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def sampler(dframe: pd.DataFrame, bands: int, per: int, thold: float):
    """
    Picks semi-random samples from a Pandas Dataframe to be used in analysis.
    Intended to be used on a pre-sorted dataframe.

    :param dframe: The dataframe to be worked on.
    :param bands: The amount of examination points where samples should be taken, i.e. resolution.
    :param per: The amount of samples taken around one single band.
    :param thold: The amount of variation (percentage).
    :return: Picked samples along with the corresponding calculated intermediaries.
    """
    df_list = dframe.values.tolist()
    df_len = len(df_list)
    results = []

    band_len = int(df_len / bands)
    low_b = int(df_len / bands / 2)  #
    high_b = int(df_len - low_b)

    logging.info(f"Calculated intermediaries. low_b={low_b}, high_b={high_b}, band_len={band_len}")

    for band_no in range(0, bands, 1):
        phase_dict = {}
        entry_list = []
        rand_idxs = []

        if band_no == 0:
            band_loc = low_b
        elif band_no == bands - 1:
            band_loc = high_b
        else:
            band_loc = int(low_b + band_no * band_len)

        high_t = int(band_loc + (band_len * thold))
        low_t = int(band_loc - (band_len * thold))

        for i in range(per):
            random_index_found = False  # used to assure the following loop runs at least once, essentially a do-while
            rand_idx = 0

            while not random_index_found:
                rand_idx = int(random.uniform(low_t, high_t))

                if rand_idx in rand_idxs:
                    logging.warning(f"Detected random index clash! Sleeping a bit before retry. (band_no: {band_no})")
                    sleep(0.25)
                elif rand_idx > df_len-1:
                    logging.warning("Random index was higher than the dataset length! Sleeping a bit before retry.")
                    sleep(0.25)
                else:
                    rand_idxs.append(rand_idx)
                    random_index_found = True

            entry_list.append(df_list[rand_idx])

        phase_dict["entries"] = entry_list
        results.append(phase_dict)
    return results
