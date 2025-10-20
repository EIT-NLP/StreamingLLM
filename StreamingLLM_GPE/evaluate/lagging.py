# coding=utf-8
# Modified 2025 by Junlong Tong (Shanghai Jiao Tong University & Eastern Institute of Technology).
#
# This file is a modified version of the 'simuleval' repository implementation from:
# The Facebook, Inc.
#
# Original license and copyright as follows:
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license of 
# Attribution-ShareAlike 4.0 International





def calculate_al_and_laal(source_length, target_length, delays):
    """
    Function to compute latency on one sentence (instance).

    Args:
        delays (List[Union[float, int]]): Sequence of delays.
        source_length (Union[float, int]): Length of source sequence.
        target_length (Union[float, int]): Length of target sequence.

    Returns:
        float: the latency score on one sentence.
    """

    if delays[0] > source_length:
        return delays[0], delays[0]

    AL = 0
    LAAL = 0
    gamma_AL = target_length / source_length
    gamma_LAAL = max(len(delays), target_length) / source_length
    tau = 0
    for t_miuns_1, d in enumerate(delays):
        if d <= source_length:
            AL += d - t_miuns_1 / gamma_AL
            LAAL += d - t_miuns_1 / gamma_LAAL
            tau = t_miuns_1 + 1

            if d == source_length:
                break
    AL /= tau
    LAAL /= tau
    return AL, LAAL
    