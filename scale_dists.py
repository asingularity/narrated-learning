import time
import cv2
import numpy as np
from math import sin, cos


def _scale_dists_rank(dists):
    '''
        such that linear for:
        min distance pair -> output confidence = 1.0
        max distance pair -> output confidence = 0.0
        hard nonlinearity otherwise (maxed out to 0 or 1)
    '''

    dists_copy = dists.copy()
    temp = np.argsort(dists_copy)

    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(dists_copy))
    dists_copy = ranks

    dists_copy = (dists_copy * 1.0 / len(dists_copy)).astype(dists.dtype)
    return dists_copy


def _scale_dists_max(dists):
    '''
        such that linear for:
        min distance pair -> output confidence = 1.0
        max distance pair -> output confidence = 0.0
        hard nonlinearity otherwise (maxed out to 0 or 1)
    '''

    temp = np.argsort(dists)

    dists_copy = np.zeros_like(dists)
    val = 1.0

    top_k = 30  # 10
    for k in range(top_k):
        dists_copy[temp[k]] = val
        val *= 0.99

    dists_copy = dists_copy.astype(dists.dtype)
    return dists_copy


def _scale_dists_simple(dists):
    min_dist = np.amin(dists)
    max_dist = np.amax(dists)
    if max_dist == 0:
        dists_copy = dists.copy()
    elif max_dist - min_dist == 0:
        dists_copy = dists * 1.0 / max_dist
    else:
        dists_copy = (dists - min_dist) * 1.0 / (max_dist - min_dist)

    return dists_copy


def _scale_dists_rank_2(dists):
    '''
        such that linear for:
        min distance pair -> output confidence = 1.0
        max distance pair -> output confidence = 0.0
        hard nonlinearity otherwise (maxed out to 0 or 1)
    '''

    dists_copy = dists.copy()
    temp = np.argsort(dists_copy)

    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(dists_copy))[::-1]

    dists_copy = ranks
    dists_copy = (dists_copy * 1.0 / (len(dists_copy) - 1)).astype(dists.dtype)

    # new part: exponential decay
    # dists_copy * 1.0 / np.exp(1.0 - dists_copy)

    return dists_copy


def test_scale_dists():
    '''

    cases to test
        distances:
            all zeros
            all non-zero
            distribution: varied but flat
            distribution: varied but clear max / min

    :return:
    '''


if __name__ == '__main__':
    test_scale_dists()