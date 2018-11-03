import numpy as np
import random
from time import time


def test_sorted_dict():
    from sortedcontainers import SortedDict
    rows = 2000

    a = SortedDict()

    # fill initially
    for k in range(rows * rows):
        a[random.random()] = (random.randint(1, rows - 1), random.randint(1, rows - 1))

    print('Start...')

    t0 = time()
    frames = 100
    for k in range(frames):
        for r in range(rows):
            a.popitem()
            a[random.random()] = (random.randint(1, rows - 1), random.randint(1, rows - 1))
    t1 = time()

    print('FPS: ', frames * 1.0 / (t1 - t0))


def test_all_all_array():
    rows = 20000
    d = np.random.random((rows, rows))
    actual_min = np.amin(d)
    print(d.shape)
    print('min', actual_min)
    print()
    print('Start...')

    t0 = time()
    frames = 5000
    for k in range(frames):
        # choose a random row, and replaces all distances to/from it with a new float array
        rep_row_ind = random.randint(1, rows - 1)
        new_row = np.random.random(rows)[:]
        d[rep_row_ind, :] = new_row
        d[:, rep_row_ind] = new_row

        min_d = np.amin(new_row)
        if min_d < actual_min:
            actual_min = min_d
        #(random.randint(1, rows - 1), random.randint(1, rows - 1))
    t1 = time()

    print('FPS: ', frames * 1.0 / (t1 - t0))
    print()
    print('min', actual_min)


if __name__ == '__main__':
    #test_sorted_dict()
    test_all_all_array()