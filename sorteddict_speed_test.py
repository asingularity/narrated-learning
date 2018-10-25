import random
from time import time
from sortedcontainers import SortedDict


def test_sorted_dict():
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

if __name__ == '__main__':
    test_sorted_dict()
