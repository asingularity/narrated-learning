import time
import numpy as np
from math import sqrt

if __name__ == '__main__':

    rows = 200
    cols = 200
    t0 = time.time()
    for r1 in range(rows):
        print r1
        for c1 in range(cols):
            for r2 in range(rows):
                for c2 in range(cols):
                    dist = sqrt(pow(r1 - r2, 2)  + pow(c1 - c2, 2))
    t1 = time.time()

    print 'elapsed: ', t1 - t0