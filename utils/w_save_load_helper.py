import numpy as np


def save_W_prob(filename, nz, W):
    f = open(filename, 'w')

    for from_index in range(len(nz)):
        for to_index in nz[from_index]:
            f.write("%d %f "% (to_index, W[to_index, from_index]))
        f.write('\n')
    f.close()


def load_W_prob(filename):

    f = open(filename, 'r')

    nz = []
    W = np.zeros((8000, 8000), np.float32)

    from_index = 0
    for line in f:
        nz.append([])
        lst = line.split(' ')

        odd = True
        to_index = None

        for thing in lst:
            if len(thing.strip()) > 0:
                if odd:
                    to_index = int(thing.strip())
                else:
                    assert to_index is not None
                    val = float(thing.strip())

                    nz[from_index].append(to_index)
                    W[to_index, from_index] = val
                odd = not odd

        from_index += 1

    f.close()
    assert from_index == 8000

    return nz, W

