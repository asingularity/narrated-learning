import numpy as np
import random
from time import time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import skcuda.linalg as linalg
import skcuda.misc as misc


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
    rows = 2000
    d = np.random.random((rows, rows))
    actual_min = np.amin(d)
    print(d.shape)
    print('min', actual_min)
    print()
    print('Start...')

    t0 = time()
    frames = 500
    for k in range(frames):
        # choose a random row, and replaces all distances to/from it with a new float array
        rep_row_ind = random.randint(1, rows - 1)
        new_row = np.random.random(rows)[:]
        d[rep_row_ind, :] = new_row
        d[:, rep_row_ind] = new_row

        min_d = np.amin(new_row)
        if min_d < actual_min:
            actual_min = min_d

    t1 = time()

    print('FPS: ', frames * 1.0 / (t1 - t0))
    print()
    print('min', actual_min)


def test_min_cpu(rows, frames, dtype):
    d = np.random.random((rows * rows)).astype(dtype)
    min_d = np.amin(d)

    t0 = time()
    print('Start...')
    for k in range(frames):
        min_d = np.argmin(d)

    t1 = time()
    print('Done. min', min_d)

    print('FPS: ', frames * 1.0 / (t1 - t0))


def test_min_gpu(rows, frames, dtype):
    DO_SET_DISTS_TEST = True

    linalg.init()

    d = np.random.random((rows * rows, 1)).astype(dtype)
    min_d = np.amin(d)
    min_d_gpu = None

    d_gpu = gpuarray.to_gpu(np.ascontiguousarray(d))

    t0 = time()
    print('Start...')
    out_list = []
    for k in range(frames):
        min_d_gpu = misc.min(d_gpu)
        #print(min_d_gpu)
        out_list.append(min_d_gpu)
        #min_d_gpu = gpuarray.min(d_gpu)

        #min_index_gpu = misc.argmin(d_gpu, axis=0, keepdims=False)
        #min_index = min_index_gpu.get()[0]
#        out_list.append(min_index)
        #print(min_index)

        #print(min_index, min_d, d[min_index, 0])

        #print(min_d_gpu, min_index, d[min_index])
        if DO_SET_DISTS_TEST:
            rand_row = random.randint(1, rows - 1)
            new_row_to_table_dists = np.random.random(rows).astype(dtype)
            new_row_to_table_dists_gpu = gpuarray.to_gpu(new_row_to_table_dists)

            misc.set_by_index(dest_gpu=d_gpu, ind=rand_row * rows + np.arange(rows), src_gpu=new_row_to_table_dists_gpu, ind_which='dest')
            misc.set_by_index(dest_gpu=d_gpu, ind=rand_row + rows * np.arange(rows), src_gpu=new_row_to_table_dists_gpu, ind_which='dest')

    t1 = time()

    print('Done. min', min_d, min_d_gpu)
    print('FPS: ', frames * 1.0 / (t1 - t0))


if __name__ == '__main__':
    #test_sorted_dict()
    #test_all_all_array()

    rows = 10000
    frames = 20
    dtype = np.float32

    print()
    print('Starting test: rows: ', rows, ', frames:', frames)
    print()
    print('GPU')
    test_min_gpu(rows, frames, dtype)
    print()
    print('CPU')
    test_min_cpu(rows, frames, dtype)
