
import numpy as np
import time
from brute_force_knn import knn
from sklearn.neighbors import KDTree
from knn_cython import knn_query
from knn_parallel import knn_query as knn_parallel_query
import pickle
from fast_save_matrix import savetxt
from OLD_fast_save_matrix import savetxt as savetxt_old
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath

USERNAME = 'intec'


def run_test_incremental():
    frames = 1000000
    dim = 20
    data = np.random.random((frames, dim))
    last_time = time.time()
    last_frame = 0

    for frame in range(frames):
        if frame > 2:
            knn_1 = knn(data[0:frame - 1, :])
            dist, ind = knn_1.query([data[frame, :]], 1)
        if time.time() - last_time > 5:
            FPS = (frame - last_frame) * 1.0 / (time.time() - last_time)
            print 'frame: ', frame, 'FPS: ', FPS
            last_frame = frame
            last_time = time.time()


def run_test_full(tree, test_seconds):
    data_frames = 1000000 * 2
    dim = 20
    #data = 1.0 + 0.0 * np.random.random((data_frames, dim))
    data = np.random.random((data_frames, dim)).astype(np.float32)

    if tree:
        print 'building KDTree...'
        build_start_time = time.time()
        knn_1 = KDTree(data, leaf_size=40)
        build_end_time = time.time()
        print 'done. time: ' + str(build_end_time - build_start_time)
    else:
        print 'building KNN...'
        knn_1 = knn(data)
        print 'done.'

    start_time = time.time()
    last_time = time.time()
    last_frame = 0

    test_frames = 10000

    for frame in range(test_frames):

        dist, ind = knn_1.query([np.random.random(dim).astype(np.float32)], 1)
        if frame == 0:
            print dist, ind
        if time.time() - last_time > 5:
            FPS = (frame - last_frame) * 1.0 / (time.time() - last_time)
            print 'frame: ', frame, 'FPS: ', FPS
            last_frame = frame
            last_time = time.time()
        if time.time() - start_time > test_seconds:
            return


def run_cython_knn_test(test_seconds, parallel=False):
    data_frames = 1000000 * 2
    dim = 20
    data = np.random.random((data_frames, dim)).astype(np.float32)
    tmp = np.zeros(data_frames).astype(np.float).astype(np.float32)
    start_time = time.time()
    last_time = time.time()
    last_frame = 0

    test_frames = 10000

    for frame in range(test_frames):

        query_data = np.random.random(dim).astype(np.float32)

        if parallel:
            dist, ind = knn_parallel_query(data, query_data, tmp, data_frames, dim)
        else:
            dist, ind = knn_query(data, query_data, data_frames, dim)
        if frame == 0:
            print dist, ind
        if time.time() - last_time > 5:
            FPS = (frame - last_frame) * 1.0 / (time.time() - last_time)
            print 'frame: ', frame, 'FPS: ', FPS
            last_frame = frame
            last_time = time.time()
        if time.time() - start_time > test_seconds:
            return


def cuda_init(data, query_data):
    data_gpu = cuda.mem_alloc(data.size * data.dtype.itemsize)
    arr_gpu = cuda.mem_alloc(query_data.size * query_data.dtype.itemsize)
    mod = SourceModule("""
        __global__ void knn_query(float *data, float *arr)
        {
          int idx = threadIdx.x + threadIdx.y*4;
          a[idx] *= 2;
        }
        """)
    func = mod.get_function("knn_query")
    cuda.memcpy_htod(data_gpu, data)

    return data_gpu, arr_gpu, func
    #return func


def cuda_query(query_data, a_gpu, b_gpu, func, b_doubled):

    cuda.memcpy_htod(b_gpu, query_data)
    func(a_gpu, b_gpu, block=(4, 4, 1))
    #   block=(threads_x * threads_y * blocks) ?
    #   no... there's also
    #       block = (32, 1, 1), grid=(2, 1)

    cuda.memcpy_dtoh(b_doubled, b_gpu)

    #a_doubled[:, :] = 2.0 * data[:, :]

    #diff = data - query_data
    #dists = np.sum(np.fabs(diff), axis=1)
    #ind = np.argmin(dists)
    #dist = dists[ind]

    dist, ind = None, None
    return dist, ind


def run_cuda_test(test_seconds):
    data_frames = 1000000 * 2
    dim = 20
    data = np.random.random((data_frames, dim)).astype(np.float32)

    query_data = np.random.random(dim).astype(np.float32)
    b_doubled = np.empty_like(query_data)
    a_gpu, b_gpu, func = cuda_init(data, query_data)

    start_time = time.time()
    last_time = time.time()
    last_frame = 0

    test_frames = 1000000

    for frame in range(test_frames):
        query_data = np.random.random(dim).astype(np.float32)
        dist, ind = cuda_query(query_data, a_gpu, b_gpu, func, b_doubled)

        if frame == 0:
            print dist, ind
        if time.time() - last_time > 1:
            FPS = (frame - last_frame) * 1.0 / (time.time() - last_time)
            print 'frame: ', frame, 'FPS: ', FPS
            last_frame = frame
            last_time = time.time()
        if time.time() - start_time > test_seconds:
            return


def knn_save_test(optimized):
    data_frames = 1000000 * 2
    dim = 20
    data = np.random.random((data_frames, dim))
    start_time = time.time()
    print 'saving knn of size: ', data_frames, dim
    if optimized == 3:
        print '--- savetxt cython'
        savetxt('temp.dat', data)
    elif optimized == 2:
        print '--- pickle'
        f = open('temp.pkl', 'w')
        pickle.dump(data, f)
        f.close()
    elif optimized == 1:
        print '--- savetxt python'
        savetxt_old('temp.dat', data)
    else:
        print '--- np.savetxt'
        np.savetxt('temp.dat', data)
    print '...done.'

    print 'time: ', time.time() - start_time

if __name__ == '__main__':

    test_save = False
    test_run = True

    if test_save:
        np.random.seed(0)
        knn_save_test(optimized=3)
        np.random.seed(0)
        knn_save_test(optimized=2)
        np.random.seed(0)
        knn_save_test(optimized=1)
        np.random.seed(0)
        knn_save_test(optimized=0)

    if test_run:
        test_seconds = 6
        np.random.seed(0)
        print '--- cuda ---'
        run_cuda_test(test_seconds=test_seconds)
        np.random.seed(0)
        print '--- numpy ---'
        run_test_full(tree=False, test_seconds=test_seconds)
        np.random.seed(0)
        print '--- cython ---'
        run_cython_knn_test(test_seconds=test_seconds)
        np.random.seed(0)
        print '--- cython parallel ---'
        run_cython_knn_test(test_seconds=test_seconds, parallel=True)
