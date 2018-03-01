
import numpy as np
import time
from brute_force_knn import knn
from knn_cython import knn_query
from knn_parallel import knn_query as knn_parallel_query
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import skcuda.linalg as linalg

import time
import numpy as np
from matrix_vector_dist_parallel import knn_query as matrix_vect_dist_cython
from brute_force_knn import matrix_vect_dist_numpy

USERNAME = 'intec'


SMALL_TEST = 0
if SMALL_TEST:
    FRAMES = 2000000
    DIM = 8 * 3
else:
    FRAMES = 10000
    DIM = 40000


class CudaQuery(object):
    def __init__(self, input_data):
        self.original_input_data_shape = input_data.shape
        query_data = np.zeros(input_data.shape[1], np.float32)

        self.input_data = input_data #.flatten()  # row-major order. do not have to flatten.
        self.tmp_data = np.zeros(self.input_data.shape, np.float32)

        mod = self._get_mod()
        #self.func = mod.get_function("knn_query")

        #self.input_data_gpu = cuda.mem_alloc(self.input_data.size * self.input_data.dtype.itemsize)
        #self.query_data_gpu = cuda.mem_alloc(query_data.size * query_data.dtype.itemsize)
        #self.tmp_data_gpu = cuda.mem_alloc(self.tmp_data.size * self.tmp_data.dtype.itemsize)

        #cuda.memcpy_htod(self.input_data_gpu, self.input_data)

        self.X = np.zeros((1, DIM)).astype(np.float32)
        self.input_data_transpose = np.ascontiguousarray(np.transpose(self.input_data))

        self.term_2 = np.sum(self.input_data ** 2, axis=1)

        print 'contiguous?'
        print self.X.flags['C_CONTIGUOUS']
        print self.input_data_transpose.flags['C_CONTIGUOUS']

        linalg.init()
        self.i_d_t_gpu = gpuarray.to_gpu(self.input_data_transpose)

    def _get_mod(self):
        mod = SourceModule("""
            __global__ void knn_query(float *data, float *arr, float *tmp_data)
            {
              int dim = """ + str(DIM) + """;
              int idx = threadIdx.x;

              if (idx < dim)
              {
                  int k = 0;
                  float diff = 0.0;

                  int frames = """ + str(FRAMES) + """;

                  for (k = idx; k < idx + dim * frames; k += dim)
                  {
                        diff = abs(data[k] - arr[idx]);
                        tmp_data[k] = diff;
                  }
              }
              else
              {
                return;
              }
            }
            """)
        return mod

    def query(self, query_data):
        X = self.X
        X[0, :] = query_data[:]

        start_all = time.time()
        i_d_t_gpu = self.i_d_t_gpu

        #i_d_t_gpu = gpuarray.to_gpu(np.random.randn(4, 4).astype(np.float32))
        #X_gpu = gpuarray.to_gpu(np.random.randn(4, 4).astype(np.float32))

        X_gpu = gpuarray.to_gpu(X)
        start_1 = time.time()
        # DOESN'T WORK:
        # term_1 = gpuarray.dot(X_gpu, i_d_t_gpu).get()
        # WORKS:
        term_1 = linalg.dot(X_gpu, i_d_t_gpu).get()
        end_1 = time.time() - start_1

        start_2 = time.time()
        term_1 = -2 * term_1  # 15%
        end_2 = time.time() - start_2

        term_2 = self.term_2

        term_3 = np.sum(X ** 2, axis=1)[:, np.newaxis]

        dists = term_1 + term_2 + term_3  # 25%

        #ind = np.argmin(dists)  # <= 10%

        #dist = dists[0, ind]
        #end_all = time.time() - start_all

        # print 'ratio of time spent in dot product: ', end_1 / end_all
        # print 'ratio of time spent in 2: ', end_2 / end_all
        return dists


def run_cuda_test(test_seconds):
    data_frames = FRAMES
    dim = DIM
    data = np.random.random((data_frames, dim)).astype(np.float32)
    print 'data first: ', data[0, 0], data[0, -1]

    # --- start specific data init ---
    cuda_query = CudaQuery(input_data=data)
    # --- end specific data init ---

    start_time = time.time()
    last_time = time.time()
    last_frame = 0
    test_frames = 1000000
    np.random.seed(10)
    for frame in range(test_frames):

        query_data = np.random.random(dim).astype(np.float32)

        tmp = cuda_query.query(query_data=query_data)
        tmp = tmp[0]

        if frame == 0 or frame == 3:
            print '      *** frame ***', frame
            print '      query data: ', query_data[0], query_data[-1]
            print '      results: ', tmp[0:5]
        if time.time() - last_time > 1 or time.time() - start_time > test_seconds:
            FPS = (frame - last_frame) * 1.0 / (time.time() - last_time)
            print 'frame: ', frame, 'FPS: ', FPS
            last_frame = frame
            last_time = time.time()

            if time.time() - start_time > test_seconds:
                return


def run_numpy_test(test_seconds):
    data_frames = FRAMES
    dim = DIM
    data = np.random.random((data_frames, dim)).astype(np.float32)
    print 'data first: ', data[0, 0], data[0, -1]

    # --- start specific data init ---
    tmp = np.zeros(data_frames).astype(np.float).astype(np.float32)
    # --- end specific data init ---

    start_time = time.time()
    last_time = time.time()
    last_frame = 0
    test_frames = 100000
    np.random.seed(10)
    for frame in range(test_frames):

        query_data = np.random.random(dim).astype(np.float32)

        ret = matrix_vect_dist_numpy(data, query_data, tmp, data_frames, dim)
        assert ret == 1

        if frame == 0 or frame == 3:
            print '      *** frame ***', frame
            print '      query data: ', query_data[0], query_data[-1]
            print '      results: ', tmp[0:5]
        if time.time() - last_time > 5:
            FPS = (frame - last_frame) * 1.0 / (time.time() - last_time)
            print 'frame: ', frame, 'FPS: ', FPS
            last_frame = frame
            last_time = time.time()
        if time.time() - start_time > test_seconds:
            return


def run_cython_knn_test(test_seconds):
    data_frames = FRAMES
    dim = DIM
    data = np.random.random((data_frames, dim)).astype(np.float32)
    print 'data first: ', data[0, 0], data[0, -1]

    # --- start specific data init ---
    tmp = np.zeros(data_frames).astype(np.float).astype(np.float32)
    # --- end specific data init ---

    start_time = time.time()
    last_time = time.time()
    last_frame = 0
    test_frames = 100000
    np.random.seed(10)
    for frame in range(test_frames):

        query_data = np.random.random(dim).astype(np.float32)

        ret = matrix_vect_dist_cython(data, query_data, tmp, data_frames, dim)
        assert ret == 1

        if frame == 0 or frame == 3:
            print '      *** frame ***', frame
            print '      query data: ', query_data[0], query_data[-1]
            print '      results: ', tmp[0:5]
        if time.time() - last_time > 5:
            FPS = (frame - last_frame) * 1.0 / (time.time() - last_time)
            print 'frame: ', frame, 'FPS: ', FPS
            last_frame = frame
            last_time = time.time()
        if time.time() - start_time > test_seconds:
            return


if __name__ == '__main__':
    '''

    '''

    test_seconds = 11

    print
    print '--- cuda ---'
    np.random.seed(2)
    run_cuda_test(test_seconds=test_seconds)

    print
    print '--- cython ---'
    np.random.seed(2)
    run_cython_knn_test(test_seconds=test_seconds)

    print
    print '--- cpu ---'
    np.random.seed(2)
    run_numpy_test(test_seconds=test_seconds)
