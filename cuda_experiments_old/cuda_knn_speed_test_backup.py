
import numpy as np
import time
from knn_cython import knn_query
from knn_parallel import knn_query as knn_parallel_query
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath

SMALL_TEST = 1

USERNAME = 'intec'
if SMALL_TEST:
    FRAMES = 25600 * 2
    DIM = 20
else:
    FRAMES = 1000000 * 2  # * 4
    DIM = 40 * 3


class CudaQuery(object):
    def __init__(self, input_data):
        self.original_input_data_shape = input_data.shape
        query_data = np.zeros(input_data.shape[1], np.float32)

        self.input_data = input_data.flatten()  # row-major order. do not have to flatten.
        self.tmp_data = np.zeros(self.input_data.shape, np.float32)

        mod = self._get_mod()
        self.func = mod.get_function("knn_query")

        self.input_data_gpu = cuda.mem_alloc(self.input_data.size * self.input_data.dtype.itemsize)
        self.query_data_gpu = cuda.mem_alloc(self.input_data.size * self.input_data.dtype.itemsize)
        self.tmp_data_gpu = cuda.mem_alloc(self.tmp_data.size * self.tmp_data.dtype.itemsize)

        cuda.memcpy_htod(self.input_data_gpu, self.input_data)

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
                        diff = abs(data[k] - arr[k]);
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

        query_data_expanded = np.zeros(self.original_input_data_shape, np.float32)
        query_data_expanded[:, :] = query_data[:]
        query_data_expanded = query_data_expanded.flatten()

        cuda.memcpy_htod(self.query_data_gpu, query_data_expanded)
        self.func(self.input_data_gpu, self.query_data_gpu, self.tmp_data_gpu, block=(128, 1, 1))

        # TODO replace these with cuda-based sum routine
        cuda.memcpy_dtoh(self.tmp_data, self.tmp_data_gpu)
        dists = np.sum(self.tmp_data.reshape(self.original_input_data_shape), axis=1)
        ind = np.argmin(dists)
        dist = dists[ind]

        #dist = 0
        #ind = 0

        return dist, ind


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

        dist, ind = cuda_query.query(query_data=query_data)

        if frame == 0 or frame == 3:
            print '      *** frame ***', frame
            print '      query data: ', query_data[0], query_data[-1]
            print '      results: ', dist, ind
        if time.time() - last_time > 1 or time.time() - start_time > test_seconds:
            FPS = (frame - last_frame) * 1.0 / (time.time() - last_time)
            print 'frame: ', frame, 'FPS: ', FPS
            last_frame = frame
            last_time = time.time()

            if time.time() - start_time > test_seconds:
                return


def run_cython_knn_test(test_seconds, parallel=False):
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

        if parallel:
            dist, ind = knn_parallel_query(data, query_data, tmp, data_frames, dim)
        else:
            dist, ind = knn_query(data, query_data, data_frames, dim)
        if frame == 0 or frame == 3:
            print '      *** frame ***', frame
            print '      query data: ', query_data[0], query_data[-1]
            print '      results: ', dist, ind
        if time.time() - last_time > 5:
            FPS = (frame - last_frame) * 1.0 / (time.time() - last_time)
            print 'frame: ', frame, 'FPS: ', FPS
            last_frame = frame
            last_time = time.time()
        if time.time() - start_time > test_seconds:
            return


if __name__ == '__main__':

    test_seconds = 6

    # print '--- cython single core ---'
    # np.random.seed(0)
    # run_cython_knn_test(test_seconds=test_seconds, parallel=False)

    print '--- cuda ---'
    np.random.seed(0)
    run_cuda_test(test_seconds=test_seconds)

    print '--- cython parallel ---'
    np.random.seed(0)
    run_cython_knn_test(test_seconds=test_seconds, parallel=True)

