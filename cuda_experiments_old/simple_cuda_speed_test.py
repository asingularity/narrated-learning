
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
FRAMES = 25600 #1000000 * 4
DIM = 20 #40 * 3


def cuda_init(data):
    data_gpu = cuda.mem_alloc(data.size * data.dtype.itemsize)

    abs_dists_tmp = np.zeros(FRAMES).astype(np.float32)
    abs_dists_gpu = cuda.mem_alloc(abs_dists_tmp.size * abs_dists_tmp.dtype.itemsize)

    mod = SourceModule("""
        __global__ void knn_query(float *data, float *abs_dists)
        {
            int dim = """ + str(DIM) + """;
            int frames = """ + str(FRAMES) + """;
            //int idx = threadIdx.x + threadIdx.y * 16; // 4;
            int idx = threadIdx.x + blockIdx.x* blockDim.x;
            //int idx = blockIdx.x;

            int num_threads = 256;
            for (int k = idx; k < dim * frames; k+=num_threads)
            {
                data[k] += 0.1;
            }
        }
        """)


    func = mod.get_function("knn_query")
    cuda.memcpy_htod(data_gpu, data)

    return data_gpu, abs_dists_gpu, func


def cuda_query(cuda_data_gpu, data, cuda_abs_dists_gpu, abs_dists, func):
    cuda.memcpy_htod(cuda_abs_dists_gpu, abs_dists)

    #func(cuda_data_gpu, block=(256, 1, 1))
    func(cuda_data_gpu, block=(32, 1, 1), grid=(256 / 32, 1))

    cuda.memcpy_dtoh(data, cuda_data_gpu)


def run_cuda_test(test_seconds):
    data_frames = FRAMES
    dim = DIM
    data = np.random.random((data_frames, dim)).astype(np.float32)
    abs_dists = np.zeros(FRAMES).astype(np.float32)
    print 'data first: ', data[0, 0], data[0, -1]

    cuda_data_gpu, cuda_abs_dists_gpu, cuda_func = cuda_init(data)

    start_time = time.time()
    last_time = time.time()
    last_frame = 0

    test_frames = 1000000

    np.random.seed(10)
    for frame in range(test_frames):
        cuda_query(cuda_data_gpu, data, cuda_abs_dists_gpu, abs_dists, cuda_func)

        if frame < 10:
            print '      *** frame ***', frame
            print '      query data: ', data[0], data[-1]

        if time.time() - last_time > 1 or time.time() - start_time > test_seconds:
            FPS = (frame - last_frame) * 1.0 / (time.time() - last_time)
            print 'frame: ', frame, 'FPS: ', FPS
            last_frame = frame
            last_time = time.time()

            if time.time() - start_time > test_seconds:
                return


if __name__ == '__main__':
    test_seconds = 6
    np.random.seed(0)
    print '--- cuda ---'
    run_cuda_test(test_seconds=test_seconds)
