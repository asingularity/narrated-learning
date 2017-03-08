
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
FRAMES = 1000000 * 4
DIM = 40 * 3

def run_test_incremental():
    frames = FRAMES
    dim = DIM
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
    data_frames = FRAMES
    dim = DIM
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
    data_frames = FRAMES
    dim = DIM
    data = np.random.random((data_frames, dim)).astype(np.float32)
    print 'data first: ', data[0, 0], data[0, -1]
    tmp = np.zeros(data_frames).astype(np.float).astype(np.float32)
    start_time = time.time()
    last_time = time.time()
    last_frame = 0

    test_frames = 10000

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


def cuda_init(data, query_data):
    data_gpu = cuda.mem_alloc(data.size * data.dtype.itemsize)
    arr_gpu = cuda.mem_alloc(query_data.size * query_data.dtype.itemsize)

    num_threads = 16 * 16
    min_indices_tmp = np.ones(num_threads).astype(np.long)
    min_indices_gpu = cuda.mem_alloc(min_indices_tmp.size * min_indices_tmp.dtype.itemsize)
    min_dists_tmp = np.ones(num_threads).astype(np.float32)
    min_dists_gpu = cuda.mem_alloc(min_dists_tmp.size * min_dists_tmp.dtype.itemsize)

    # num_entries_per_thread = 4000000 / (4 * 4)
    # min_indices, min_values: storing result, one index is one result from one thread
    #   indexed by idx

    mod = SourceModule("""
        __global__ void knn_query(float *data, float *arr, long *min_indices, float *min_dists)
        {
          long num_entries_per_thread = 15625; //250000;
          int dim = 40 * 3;
          long lda = 4000000;
          int idx = threadIdx.x + threadIdx.y * 16; // 4;

          long r0 = idx * num_entries_per_thread;
          long r1 = r0 + num_entries_per_thread;
          float min_dist = 9999.9;
          long min_index = -1;
          float dist = 0.0;
          long k = 0;
          float diff = 0.0;

          for (k = r0; k < r1; k++)
          {
                dist = 0.0;

                for (int dim_index = 0; dim_index < dim; dim_index++)
                {
                    diff = data[dim_index + k * dim] - arr[dim_index];
                    dist = dist + abs(diff);
                }

                if (dist < min_dist)
                {
                    min_dist = dist;
                    min_index = k;
                }
          }

          min_dists[idx] = min_dist;
          min_indices[idx] = min_index;
        }
        """)
    func = mod.get_function("knn_query")
    cuda.memcpy_htod(data_gpu, data)
    cuda.memcpy_htod(min_dists_gpu, min_dists_tmp)
    cuda.memcpy_htod(min_indices_gpu, min_indices_tmp)

    return data_gpu, arr_gpu, min_indices_gpu, min_indices_tmp, min_dists_gpu, min_dists_tmp, func

#@profile
def cuda_query(query_data, cuda_data_gpu, cuda_arr_gpu, func, b_doubled, data, min_indices_gpu, min_indices_tmp, min_dists_gpu, min_dists_tmp):

    # copying entire KNN table over again:
    # this slows down things a lot!
    # cuda.memcpy_htod(cuda_data_gpu, data)

    cuda.memcpy_htod(cuda_arr_gpu, query_data)

    func(cuda_data_gpu, cuda_arr_gpu, min_indices_gpu, min_dists_gpu, block=(16, 16, 1))

    cuda.memcpy_dtoh(min_dists_tmp, min_dists_gpu)
    cuda.memcpy_dtoh(min_indices_tmp, min_indices_gpu)

    argm = np.argmin(min_dists_tmp)

    dist = min_dists_tmp[argm]
    ind = min_indices_tmp[argm]

    return dist, ind


def run_cuda_test(test_seconds):
    data_frames = FRAMES
    dim = DIM
    data = np.random.random((data_frames, dim)).astype(np.float32)
    print 'data first: ', data[0, 0], data[0, -1]

    query_data = np.random.random(dim).astype(np.float32)
    b_doubled = np.empty_like(query_data)
    cuda_data_gpu, cuda_arr_gpu, min_indices_gpu, min_indices_tmp, min_dists_gpu, min_dists_tmp, cuda_func = cuda_init(data, query_data)

    start_time = time.time()
    last_time = time.time()
    last_frame = 0

    test_frames = 1000000

    np.random.seed(10)
    for frame in range(test_frames):
        query_data = np.random.random(dim).astype(np.float32)
        dist, ind = cuda_query(query_data, cuda_data_gpu, cuda_arr_gpu, cuda_func, b_doubled, data, min_indices_gpu, min_indices_tmp, min_dists_gpu, min_dists_tmp)

        if frame == 0 or frame == 3:
            print '      *** frame ***', frame
            print '      query data: ', query_data[0], query_data[-1]
            print '      results: ', dist, ind
        if time.time() - last_time > 1:
            FPS = (frame - last_frame) * 1.0 / (time.time() - last_time)
            print 'frame: ', frame, 'FPS: ', FPS
            last_frame = frame
            last_time = time.time()
        if time.time() - start_time > test_seconds:
            return


def knn_save_test(optimized):
    data_frames = FRAMES
    dim = DIM
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
        #print '--- numpy ---'
        #run_test_full(tree=False, test_seconds=test_seconds)
        np.random.seed(0)
        #print '--- cython ---'
        #run_cython_knn_test(test_seconds=test_seconds)
        np.random.seed(0)
        print '--- cython parallel ---'
        run_cython_knn_test(test_seconds=test_seconds, parallel=True)
