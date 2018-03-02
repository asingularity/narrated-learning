
import time
import numpy as np
from matrix_vector_dist_parallel import knn_query as matrix_vect_dist_cython
from brute_force_knn import matrix_vect_dist_numpy
from cuda_dist_query import CudaQuery

USERNAME = 'intec'


SMALL_TEST = 0
if SMALL_TEST:
    FRAMES = 2000000
    DIM = 8 * 3
else:
    FRAMES = 10000
    DIM = 40000


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
