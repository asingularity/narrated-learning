
import time
import timeit
import numpy as np
from matrix_vector_dist_parallel import knn_query as matrix_vect_dist_cython
from brute_force_knn import matrix_vect_dist_numpy
from cuda_dist_query import CudaTable

USERNAME = 'intec'


SMALL_TEST = 0
if SMALL_TEST:
    FRAMES = 40000
    DIM = 48
else:
    FRAMES = 10000
    DIM = 20000

INCLUDE_ADAPT_ROW = True
INCLUDE_SORTED_DIST = True
INCLUDE_CHANGE_ROW = True
# TODO test adapt of a row


def run_cuda_test(test_seconds):
    data_frames = FRAMES
    dim = DIM
    data = np.random.random((data_frames, dim)).astype(np.float32)
    print 'data gigabytes:', (data.size * 4.0) / (1e9)
    print
    print 'data first: ', data[0, 0], data[0, -1]

    new_row_3 = data[3, :] * 2.0  # * 0.9 #* 2.0
    # --- start specific data init ---

    input_dim = dim / 3
    output_dim = dim / 3
    context_dim = dim - input_dim - output_dim

    cuda_query = CudaTable(num_entries=data_frames,
                           input_dim=input_dim,
                           output_dim=output_dim,
                           context_dim=context_dim,
                           table=data,
                           include_layers=['ioc'])
    # --- end specific data init ---

    start_time = time.time()
    last_time = time.time()
    last_frame = 0
    test_frames = 1000000
    np.random.seed(10)

    t_spent_query = 0.0
    t_spent_sort = 0.0

    for frame in range(test_frames):

        query_data = np.random.random(dim).astype(np.float32)

        if INCLUDE_ADAPT_ROW and frame == 5:
            # when run on every frame: this test slows down FPS about 30%

            print('(testing adapt row)')
            i_0, o_0, c_0 = cuda_query.get_matrix_row(row_index=5)
            cuda_query.adapt(row_index=5,
                             rate=0.5,
                             row_input=new_row_3[0:input_dim],
                             row_output=new_row_3[input_dim:input_dim + output_dim],
                             row_context=new_row_3[input_dim + output_dim::])
            i_1, o_1, c_1 = cuda_query.get_matrix_row(row_index=5)
            r_0 = np.concatenate((i_0, o_0, c_0))
            r_1 = np.concatenate((i_1, o_1, c_1))
            r_1_test = r_0 * 0.5 + new_row_3 * 0.5
            assert np.sum(np.fabs(r_1 - r_1_test)) == 0.0

            cuda_query.set_matrix_row(row_index=5,
                                      row_input=i_0,
                                      row_output=o_0,
                                      row_context=c_0)

        if INCLUDE_CHANGE_ROW:
            cuda_query.set_matrix_row(row_index=3,
                                      row_input=new_row_3[0:input_dim],
                                      row_output=new_row_3[input_dim:input_dim + output_dim],
                                      row_context=new_row_3[input_dim + output_dim::])

        t0 = time.time()
        tmp = cuda_query.query(query_input=query_data[0:input_dim],
                               query_output=query_data[input_dim:input_dim + output_dim],
                               query_context=query_data[input_dim + output_dim::])

        t_spent_query += time.time() - t0

        if INCLUDE_SORTED_DIST:
            t0 = time.time()
            sorted_dist_indices = np.argsort(tmp)
            t_spent_sort += time.time() - t0

        if frame == 0 or frame == 3:
            print '      *** frame ***', frame
            print '      query data: ', query_data[0], query_data[-1]
            print '      results: ', tmp[0:5]
            if INCLUDE_SORTED_DIST:
                print '      sorted dist ind: ', sorted_dist_indices[0], sorted_dist_indices[-1]
        if time.time() - last_time > 1 or time.time() - start_time > test_seconds:
            FPS = (frame - last_frame) * 1.0 / (time.time() - last_time)
            print 'frame: ', frame, 'FPS: ', FPS, 'query:', t_spent_query / (t_spent_sort + t_spent_query), 'sort:', t_spent_sort / (t_spent_query + t_spent_sort)
            last_frame = frame
            last_time = time.time()

            if time.time() - start_time > test_seconds:
                return


def run_numpy_test(test_seconds):
    data_frames = FRAMES
    dim = DIM
    data = np.random.random((data_frames, dim)).astype(np.float32)
    print 'data first: ', data[0, 0], data[0, -1]

    new_row_3 = data[3, :] * 2.0  # * 0.9 #* 2.0
    # --- start specific data init ---
    tmp = np.zeros(data_frames).astype(np.float).astype(np.float32)
    # --- end specific data init ---

    start_time = time.time()
    last_time = time.time()
    last_frame = 0
    test_frames = 100000
    np.random.seed(10)

    t_spent_query = 0.0
    t_spent_sort = 0.0

    for frame in range(test_frames):

        query_data = np.random.random(dim).astype(np.float32)

        if INCLUDE_CHANGE_ROW:
            data[3, :] = new_row_3

        t0 = time.time()
        ret = matrix_vect_dist_numpy(data, query_data, tmp, data_frames, dim)
        t_spent_query += time.time() - t0

        if INCLUDE_SORTED_DIST:
            t0 = time.time()
            sorted_dist_indices = np.argsort(tmp)
            t_spent_sort += time.time() - t0

        assert ret == 1

        if frame == 0 or frame == 3:
            print '      *** frame ***', frame
            print '      query data: ', query_data[0], query_data[-1]
            if INCLUDE_SORTED_DIST:
                print '      sorted dist ind: ', sorted_dist_indices[0], sorted_dist_indices[-1]
            print '      results: ', tmp[0:5]
        if time.time() - last_time > 5:
            FPS = (frame - last_frame) * 1.0 / (time.time() - last_time)
            print 'frame: ', frame, 'FPS: ', FPS, 'query:', t_spent_query / (t_spent_sort + t_spent_query), 'sort:', t_spent_sort / (t_spent_query + t_spent_sort)
            last_frame = frame
            last_time = time.time()
        if time.time() - start_time > test_seconds:
            return


def run_cython_knn_test(test_seconds):
    data_frames = FRAMES
    dim = DIM
    data = np.random.random((data_frames, dim)).astype(np.float32)
    print 'data first: ', data[0, 0], data[0, -1]

    new_row_3 = data[3, :] * 2.0  # * 0.9 #* 2.0
    # --- start specific data init ---
    tmp = np.zeros(data_frames).astype(np.float).astype(np.float32)
    # --- end specific data init ---

    start_time = time.time()
    last_time = time.time()
    last_frame = 0
    test_frames = 100000
    np.random.seed(10)

    t_spent_query = 0.0
    t_spent_sort = 0.0

    for frame in range(test_frames):

        query_data = np.random.random(dim).astype(np.float32)

        if INCLUDE_CHANGE_ROW:
            data[3, :] = new_row_3

        t0 = time.time()
        ret = matrix_vect_dist_cython(data, query_data, tmp, data_frames, dim)
        t_spent_query += time.time() - t0

        if INCLUDE_SORTED_DIST:
            t0 = time.time()
            sorted_dist_indices = np.argsort(tmp)
            t_spent_sort += time.time() - t0

        assert ret == 1

        if frame == 0 or frame == 3:
            print '      *** frame ***', frame
            print '      query data: ', query_data[0], query_data[-1]
            print '      results: ', tmp[0:5]
            if INCLUDE_SORTED_DIST:
                print '      sorted dist ind: ', sorted_dist_indices[0], sorted_dist_indices[-1]
        if time.time() - last_time > 5:
            FPS = (frame - last_frame) * 1.0 / (time.time() - last_time)
            print 'frame: ', frame, 'FPS: ', FPS, 'query:', t_spent_query / (t_spent_sort + t_spent_query), 'sort:', t_spent_sort / (t_spent_query + t_spent_sort)
            last_frame = frame
            last_time = time.time()
        if time.time() - start_time > test_seconds:
            return


if __name__ == '__main__':
    '''

    '''

    # TODO also measure time spent in each operation
    #   timeit?

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
