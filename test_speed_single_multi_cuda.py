import time
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import skcuda.linalg as linalg
import skcuda.misc as misc

from cuda_multi_table import DumbCudaMultiTable, CudaMultiTable
from utils.load_sim_history import load_states_history, load_td_info_history
from utils.fps_counter import FPSCounter
from brain_components_classes.states_history import StatesLimitedHistory


NL_SIM_DIR = '/srv/projects/NL-sim/'


'''

    test_type = 0

    if test_type == 0:
        num_entries_gpu = num_entries
        num_tables_gpu = num_tables
        query_rows = num_tables
    else:
        num_entries_gpu = num_tables * num_entries
        num_tables_gpu = 1
        query_rows = num_tables


'''


#@profile
def run_basic_ops_speed_test():
    '''

    :return:
    '''

    fps = FPSCounter(params={'display_every_k_seconds': 1})

    linalg.init()

    test_secs = 5

    input_dim = 96
    num_tables = 1
    num_entries = 8000
    query_rows = 1000  # num_tables

    num_entries_gpu = num_entries
    num_tables_gpu = num_tables

    use_kernel = False

    table_i_only = np.random.random((num_entries_gpu, input_dim)).astype(np.float32)
    table_i_gpu = gpuarray.to_gpu(np.ascontiguousarray(np.transpose(table_i_only)))

    t0 = time.time()
    sum = None
    while time.time() < t0 + test_secs:
        fps.update()

        if use_kernel:
            pass
        else:

            for tmp in range(num_tables_gpu):
                query_arr = np.random.random((query_rows, input_dim)).astype(np.float32)

                X_gpu = gpuarray.to_gpu(query_arr)
                term_1 = linalg.dot(X_gpu, table_i_gpu).get()

                # make sure it is using the retrieved values, in case that matters
                if sum is None:
                    sum = term_1[0].copy()
                else:
                    sum += term_1[0]

    print(sum[0], sum[1], sum[-2], sum[-1])
    # linalg.dot (query)

    # set_by_index (insert)

    # (?) all to all distances


def run_multi_table_speed_tests():
    '''

    (1) for 1, 10 tables:
            DumbCudaMultiTable.query
            CudaMultiTable.query

    (2) for 1, 10 tables:
        (init)
            DumbCudaMultiTable.set_matrix_row, with fast init True
            CudaMultiTable.set_matrix_row, with fast init True
        (after)
            DumbCudaMultiTable.set_matrix_row, with fast init False
            CudaMultiTable.set_matrix_row, with fast init False

    Also: need to fix cuda_table_test.py and update for new functionality!
        Specifically need testing: INCLUDE_ADAPT_ROW, INCLUDE_CHANGE_ROW

    :return:
    '''

    sim_load_name = '48DIMx1M_states_positions_saved_2018-01-31T13:09:15.676826'
    plots_save_folder = NL_SIM_DIR + sim_load_name + '/'
    states_history = load_states_history(plots_save_folder)
    td_info_history = load_td_info_history(plots_save_folder)

    dim = states_history.shape[1]
    max_history_length = states_history.shape[0]

    # TODO also compare on 1 table with less entries and later after init to verify checksum holds on replace of rows

    test_type = 0

    if test_type == 0:
        num_entries = 8000
        num_tables = 10
        checksum_steps_after_init = 100
    else:
        num_entries = 800
        num_tables = 10
        checksum_steps_after_init = 1000

    test_set_after_init = True
    after_init_test_time = 6

    params = {'num_entries_list': [num_entries] * num_tables,
              'input_dim_list': [dim] * num_tables}

    for test_iteration in range(2):
        print()
        print('****************************************************************')
        np.random.seed(123)
        ran_checksum = False
        print()
        print('num_entries', num_entries)
        print('num_tables', num_tables)
        print()
        # DumbCudaMultiTable, CudaMultiTable
        if test_iteration == 0:
            print('*** DumbCudaMultiTable ***')
            tables = DumbCudaMultiTable(params=params)
        else:
            print('*** CudaMultiTable ***')
            tables = CudaMultiTable(params=params)
        print()

        fps = FPSCounter(params={'display_every_k_seconds': 1})

        print()
        print('--- INIT: fast_init=True ---')
        print()

        t_start = np.inf

        t = 0
        while time.time() < t_start + after_init_test_time or not ran_checksum:

            fps.update()

            input_state = states_history[t, :]

            row_inputs = [input_state] * num_tables

            dists = tables.query(query_inputs=row_inputs)
            assert len(dists) == num_tables

            if t < num_entries:
                row_indices = [t] * num_tables

                for k in range(num_tables):
                    dists[k][row_indices[k]] = np.inf

                tables.set_matrix_rows(table_indices=list(range(num_tables)),
                                       row_indices=row_indices,
                                       row_inputs=row_inputs,
                                       rows_to_table_dists=dists,
                                       fast_init=[True]*num_tables)
            elif t == num_entries:
                tables.post_init()
                print()
                print('--- RUN: fast_init=False ---')
                print()
                t_start = time.time()
            else:
                if test_set_after_init:
                    row_indices = list(np.random.randint(0, num_entries, num_tables))

                    for k in range(num_tables):
                        dists[k][row_indices[k]] = np.inf

                    tables.set_matrix_rows(table_indices=list(range(num_tables)),
                                           row_indices=row_indices,
                                           row_inputs=row_inputs,
                                           rows_to_table_dists=dists,
                                           fast_init=[False]*num_tables)

                if t == num_entries + checksum_steps_after_init:
                    print()
                    print('checksum:', dists[0][0], dists[0][10], dists[1][3], dists[1][123])
                    print()
                    ran_checksum = True
            t += 1

if __name__ == '__main__':
    run_basic_ops_speed_test()
    #run_multi_table_speed_tests()
