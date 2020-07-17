
import random
from utils.fps_counter import FPSCounter
import time
import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import skcuda.linalg as linalg
import skcuda.misc as misc

from scipy.sparse import lil_matrix

from cython_match import sum_match, learn_update

#import h5py
#import bcolz


DTYPE = np.int32


def arr_size_gb(arr):
    '''

    :param arr: two-dimensional array, of a 32-bit type
    :return:
    '''

    r = arr.shape[0]
    c = arr.shape[1]

    # assume 32 bits
    bits = r * c * 32
    bytes = bits / 8
    gb = bytes / (1e9)
    return gb


class MultiSparseBinaryKNN(object):
    def __init__(self, params):
        '''

        Multiple independent SparseBinaryKNN, grouped for computational efficiency
        Uses GPU like cuda table, for computational efficiency

        :param params:

            'num_knn': (how many knn)
            'sparse_io_dim': self.rows_per_tile     (per knn)
            'num_sparse_inputs': num_predictor_tiles * len(self.prediction_tau_list)    (per knn)
            'num_sparse_outputs': 1     (per knn)
            'num_rows': 2000
        '''

        # for now, we use cython parallelization before full cuda implementation

        # ************ global vars ************

        # for now, assume only one sparse output, with one-hot
        assert params['num_sparse_outputs'] == 1

        # this is max_index + 1 that we would encounter;
        # ie [0, sparse_io_dim-1] is range of input or output index per sparse io unit
        self.sparse_io_dim = params['sparse_io_dim']
        self.num_sparse_inputs = params['num_sparse_inputs']
        self.N = params['num_rows']  # total number of rows learned per KNN

        self.num_knn = params['num_knn']

        # ************ per knn vars ************

        self.learn_index = np.zeros(self.num_knn, DTYPE)  # how many rows have we learned so far

        # ************ knn **************

        if params['random_init']:
            assert False, 'needs implement!'
            self.input_arr = np.random.randint(0, self.sparse_io_dim, (self.num_knn * self.N, self.num_sparse_inputs), DTYPE)  # int because this is just indices. dim=num_sparse_inputs since assuming one-hot for now on input
            self.output_prop_count = np.random.randint(1, 100, (self.num_knn * self.N, self.sparse_io_dim)).astype(np.float32)
            self.output_prop_arr = np.random.random((self.num_knn * self.N, self.sparse_io_dim)).astype(np.float32)
        else:
            self.input_arr = np.zeros((self.num_knn * self.N, self.num_sparse_inputs), DTYPE)  # int because this is just indices. dim=num_sparse_inputs since assuming one-hot for now on input

            print('starting sparse init...')
            self.output_prop_count = lil_matrix((self.num_knn * self.N, self.sparse_io_dim), dtype=np.float32)
            self.output_prop_count.tocsr()

            print('done sparse init.')

        print()
        print('Init of multi sparse binary knn')
        print('    input_arr.shape:', self.input_arr.shape, ', size in GB:', arr_size_gb(self.input_arr))
        print('    output_prop_count.shape:', self.input_arr.shape, ', size in GB:', arr_size_gb(self.output_prop_count))
        # print('    output_prop_arr.shape:', self.input_arr.shape, ', size in GB:', arr_size_gb(self.output_prop_arr))
        print()

        # debug printing
        self.printed_message = [False, False]
        self.messages = ['SparseBinaryKNN::train: learning is started!',
                         'SparseBinaryKNN::train: learning is completed! Weight learning started!']

        self.reset_stats_every_k_sec = 10
        self.last_stat_reset = time.time()
        self.match_ratio_sum = 0.0
        self.match_ratio_num = 0
        self.sum_tie_matches = 0.0

        self.skipped_train_cnt = 0
        self.done_train_cnt = 0

        # cython
        self.use_cython = params['use_cython']
        assert self.use_cython, "no longer supporting non-cython"

        # cuda
        self.use_cuda = params['use_cuda']
        if self.use_cuda:
            assert False, "cuda implementation not implemented yet!"
            linalg.init()

            self.input_arr_gpu = gpuarray.to_gpu(self.input_arr)

        self.t = 0  # only used for debug / printing
        self.printed_messages = []

    def _print_message_once(self, message):
        if not message in self.printed_messages:
            print()
            print(self.t, ' ', message)
            print()
            self.printed_messages.append(message)

    # @profile
    def predict(self, knn_input_win_rows_2d_arr):
        '''

        :param knn_input_win_rows_list: list of arrays, one array per knn (or 2D / extended array?)
                each array len: num_sparse_inputs
        :return: win_rows_arr: list of win rows, len: number of knn (or array)
        '''

        # knn_input_win_rows_2d_arr: (self.num_knn * self.num_sparse_inputs)

        assert knn_input_win_rows_2d_arr.shape[0] == self.num_knn
        assert knn_input_win_rows_2d_arr.shape[1] == self.num_sparse_inputs

        # knn_input_win_rows_2d_arr:  # (196, 63) : (num_knn, num_sparse_inputs)
        # self.input_arr:             # (392000, 63) : (num_knn * N, num_sparse_inputs)

        # if self.use_cuda:
        #     expand_input_gpu = gpuarray.to_gpu(expand_input)
        #     tmp_gpu = misc.sum(misc.subtract(self.input_arr_gpu, expand_input_gpu)==0, axis=1)
        #     tmp = tmp_gpu.get()

        arr_out = np.zeros(self.input_arr.shape[0], DTYPE)
        sum_match(self.input_arr, knn_input_win_rows_2d_arr, arr_out, DTYPE(self.num_knn), DTYPE(self.N), self.learn_index)
        tmp = arr_out
        # tmp: (392000,)

        # now calc win rows: argmax per block of N
        # or, we reshape appropriately

        tmp2 = tmp.reshape((self.num_knn, self.N))
        win_knn_rows_arr = np.argmax(tmp2, axis=1)

        # win_knn_rows_arr: (self.num_knn,)  -- values in range [0, self.N]
        # self.output_prop_arr: (self.num_knn * self.N, self.sparse_io_dim)

        rel_knn_rows = np.arange(self.num_knn) * self.N + win_knn_rows_arr  # relative indices to output_prop_arr
        win_rows_arr = np.argmax(self.output_prop_count[rel_knn_rows, :].toarray(), axis=1)

        self.t += 1
        assert win_rows_arr.shape[0] == self.num_knn
        return win_rows_arr

    # @profile
    def train(self, knn_input_win_rows_2d_arr, knn_output_win_row_arr):
        '''

        :param knn_input_win_rows_list: list of arrays, one array per knn (or 2D / extended array?)
                    each array len: num_sparse_inputs
        :param knn_output_win_row_list: array, len: number of knn
        :return:
        '''

        # we can do this, but not tau - tau is internal to dynamic_tiles, but it is implicit here one step (next step)
        # other issue is what if between that tau, things changed?
        # better to not change this and go by original logic for now from sparse_binary_knn.py

        # (1) lookup based on input

        # (2) for each knn:
        #   if learn index < N
        #       if exact input row not already in table:
        #           learn this row, and set output with count 1
        #   else
        #       update output prop count and prop arr for winning row

        arr_out = np.zeros(self.input_arr.shape[0], DTYPE)
        sum_match(self.input_arr, knn_input_win_rows_2d_arr, arr_out, DTYPE(self.num_knn), DTYPE(self.N), self.learn_index)
        tmp = arr_out

        tmp2 = tmp.reshape((self.num_knn, self.N))
        win_knn_rows_arr = np.argmax(tmp2, axis=1).astype(DTYPE)

        # why did we have this here?
        # rel_knn_rows = np.arange(self.num_knn) * self.N + win_knn_rows_arr  # relative indices to output_prop_arr
        # win_rows_arr = np.argmax(self.output_prop_arr[rel_knn_rows, :].toarray(), axis=1)

        len_input = len(knn_input_win_rows_2d_arr[0, :])

        output_prop_count_incr_rows = np.zeros(self.num_knn, np.int32)
        output_prop_count_incr_cols = np.zeros(self.num_knn, np.int32)
        output_prop_count_incr_vals = np.zeros(self.num_knn, np.int32)

        learn_update(self.num_knn,
                     np.int32(self.N),
                     tmp,
                     self.learn_index,
                     self.input_arr,
                     np.int32(len_input),
                     knn_input_win_rows_2d_arr,
                     knn_output_win_row_arr,
                     win_knn_rows_arr,
                     output_prop_count_incr_rows,
                     output_prop_count_incr_cols,
                     output_prop_count_incr_vals)

        # now update sparse prop_count, recompute prop_arr for changed rows
        #   this probably needs the arrays to be in csr form to be fast computing

        self.output_prop_count[output_prop_count_incr_rows, output_prop_count_incr_cols] += output_prop_count_incr_vals

        # realization: don't even need output_prop_arr at all, count is sufficient


# @profile
def main():
    random.seed(1)
    np.random.seed(1)

    test_large = True

    if test_large:
        sparse_io_dim = 800
        num_knn = 3364
        num_sparse_inputs = 63
        num_rows = 2000
    else:
        sparse_io_dim = 800
        num_knn = 40
        num_sparse_inputs = 63
        num_rows = 200

    mp = MultiSparseBinaryKNN(params={
        'use_cuda': False,  # not implemented!
        'use_cython': True,  # must be True
        'sparse_io_dim': sparse_io_dim,  # rows per tile of cuda tables
        'num_sparse_inputs': num_sparse_inputs,
        'num_sparse_outputs': 1,
        'num_rows': num_rows,
        'num_knn': num_knn,
        'random_init': False  # for debugging
    })

    fps = FPSCounter()

    for k in range(1000):

        tiles_inputs = np.random.randint(0, sparse_io_dim - 1, (num_knn, num_sparse_inputs)).astype(DTYPE)
        predict_win_rows = mp.predict(knn_input_win_rows_2d_arr=tiles_inputs)
        mp.train(knn_input_win_rows_2d_arr=tiles_inputs, knn_output_win_row_arr=np.random.randint(0, sparse_io_dim - 1, num_knn).astype(DTYPE))
        fps.update()

    print(predict_win_rows)


if __name__ == '__main__':
    main()
















