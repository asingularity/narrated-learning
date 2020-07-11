
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

from cython_match import sum_match, learn_update


DTYPE = np.int32


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


        # for now, we could use cython parallelization before full cuda implementation

        # ************ global vars ************

        # for now, assume only one sparse output, with one-hot
        assert params['num_sparse_outputs'] == 1

        # this is max_index + 1 that we would encounter;
        # ie [0, sparse_io_dim-1] is range of input or output index per sparse io unit
        self.sparse_io_dim = params['sparse_io_dim']
        self.num_sparse_inputs = params['num_sparse_inputs']
        self.N = params['num_rows']  # total number of rows learned per KNN

        self.num_knn = params['num_knn']

        # deprecated (unused) from SparseBinaryKNN:
        #   self.learn_every_k
        #   self.curr_k_step

        # ************ per knn vars ************

        self.learn_index = np.zeros(self.num_knn, DTYPE)  # how many rows have we learned so far

        # ************ knn **************

        if params['random_init']:
            self.input_arr = np.random.randint(0, self.sparse_io_dim, (self.num_knn * self.N, self.num_sparse_inputs), DTYPE)  # int because this is just indices. dim=num_sparse_inputs since assuming one-hot for now on input

            self.output_prop_count = np.random.randint(1, 100, (self.num_knn * self.N, self.sparse_io_dim)).astype(np.float32)
            self.output_prop_arr = np.random.random((self.num_knn * self.N, self.sparse_io_dim)).astype(np.float32)
        else:
            self.input_arr = np.zeros((self.num_knn * self.N, self.num_sparse_inputs), DTYPE)  # int because this is just indices. dim=num_sparse_inputs since assuming one-hot for now on input

            self.output_prop_count = np.zeros((self.num_knn * self.N, self.sparse_io_dim), np.float32)
            self.output_prop_arr = np.zeros((self.num_knn * self.N, self.sparse_io_dim), np.float32)



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

        if self.use_cython:
            self.input_arrs_list = []
            # try: one arr per knn instead of monolithic large array
            # for knn in range(self.num_knn):
            #     input_arr_knn = self.input_arr[knn * self.N:(knn + 1) * self.N, :].copy()
            #     self.input_arrs_list.append(input_arr_knn)

        # cuda
        self.use_cuda = params['use_cuda']
        if self.use_cuda:
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

    #@profile
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

        # TODO incorporate learn_index, have to do it per block... or work around by appropriate initialization
        # we could pass learn_index array to cython, easy to incorporate this there in the loop

        # if self.use_cuda:
        #     expand_input_gpu = gpuarray.to_gpu(expand_input)
        #     tmp_gpu = misc.sum(misc.subtract(self.input_arr_gpu, expand_input_gpu)==0, axis=1)
        #     tmp = tmp_gpu.get()

        if self.use_cython:
            arr_out = np.zeros(self.input_arr.shape[0], DTYPE)
            sum_match(self.input_arr, knn_input_win_rows_2d_arr, arr_out, DTYPE(self.num_knn), DTYPE(self.N), self.learn_index)
            tmp = arr_out
        else:
            expand_input = np.repeat(knn_input_win_rows_2d_arr, self.N, axis=0)
            tmp = np.sum((self.input_arr - expand_input) == 0, axis=1)

        # tmp: (392000,)

        # now calc win rows: argmax per block of N
        # or, we reshape appropriately

        tmp2 = tmp.reshape((self.num_knn, self.N))
        win_knn_rows_arr = np.argmax(tmp2, axis=1)

        # this is wrong but used for speed testing:
        # win_rows_arr = win_knn_rows_arr  # len: num_knn: winning knn-row per knn

        # this was line from single knn:
        # win_row = np.argmax(self.output_prop_arr[win_knn_row, :])

        # win_knn_rows_arr: (self.num_knn,)  -- values in range [0, self.N]
        # self.output_prop_arr: (self.num_knn * self.N, self.sparse_io_dim)

        rel_knn_rows = np.arange(self.num_knn) * self.N + win_knn_rows_arr  # relative indices to output_prop_arr
        win_rows_arr = np.argmax(self.output_prop_arr[rel_knn_rows, :], axis=1)

        self.t += 1
        assert win_rows_arr.shape[0] == self.num_knn
        return win_rows_arr

    #@profile
    def train(self, knn_input_win_rows_2d_arr, knn_output_win_row_arr):
        '''

        :param knn_input_win_rows_list: list of arrays, one array per knn (or 2D / extended array?)
                    each array len: num_sparse_inputs
        :param knn_output_win_row_list: array, len: number of knn
        :return:
        '''

        # possibly instead of passing in input rows:
        #   !!! unclear if this works given how make_prediction_knn, learn_prediction_knn work in dynamic_tiles.py !!!
        #   - tau_predict should be a class parameter, per knn
        #   - input should be stored in states limited history
        #
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

        if self.use_cython:
            arr_out = np.zeros(self.input_arr.shape[0], DTYPE)
            sum_match(self.input_arr, knn_input_win_rows_2d_arr, arr_out, DTYPE(self.num_knn), DTYPE(self.N), self.learn_index)
            tmp = arr_out
        else:
            expand_input = np.repeat(knn_input_win_rows_2d_arr, self.N, axis=0)
            tmp = np.sum((self.input_arr - expand_input) == 0, axis=1)

        tmp2 = tmp.reshape((self.num_knn, self.N))
        win_knn_rows_arr = np.argmax(tmp2, axis=1).astype(DTYPE)

        rel_knn_rows = np.arange(self.num_knn) * self.N + win_knn_rows_arr  # relative indices to output_prop_arr
        win_rows_arr = np.argmax(self.output_prop_arr[rel_knn_rows, :], axis=1)

        len_input = len(knn_input_win_rows_2d_arr[0, :])

        # move the below into cython as well!
        #   started learn_update function in cython_match.pyx

        if self.use_cython:
            '''
            def learn_update(np.int32_t num_knn,
                 np.int32_t N,
                 np.ndarray[np.int32_t, ndim=1] tmp,
                 np.ndarray[np.int32_t, ndim=1] learn_index,
                 np.ndarray[np.int32_t, ndim=2] input_arr,
                 np.int32_t len_input,
                 np.ndarray[np.int32_t, ndim=2] knn_input_win_rows_2d_arr,
                 np.ndarray[np.int32_t, ndim=2] output_prop_count,
                 np.ndarray[np.int32_t, ndim=2] output_prop_arr,
                 np.ndarray[np.int32_t, ndim=1] knn_output_win_row_arr,
                 np.ndarray[np.int32_t, ndim=1] win_knn_rows_arr):
            '''

            output_prop_sum_tmp_arr = np.zeros(self.num_knn, np.float32)
            # print(knn_input_win_rows_2d_arr.shape[1], knn_input_win_rows_2d_arr.shape, self.input_arr.shape)
            learn_update(self.num_knn,
                         np.int32(self.N),
                         tmp,
                         self.learn_index,
                         self.input_arr,
                         np.int32(len_input),
                         knn_input_win_rows_2d_arr,
                         self.output_prop_count,
                         self.output_prop_arr,
                         knn_output_win_row_arr,
                         win_knn_rows_arr,
                         output_prop_sum_tmp_arr)
        else:

            # print('')
            # print(win_knn_rows_arr.shape)  # (196,)
            # print(self.output_prop_count.shape)  # (392000, 800)
            # print(self.output_prop_count[win_knn_rows_arr[0], :].shape)  # (800,)

            for knn in range(self.num_knn):
                match = tmp[knn * self.N:(knn + 1) * self.N]
                if self.learn_index[knn] < self.N:
                    if np.amax(match) < len_input:  # not perfect match
                        self._print_message_once("--- at least one knn add a row")
                        self.input_arr[knn * self.N + self.learn_index[knn], :] = knn_input_win_rows_2d_arr[knn, :]
                        self.output_prop_count[knn * self.N + self.learn_index[knn], knn_output_win_row_arr[knn]] = 1
                        self.output_prop_arr[knn * self.N + self.learn_index[knn], knn_output_win_row_arr[knn]] = 1.0

                        self.learn_index[knn] += 1
                else:
                    self._print_message_once("--- at least one knn updated a row output")
                    self.output_prop_count[win_knn_rows_arr[knn], knn_output_win_row_arr[knn]] += 1
                    self.output_prop_arr[win_knn_rows_arr[knn], :] = self.output_prop_count[win_knn_rows_arr[knn], :] * 1.0 / np.sum(self.output_prop_count[win_knn_rows_arr[knn], :])


#@profile
def main():
    # {'sparse_io_dim': 800, 'num_sparse_inputs': 63, 'num_sparse_outputs': 1, 'num_rows': 2000, 'learn_row_every_k': 1}

    random.seed(1)
    np.random.seed(1)

    test_large = True

    if test_large:
        sparse_io_dim = 800
        num_knn = 196
        num_sparse_inputs = 63
        num_rows = 2000
    else:
        sparse_io_dim = 800
        num_knn = 40
        num_sparse_inputs = 63
        num_rows = 200

    mp = MultiSparseBinaryKNN(params={
        'use_cuda': False,  # not implemented!
        'use_cython': True,
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

    # TODO need a check on training that makes sure to include filled knn tables as well!!! i.e. executes all code above

    print(predict_win_rows)


if __name__ == '__main__':
    main()
















