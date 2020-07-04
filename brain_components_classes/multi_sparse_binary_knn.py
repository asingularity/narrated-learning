
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

from cython_match import sum_match


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

        self.input_arr = np.zeros((self.num_knn * self.N, self.num_sparse_inputs), DTYPE)  # int because this is just indices. dim=num_sparse_inputs since assuming one-hot for now on input
        self.output_arr = np.zeros(self.num_knn * self.N, DTYPE)  # dim=1 since assuming one-hot for now on output

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

        # cuda
        self.use_cuda = params['use_cuda']
        if self.use_cuda:
            linalg.init()

            self.input_arr_gpu = gpuarray.to_gpu(self.input_arr)

    def _print_message_once(self, index):
        if not self.printed_message[index]:
            print()
            print(self.messages[index])
            print()
            self.printed_message[index] = True

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

        expand_input = np.repeat(knn_input_win_rows_2d_arr, self.N, axis=0)

        # TODO incorporate learn_index, have to do it per block... or work around by appropriate initialization

        if self.use_cuda:
            expand_input_gpu = gpuarray.to_gpu(expand_input)
            tmp_gpu = misc.sum(misc.subtract(self.input_arr_gpu, expand_input_gpu)==0, axis=1)
            tmp = tmp_gpu.get()
        elif self.use_cython:
            #print(self.input_arr.dtype, expand_input.dtype)
            tmp = sum_match(self.input_arr, expand_input)
        else:
            tmp = np.sum((self.input_arr - expand_input) == 0, axis=1)
            # tmp = np.zeros(self.input_arr.shape[0])

        # tmp: (392000,)

        # now calc win rows: argmax per block of N
        # or, we reshape appropriately

        tmp2 = tmp.reshape((self.num_knn, self.N))
        win_knn_rows_arr = np.argmax(tmp2, axis=1)

        # TODO this is wrong:
        win_rows_arr = win_knn_rows_arr

        # TODO instead, missing a piece here: the lookup after win_knn_row
        # new way: use prop
        # win_knn_row = np.argmax(tmp)
        # win_row = np.argmax(self.output_prop_arr[win_knn_row, :])

        assert win_rows_arr.shape[0] == self.num_knn
        return win_rows_arr

    def train(self, knn_input_win_rows_2d_arr, knn_output_win_row_arr):
        '''

        :param knn_input_win_rows_list: list of arrays, one array per knn (or 2D / extended array?)
                    each array len: num_sparse_inputs
        :param knn_output_win_row_list: array, len: number of knn
        :return:
        '''



if __name__ == '__main__':
    # {'sparse_io_dim': 800, 'num_sparse_inputs': 63, 'num_sparse_outputs': 1, 'num_rows': 2000, 'learn_row_every_k': 1}

    np.random.seed(1)

    sparse_io_dim = 800
    num_knn = 196
    num_sparse_inputs = 63

    mp = MultiSparseBinaryKNN(params={
        'use_cuda': False,
        'use_cython': True,
        'sparse_io_dim': sparse_io_dim,  # rows per tile of cuda tables
        'num_sparse_inputs': num_sparse_inputs,
        'num_sparse_outputs': 1,
        'num_rows': 2000,
        'num_knn': num_knn
    })

    fps = FPSCounter()

    for k in range(1000):

        tiles_inputs = np.random.randint(0, sparse_io_dim - 1, (num_knn, num_sparse_inputs)).astype(DTYPE)
        predict_win_rows = mp.predict(knn_input_win_rows_2d_arr=tiles_inputs)

        # print(predict_win_rows)

        fps.update()


















