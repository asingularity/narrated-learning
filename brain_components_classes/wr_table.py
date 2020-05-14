
import time
import cv2
import random
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import skcuda.linalg as linalg
import skcuda.misc as misc
from cuda_dist_query import CudaTable
from utils.fps_counter import FPSCounter


class WRTableLimitOneIn(object):
    def __init__(self, num_entries, input_dim, num_q=1, disable_row_row_dists=False):

        self.use_gpu = False

        self.num_entries = num_entries
        self.input_dim = input_dim

        assert num_q == 1
        self.num_q = num_q  # number of query input rows

        #table_i = np.random.random((num_entries, input_dim)).astype(np.float32)
        table_i = 0.5 * np.ones((num_entries, input_dim), np.float32)

        self.table_i = table_i
        self.all_diffs = np.zeros_like(self.table_i)

        self.tmp_i = 0

    def _init_num_query_inputs(self, num_query_inputs):
        '''

        use as:
            at start:
            self.num_query_inputs = None

            later:
            if self.num_query_inputs is None:
                self._init_num_query_inputs(num_query_inputs=query_inputs.shape[0])

        :param num_query_inputs:
        :return:
        '''

        assert False, 'unused right now'

        self.num_query_inputs = num_query_inputs
        # for 1200 X 256 X 1024, this takes 2 GB

        self.all_diffs = np.zeros((self.num_entries, self.input_dim), np.float32)
        # (1200, 256, 1024)

    # @profile

    def query_multiple_rows(self, query_inputs):
        assert query_inputs.shape[0] == 1, 'only one input row supported currently'
        input_row = query_inputs.flatten()  # one input row only

        self.table_i[self.tmp_i, :] = input_row[:]
        self.tmp_i += 1
        if self.tmp_i == self.table_i.shape[0]:
            self.tmp_i = 0


    def query_multiple_rows_OLD(self, query_inputs):
        '''

        :param query_inputs:
        :return:
        '''

        # self.table i:  (1200, 1024):  entries X pixels
        # query_inputs: (256, 1024):   tiles X pixels

        # calculate and store diff per pixel across all query_inputs and all rows

        # diff = self.table_i - query_inputs
        # print()
        # print(diff.shape)
        # print()

        # dists: (1, 1200)
        # argmin_dists: (1,)

        query_inputs_flatten = query_inputs.flatten()  # one input row only

        self.all_diffs[:, :] = self.table_i - query_inputs_flatten  # half the time
        self.all_diffs[:, :] = np.abs(self.all_diffs[:, :])  # half the time

        do_adapt = True

        if do_adapt:
            use_wr = True

            # TODO try that it only learns inside the win region?

            if use_wr:
                # compute win regions per table row, using all_diffs
                #   self.all_diffs.shape:  # (1200, 16384)
                winning_row_per_pixel = np.argmin(self.all_diffs, 0)  # (16384,)
                win_regions = np.zeros((self.num_entries, self.input_dim), np.int)
                win_regions[winning_row_per_pixel, np.arange(win_regions.shape[1])] = 1

                # try here: only learn inside win region, all learn equal learning rate
                learn_rate = 0.01
                keep_rate = 1.0 - learn_rate

                term_1 = keep_rate * self.table_i

                query_inputs_tiled = np.tile(query_inputs_flatten, (win_regions.shape[0], 1))
                term_2 = learn_rate * query_inputs_tiled

                nnz_win = np.nonzero(win_regions)
                self.table_i[nnz_win] = term_1[nnz_win] + term_2[nnz_win]

                if 0:
                    # converge towards mean
                    learning_rate =  0.0 * np.ones(self.table_i.shape[0], np.float32)
                    keep_rate = 1.0 - learning_rate

                    tmp1 = learning_rate[:, np.newaxis]
                    tmp2 = query_inputs_flatten[np.newaxis, :]
                    term_1 = np.dot(tmp1, tmp2)  # (1200, 16384)

                    term_2 = np.multiply(keep_rate[:, np.newaxis], self.table_i)  # (1200, 16384)

                    self.table_i = term_1 + term_2

            else:
                win_regions = np.ones((self.num_entries, self.input_dim), np.int)

                # for each input row: for all rows: compute average pixel error inside win regions

                wr_diff_sums = np.sum(np.multiply(self.all_diffs, win_regions), axis=1)  # length: num entries
                wr_diff_means = np.multiply(wr_diff_sums, 1.0 / np.sum(win_regions, axis=1))

                # for each input row: compute rank of table rows
                ranks = np.zeros(wr_diff_means.shape[0], np.int)
                argsort_diff_means = np.argsort(wr_diff_means)
                ranks[argsort_diff_means] = np.arange(wr_diff_means.shape[0])  # lower rank is smaller dist

                # for each input row: adapt proportionate to rank

                if 1:
                    base_rate = 0.01
                    learning_rate = base_rate * 0.01 * np.ones(wr_diff_means.shape[0], np.float32)
                    learning_rate[argsort_diff_means[0]] = base_rate

                if 0:
                    learn_tmp = ranks * 1.0 / ranks.shape[0]
                    learn_tmp = np.power(1.0 - learn_tmp, 4)

                    learning_rate_base = 0.001
                    learning_rate = learning_rate_base * learn_tmp

                keep_rate = 1.0 - learning_rate

                # learning rate: (1200,)
                # keep_rate: (1200,)
                # query_inputs: (16384,)
                # table_i: (1200, 16384)

                tmp1 = learning_rate[:, np.newaxis]
                tmp2 = query_inputs_flatten[np.newaxis, :]
                term_1 = np.dot(tmp1, tmp2)  # (1200, 16384)

                term_2 = np.multiply(keep_rate[:, np.newaxis], self.table_i)  # (1200, 16384)

                self.table_i = term_1 + term_2


    def post_init(self):
        pass

    def set_matrix_row(self, row_index, row_input, fast_init=True):
        pass

