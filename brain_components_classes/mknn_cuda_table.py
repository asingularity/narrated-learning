from brain_components_classes.multi_sparse_binary_knn import MultiSparseBinaryKNN
from cython_match import sum_match
import random
from utils.fps_counter import FPSCounter
import time
import numpy as np
import os


def notes():

    pass

    # TODO must define self.post_init_done to reflect underlying table, or make it a function
    #   can function be called without () so it just calls underlying that way?

    # TODO for get_table_ims, need cuda_table.table_i defined similarly from underlying table, or a function

    # THIS WON'T WORK- uses dot product to compute distance, not appropriate in this case
    # self.table = CudaTable(num_entries=num_entries,
    #                       input_dim=input_dim,
    #                       disable_row_row_dist=disable_row_row_dist,
    #                       enable_weight_bias=enable_weight_bias)

    # instead of using cuda table, we should fix multi sparse binary knn class for distance compute (and, optionally, cuda speedup)

    # TODO instead we are using multi sparse binary knn directly from multi layer dynamic tiles...
    # --- except this is a problem because:
    #   MultiSparseBinaryKNN:
    #       does not implement query_multiple_rows
    #       does not implement distance helper / seq-nn
    #       assumes an output (prediction) is also trained

    # TODO instead, could make this a new cuda table, containing multiple tables (one per tile)
    # but, this table would be huge! would not fit on gpu


DTYPE = np.int32


class MKNNCudaTable(object):
    def __init__(self, num_tiles, num_entries, input_dim, sparse_io_dim):
        '''

        CudaTable-like implementation of MultiSparseBinaryKNN
            adds seq-nn functionality built into cuda table

        Won't work for large tables, i.e. prediction with lateral context, because too large to store on GPU
            Since independent table per tile location
            And multiple tiles are context per location

        :param num_tiles:       number of tables
        :param num_entries:     num rows per table
        :param input_dim:       number of sub-tiles composing a tile; typically 4
        :param sparse_io_dim:   max value per element of input; rows per tile of prev layer cuda tables
        '''
        '''
        :param params:
        '''

        self.num_tiles = num_tiles          # number of tables
        self.num_entries = num_entries      # num rows per table
        self.input_dim = input_dim          # number of sub-tiles composing a tile; typically 4; i.e. num_sparse_inputs
        self.sparse_io_dim = sparse_io_dim  # max value per element of input; rows per tile of prev layer cuda tables

        # start with manual, numpy/cython implementation

        # int because this is just indices. dim=num_sparse_inputs since assuming one-hot for now on input
        self.input_arr = np.zeros((self.num_tiles * self.num_entries, self.input_dim), DTYPE)

        self.last_query_inputs = None

    def post_init(self):
        pass

    def query_multiple_rows(self, query_inputs):
        ''''

        difference from standard cuda query is that we don't care about dist, just number of non-matching elements

        '''

        self.last_query_inputs = query_inputs.copy()

        # use cython sum_match

        assert query_inputs.shape[0] == self.num_tiles
        assert query_inputs.shape[1] == self.input_dim

        # knn_input_win_rows_2d_arr:  # (196, 63) : (num_knn, num_sparse_inputs)
        # self.input_arr:             # (392000, 63) : (num_knn * N, num_sparse_inputs)

        # if self.use_cuda:
        #     expand_input_gpu = gpuarray.to_gpu(expand_input)
        #     tmp_gpu = misc.sum(misc.subtract(self.input_arr_gpu, expand_input_gpu)==0, axis=1)
        #     tmp = tmp_gpu.get()

        arr_out = np.zeros(self.input_arr.shape[0], DTYPE)
        sum_match(self.input_arr, knn_input_win_rows_2d_arr, arr_out, DTYPE(self.num_knn), DTYPE(self.N), self.learn_index)
        tmp = arr_out  # tmp: (392000,)

        # now calc win rows: argmax per block of N

        tmp2 = tmp.reshape((self.num_knn, self.N))
        win_knn_rows_arr = np.argmax(tmp2, axis=1)

        # win_knn_rows_arr: (self.num_knn,)  -- values in range [0, self.N]
        # rel_knn_rows = np.arange(self.num_knn) * self.N + win_knn_rows_arr  # relative indices to output_prop_arr


        return dists, argmin_dists

    def set_matrix_row(self, row_index, row_input, row_to_table_dists=None, row_weights=None, fast_init=False, fast_set_weight=False, override_disable_dists=False):
        pass

    def get_min_dist(self):
        '''

        :return: min_dist, index_r, index_c
        '''
        if not self.disable_row_row_dist:
            return self.d.get_min_dist()
        else:
            return None

    def seq_nn_learn_last_query(self):
        '''

        references:
            _learn_table_layer_0, from multi_layer_dynamic_tiles.py

        uses self.last_query_inputs
            from last time self.query_multiple_rows was called

        (1) if tables are not full yet, add rows to tables and update dists and init count

        (2) if tables are full, do seq-nn learning
            for each sub-table (each tile's table):
                a. get min dist [input <-> table]
                b. get min dist [table <-> table]
                c. if min_dist_[input <-> table] > min_dist_[table <-> table]:
                      - replace a min dist table row with input
                      - update dists

        :return:
        '''

        # this will be slow

# @profile
def main():
    random.seed(1)
    np.random.seed(1)

    num_tiles = 64 * 64  # 4096
    num_entries = 400  # per tile
    input_dim = 4
    sparse_io_dim = 400  # thousands; prev row num rows per tile

    DTYPE = np.float32

    cuda_table = MKNNCudaTable(num_tiles=num_tiles,  # num tables
                               num_entries=num_entries,  # per table
                               input_dim=input_dim,  # number of sub-tiles composing a tile; typically 4
                               sparse_io_dim=sparse_io_dim)  # max value per element of input; rows per tile of prev layer cuda tables

    fps = FPSCounter()

    num_input_rows = num_tiles  # how many tiles?

    for k in range(1000):

        tiles_inputs = np.random.randint(0, sparse_io_dim - 1, (num_input_rows, input_dim)).astype(DTYPE)

        # lookup
        dists, argmin_dists = cuda_table.query_multiple_rows(query_inputs=tiles_inputs)

        # train
        cuda_table.seq_nn_learn_last_query()

        fps.update()

    print(predict_win_rows)


if __name__ == '__main__':
    main()


