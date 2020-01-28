
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import skcuda.linalg as linalg
import skcuda.misc as misc
from dist_matrix_helper import DistMatrixHelper, DumbDistMatrixHelper


class CudaTable(object):
    '''
    to test:
        cuda_table_test.py
    '''

    def __init__(self, num_entries, input_dim, table=None, use_dumb_dist=False, disable_row_row_dist=False, enable_weight_bias=False):
        '''

        need to init twice: input + context, input + output + context

        :param num_entries:
        :param input_dim:
        :param output_dim:
        :param context_dim:
        :param include_layers:

        if k < self.num_layers - 1:
            include_layers = ['ioc', 'ic_only']
        else:
            include_layers = ['io_only', 'i_only']

        '''

        self.enable_weight_bias = enable_weight_bias

        self.post_init_done = False
        self.compute_d = True

        self.num_entries = num_entries
        self.input_dim = input_dim

        self.disable_row_row_dist = disable_row_row_dist

        linalg.init()
        self.size_gb = 0.0

        self.row_ages = np.zeros(self.num_entries, np.int)

        # row-row distance matrix
        if not self.disable_row_row_dist:
            if use_dumb_dist:
                self.d = DumbDistMatrixHelper(num_rows=num_entries, init_dists_val=0.0)
            else:
                self.d = DistMatrixHelper(num_rows=num_entries, init_dists_val=0.0)

        # set back to zero init
        # table = np.random.random((num_entries, input_dim + context_dim + output_dim)).astype(np.float32)

        # i_only

        if table is None:
            # TODO finish this- also incorporate into set_matrix_row
            table_i_only = np.zeros((num_entries, input_dim), np.float32)
        else:
            i_indices = np.arange(input_dim)
            table_i_only = table[:, i_indices]

        self.size_gb += (table_i_only.size * 4.0) / (1e9)

        self.table_i_gpu = gpuarray.to_gpu(np.ascontiguousarray(np.transpose(table_i_only)))
        self.X_i = np.zeros((1, table_i_only.shape[1])).astype(np.float32)
        self.term_2_i = np.sum(table_i_only ** 2, axis=1)

        # self.term_2_i_gpu = gpuarray.to_gpu(np.ascontiguousarray(self.term_2_i))

        self.pickle_save_temp = {}

        # weights
        # for color it would be:
        # per_pixel_weights = 1.0 * np.ones((num_entries, int(table_input_dim) / 3), np.float32)
        # but we are using grayscale pixels for now

        self.weight_masks = 1.0 * np.ones((num_entries, input_dim), np.float32)
        self.weight_masks_square = 1.0 * np.ones((num_entries, input_dim), np.float32)

        self.weight_masks_square_gpu = gpuarray.to_gpu(np.ascontiguousarray(np.transpose(self.weight_masks_square)))

        wsquare_x_table = 1.0 * np.ones((num_entries, input_dim), np.float32)
        self.wsquare_x_table_gpu = gpuarray.to_gpu(np.ascontiguousarray(np.transpose(wsquare_x_table)))

    def prepare_for_save(self):
        # here prepare each layer's cuda table for save: offload matrices from gpu to local!

        self.pickle_save_temp = {}

        self.pickle_save_temp['i'] = self.table_i_gpu.get()

    def init_after_load(self):
        # load tables onto GPU!
        linalg.init()

        self.table_i_gpu = gpuarray.to_gpu(self.pickle_save_temp['i'])

    def get_num_rows(self):
        return self.num_entries

    def get_size_gb(self):
        return self.size_gb

    def post_init(self):
        self.post_init_done = True
        if not self.disable_row_row_dist:
            self.d.post_init()

    def query_multiple_rows(self, query_inputs):
        '''
        assumes that only query_input is being used

        :param query_input:
        :return: list of [dists] per each query input
        '''

        X = query_inputs

        i_d_t_gpu = self.table_i_gpu
        X_gpu = gpuarray.to_gpu(X)

        # reference
        # a = np.dot(np.random.random((200, 200)), np.random.random((200, 200)))

        use_old = True  # new doesn't seem faster
        if use_old:
            if self.enable_weight_bias:

                # trick here is:
                # L2 norm of difference:
                #   (x-y)^2 = x^2 + y^2 - 2xy

                # TERM 1
                # -2 * Wi^2 * Ri * Ii
                # wsquare_r = linalg.multiply(i_d_t_gpu, self.weight_masks_square_gpu)
                wsquare_r = self.wsquare_x_table_gpu  # weight^2 * table_row

                tmp = -2 * linalg.dot(X_gpu, wsquare_r)
                term_1 = tmp.get()

                # TERM 3
                # if weighted, the actual term_3 becomes (256, 6000) instead of (256, 1)
                #   one sum per input, per weight vector
                #   where sum is: Wi^2 * Ii^2 , with i: [0, 1024)
                #
                #   which is: linalg.dot(X**2, W squared gpu) ?
                X_square_gpu = gpuarray.to_gpu(np.multiply(X, X))
                #                  (256, 1024)      (1024, 6000)
                tmp = linalg.dot(X_square_gpu, self.weight_masks_square_gpu)
                term_3 = tmp.get()  # (256, 6000)

            else:
                # TERM 1
                tmp = -2 * linalg.dot(X_gpu, i_d_t_gpu)
                term_1 = tmp.get()

                # TERM 3
                term_3 = np.sum(X ** 2, axis=1)[:, np.newaxis]

            # TERM 2

            # this is already weighted properly
            term_2 = self.term_2_i


            # 256: N: number of input vectors I in query_inputs
            # 6000: T: number of reference vectors R in table
            # 1024: D: vector length i.e. input dimensionality

            # weights_masks:        (6000, 1024)
            # weights_masks_square: (6000, 1024)
            # table_i_gpu:          (1024, 6000)
            # X:                    (256, 1024)
            # term_1:               (256, 6000)
            # term_2:               (6000,)
            # term_3:               (256, 1)
            # dists:                (256, 6000)

            dists = term_1 + term_2 + term_3

            argmin_dists = np.argmin(dists, axis=1)

        # NEW
        else:
            assert False, 'not implemented currently'

            term_1_gpu = -2 * linalg.dot(X_gpu, i_d_t_gpu)

            # This doesn't work because, not updating this when we update self.term_2_i
            term_2_gpu = self.term_2_i_gpu

            term_3 = np.sum(X ** 2, axis=1)[:, np.newaxis]
            term_3_gpu = gpuarray.to_gpu(term_3)

            tmp0 = misc.add_matvec(x_gpu=term_1_gpu, a_gpu=term_2_gpu)
            dists_gpu = misc.add_matvec(x_gpu=tmp0, a_gpu=term_3_gpu)

            # doesn't work; doesn't like unaligned dims
            # dists_gpu = term_1_gpu + term_2_gpu + term_3_gpu

            argmin_dists_gpu = misc.argmin(dists_gpu, axis=1)

            dists = dists_gpu.get()
            argmin_dists = argmin_dists_gpu.get()

        # print(dists, argmin_dists)
        # print('dists.shape', dists.shape)  # dists.shape (256, 6000)  for a total of 256 tiles of input, and 6000 entries. dist for each tile input to each table row
        # print('argmin_dists.shape', argmin_dists.shape)  # argmin_dists.shape (256,), for a total of 16 x 16 = 256 tiles to cover input. for each tile, this is the min row index, in range [0... num_entries]

        return dists, argmin_dists

    def query(self, query_input):
        '''
        at least one of arguments has to be not None

        :param query_input:
        :param query_output: can be None
        :param query_context:
        :return: dists

        TODO
        how do we do partial query?
        potential solutions:
            (1) store separate tables for:
                [input + context] (get next output)
                [input + output + context] (learning)
            (2) use reference as a subset
                but... it won't be contiguous
        '''

        # only three variations supported for now-
        #   all are not None
        #   query_output only is None
        #   TODO query_context only is None
        #   query_output and query_context are None

        assert query_input is not None

        # query_input only
        query_data = query_input
        X = self.X_i
        X[0, :] = query_data[:]

        # # old code, now it uses same query_multiple_rows instead:
        # i_d_t_gpu = self.table_i_gpu
        # X_gpu = gpuarray.to_gpu(X)
        # term_1 = linalg.dot(X_gpu, i_d_t_gpu).get()
        # term_1 = -2 * term_1
        # term_2 = self.term_2_i
        # term_3 = np.sum(X ** 2, axis=1)[:, np.newaxis]
        # dists = term_1 + term_2 + term_3

        dists, argmin_dists = self.query_multiple_rows(query_inputs=X)

        return dists[0]

    # @profile
    def set_matrix_row(self, row_index, row_input, row_to_table_dists, fast_init=False):
        '''
        all arguments have to be not None

        :param row_index:
        :param row_input:
        :param row_output:
        :param row_context:
        :return:

        '''

        col = row_index
        # transposed
        rows_i = self.input_dim
        cols = self.num_entries

        if row_input is None:
            row_input = np.zeros(self.input_dim, np.float32)

        assert row_input.shape[0] == rows_i

        row_data_i = row_input
        arr_gpu_i = gpuarray.to_gpu(row_data_i)
        misc.set_by_index(dest_gpu=self.table_i_gpu, ind=col + cols * np.arange(rows_i), src_gpu=arr_gpu_i, ind_which='dest')

        if self.enable_weight_bias:
            # need to use weights when setting this here
            self.term_2_i[row_index] = np.sum(np.multiply(row_data_i ** 2, self.weight_masks_square[row_index, :]))
        else:
            self.term_2_i[row_index] = np.sum(row_data_i ** 2)

        if not self.disable_row_row_dist:
            # print(self.num_entries, row_to_table_dists.shape, row_to_table_dists.dtype)
            # THIS IS *no longer* A BOTTLENECK SLOW STEP:
            self.d.set_row_dists(row_index=row_index, new_dists=row_to_table_dists, fast_init=fast_init)

        self.row_ages = self.row_ages + 1
        self.row_ages[row_index] = 0

        # new or replaced row: for now, set weights to all ones
        self.set_row_weights(row_index=row_index, weights=np.ones(self.input_dim, np.float32), row_values=row_data_i)

    def set_row_weights(self, row_index, weights, row_values):
        '''

        # needs row values to set wsquared*table

        :param row_index:
        :param weights:
        :param row_values:
        :return:
        '''

        # IMPORTANT: WEIGHTS NEED TO SUM TO 1
        # TODO need to update visualizer to take this into account!
        weights = weights * 1.0 / np.sum(weights)
        weights_square = weights ** 2

        self.weight_masks[row_index, :] = weights[:]
        self.weight_masks_square[row_index, :] = weights_square[:]

        if self.enable_weight_bias:
            # also need to set weights masks square gpu here!
            arr_gpu_i = gpuarray.to_gpu(weights_square)
            col = row_index
            # transposed
            rows_i = self.input_dim
            cols = self.num_entries
            misc.set_by_index(dest_gpu=self.weight_masks_square_gpu, ind=col + cols * np.arange(rows_i),
                              src_gpu=arr_gpu_i, ind_which='dest')

            arr_gpu_i_2 = gpuarray.to_gpu(np.multiply(weights_square, row_values))

            misc.set_by_index(dest_gpu=self.wsquare_x_table_gpu, ind=col + cols * np.arange(rows_i),
                              src_gpu=arr_gpu_i_2, ind_which='dest')

            # need to reset self.term_2_i here
            row_data_i, _, _ = self.get_matrix_row(row_index=row_index)
            self.term_2_i[row_index] = np.sum(np.multiply(row_data_i ** 2, self.weight_masks_square[row_index, :]))


    def get_oldest_row_ind(self):
        return np.argmax(self.row_ages)

    def get_min_dist(self):
        '''

        :return: min_dist, index_r, index_c
        '''
        if not self.disable_row_row_dist:
            return self.d.get_min_dist()
        else:
            return None

    def get_max_dist(self):
        if not self.disable_row_row_dist:
            return self.d.get_max_dist()
        else:
            return None

    def get_matrix_row(self, row_index):

        col = row_index
        rows = self.input_dim
        cols = self.num_entries

        row_i = misc.get_by_index(src_gpu=self.table_i_gpu, ind=col + cols * np.arange(rows)).get()
        row_input = row_i[0:self.input_dim]
        row_output = None
        row_context = None

        return row_input, row_output, row_context  # TODO deprecate this

    def get_table_from_gpu(self):
        '''
        WARNING: SLOW
        :return:
        '''
        try:
            table_numpy = self.table_gpu.get()
        except:
            try:
                table_numpy = self.table_io_gpu.get()
            except:
                table_numpy = self.table_i_gpu.get()

        return table_numpy

    def seq_kmeans_adapt(self, row_index, rate, row_input, row_output, row_context, row_to_table_dists):
        '''
        all arguments have to be not None

        :param row_index:
        :param rate:
        :param row_input:
        :param row_output:
        :param row_context:
        :return:
        '''

        # self.table[ind, :] = 0.9 * self.table[ind, :] + 0.1 * new_entry
        current_row_input, current_row_output, current_row_context = self.get_matrix_row(row_index=row_index)
        assert 0.0 < rate < 1.0, 'invalid rate: ' + str(rate)

        if row_input is not None:
            new_row_input = current_row_input + rate * (row_input - current_row_input)
        else:
            new_row_input = current_row_input

        if row_output is not None:
            new_row_output = current_row_output + rate * (row_output - current_row_output)
        else:
            new_row_output = current_row_output

        if row_context is not None:
            new_row_context = current_row_context + rate * (row_context - current_row_context)
        else:
            new_row_context = current_row_context

        self.set_matrix_row(row_index=row_index,
                            row_input=new_row_input,
                            row_output=new_row_output,
                            row_context=new_row_context,
                            row_to_table_dists=row_to_table_dists,
                            fast_init=True)

    def adapt(self, row_index, rate, row_input, row_output, row_context):
        '''
        all arguments have to be not None

        :param row_index:
        :param rate:
        :param row_input:
        :param row_output:
        :param row_context:
        :return:
        '''

        # self.table[ind, :] = 0.9 * self.table[ind, :] + 0.1 * new_entry
        current_row_input, current_row_output, current_row_context = self.get_matrix_row(row_index=row_index)
        assert 0.0 < rate < 1.0
        keep = 1.0 - rate

        if row_input is not None:
            new_row_input = keep * current_row_input + rate * row_input
        else:
            new_row_input = current_row_input

        if row_output is not None:
            new_row_output = keep * current_row_output + rate * row_output
        else:
            new_row_output = current_row_output

        if row_context is not None:
            new_row_context = keep * current_row_context + rate * row_context
        else:
            new_row_context = current_row_context

        self.set_matrix_row(row_index=row_index,
                            row_input=new_row_input,
                            row_output=new_row_output,
                            row_context=new_row_context)


    def _UNUSED_set_multiple_rows(self, row_indices, row_inputs, rows_to_table_dists, fast_init=False):
        '''

        set multiple rows at once

        :param row_indices: array
        :param row_inputs: list of arrays
        :param row_to_table_dists: list of arrays
        :param fast_init: bool
        :return:
        '''

        # necessary for CudaMultiTable

        # transposed
        n_rows = self.input_dim
        n_cols = self.num_entries
        cols = row_indices

        if row_inputs is not None:
            arr_input = np.zeros(self.input_dim * row_indices.shape[0], np.float32)

            k = 0
            for row_index in list(row_indices):
                row_data_i = row_inputs[k]
                row_index_int = int(row_index)
                self.term_2_i[row_index_int] = np.sum(row_data_i ** 2)
                arr_input[k * row_inputs[0].shape[0]:(k + 1) * row_inputs[0].shape[0]] = row_data_i[:]
                k += 1

            arr_gpu_i = gpuarray.to_gpu(arr_input)

            ind_1 = np.repeat(cols, n_rows)
            ind_2 = np.tile(n_cols * np.arange(n_rows), cols.shape[0])

            misc.set_by_index(dest_gpu=self.table_i_gpu, ind=ind_1 + ind_2, src_gpu=arr_gpu_i, ind_which='dest')

        # TODO enable dists below

        if rows_to_table_dists is not None:
            # print(self.num_entries, row_to_table_dists.shape, row_to_table_dists.dtype)
            # THIS IS A BOTTLENECK SLOW STEP:

            # TODO internally, this could process multiple rows in one for loop! parallelize it!

            # TODO for now make a loop!
            self.d.set_row_dists(row_index=row_index, new_dists=row_to_table_dists, fast_init=fast_init)




class CudaQuery_DEPRECATED(object):
    def __init__(self, input_data):
        self.original_input_data_shape = input_data.shape
        query_data = np.zeros(input_data.shape[1], np.float32)

        self.input_data = input_data #.flatten()  # row-major order. do not have to flatten.
        self.tmp_data = np.zeros(self.input_data.shape, np.float32)

        #self.func = mod.get_function("knn_query")

        #self.input_data_gpu = cuda.mem_alloc(self.input_data.size * self.input_data.dtype.itemsize)
        #self.query_data_gpu = cuda.mem_alloc(query_data.size * query_data.dtype.itemsize)
        #self.tmp_data_gpu = cuda.mem_alloc(self.tmp_data.size * self.tmp_data.dtype.itemsize)

        #cuda.memcpy_htod(self.input_data_gpu, self.input_data)

        self.X = np.zeros((1, input_data.shape[1])).astype(np.float32)
        self.input_data_transpose = np.ascontiguousarray(np.transpose(self.input_data))

        self.term_2 = np.sum(self.input_data ** 2, axis=1)

        #print 'contiguous?'
        #print self.X.flags['C_CONTIGUOUS']
        #print self.input_data_transpose.flags['C_CONTIGUOUS']

        linalg.init()
        self.i_d_t_gpu = gpuarray.to_gpu(self.input_data_transpose)

    def get_num_rows(self):
        return self.original_input_data_shape[0]

    def retransfer_matrix_for_test(self):
        #self.term_2 = np.sum(self.input_data ** 2, axis=1)
        del self.i_d_t_gpu
        self.i_d_t_gpu = gpuarray.to_gpu(self.input_data_transpose)

    def set_matrix_row(self, row_index, row_data):
        '''
            since we store transposed matrix: row becomes column
        '''

        col = row_index
        rows = self.input_data_transpose.shape[0]
        cols = self.input_data_transpose.shape[1]

        assert row_data.shape[0] == rows

        arr_gpu = gpuarray.to_gpu(row_data)
        misc.set_by_index(dest_gpu=self.i_d_t_gpu, ind=col + cols * np.arange(rows), src_gpu=arr_gpu, ind_which='dest')

        # also change self.term2
        # self.term_2.shape  # 10000 (original rows, transposed cols)
        # self.term_2 = np.sum(self.input_data ** 2, axis=1)
        self.term_2[row_index] = np.sum(row_data ** 2)

    def query(self, query_data):
        X = self.X
        X[0, :] = query_data[:]

        start_all = time.time()
        i_d_t_gpu = self.i_d_t_gpu

        #i_d_t_gpu = gpuarray.to_gpu(np.random.randn(4, 4).astype(np.float32))
        #X_gpu = gpuarray.to_gpu(np.random.randn(4, 4).astype(np.float32))

        X_gpu = gpuarray.to_gpu(X)
        start_1 = time.time()
        # DOESN'T WORK:
        # term_1 = gpuarray.dot(X_gpu, i_d_t_gpu).get()
        # WORKS:
        term_1 = linalg.dot(X_gpu, i_d_t_gpu).get()
        end_1 = time.time() - start_1

        start_2 = time.time()
        term_1 = -2 * term_1  # 15%
        end_2 = time.time() - start_2

        term_2 = self.term_2

        term_3 = np.sum(X ** 2, axis=1)[:, np.newaxis]

        dists = term_1 + term_2 + term_3  # 25%

        #ind = np.argmin(dists)  # <= 10%

        #dist = dists[0, ind]
        #end_all = time.time() - start_all

        # print 'ratio of time spent in dot product: ', end_1 / end_all
        # print 'ratio of time spent in 2: ', end_2 / end_all
        return dists[0]


if __name__ == '__main__':
    pass