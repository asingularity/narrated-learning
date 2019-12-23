
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

    def __init__(self, num_entries, input_dim, table=None, use_dumb_dist=False, disable_row_row_dist=False):
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

        self.pickle_save_temp = {}

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

    # @profile
    def query_multiple_rows(self, query_inputs):
        '''
        assumes that only query_input is being used

        :param query_input:
        :return: list of [dists] per each query input
        '''

        #query_arr = np.array(query_inputs, np.float32)
        X = query_inputs

        i_d_t_gpu = self.table_i_gpu
        # print('X.shape', X.shape, 'X.dtype', X.dtype)  # (1024, 48), np.float32
        X_gpu = gpuarray.to_gpu(X)

        # TODO we are not entirely sure that GPUarray does not have a bug below, when you multiply by 2 on-gpu:
        term_1 = (-2 * linalg.dot(X_gpu, i_d_t_gpu)).get()
        # term_1 = -2 * term_1
        term_2 = self.term_2_i
        term_3 = np.sum(X ** 2, axis=1)[:, np.newaxis]
        dists = term_1 + term_2 + term_3

        return dists

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
        i_d_t_gpu = self.table_i_gpu
        X_gpu = gpuarray.to_gpu(X)
        term_1 = linalg.dot(X_gpu, i_d_t_gpu).get()
        term_1 = -2 * term_1
        term_2 = self.term_2_i
        term_3 = np.sum(X ** 2, axis=1)[:, np.newaxis]
        dists = term_1 + term_2 + term_3

        return dists[0]

    def set_multiple_rows(self, row_indices, row_inputs, rows_to_table_dists, fast_init=False):
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
        self.term_2_i[row_index] = np.sum(row_data_i ** 2)

        if not self.disable_row_row_dist:
            # print(self.num_entries, row_to_table_dists.shape, row_to_table_dists.dtype)
            # THIS IS *no longer* A BOTTLENECK SLOW STEP:
            self.d.set_row_dists(row_index=row_index, new_dists=row_to_table_dists, fast_init=fast_init)

        self.row_ages = self.row_ages + 1
        self.row_ages[row_index] = 0

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