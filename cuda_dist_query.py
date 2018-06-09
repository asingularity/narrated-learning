
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import skcuda.linalg as linalg
import skcuda.misc as misc


class CudaTable(object):
    '''
    to test:
        cuda_table_test.py
    '''

    def __init__(self, num_entries, input_dim, output_dim, context_dim, table=None):
        '''

        need to init twice: input + context, input + output + context

        :param num_entries:
        :param input_dim:
        :param output_dim:
        :param context_dim:
        '''

        self.num_entries = num_entries
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim

        linalg.init()

        # set back to zero init
        # table = np.random.random((num_entries, input_dim + context_dim + output_dim)).astype(np.float32)

        if table is None:
            table = np.zeros((num_entries, input_dim + output_dim + context_dim), np.float32)

        self.table_gpu = gpuarray.to_gpu(np.ascontiguousarray(np.transpose(table)))
        self.X = np.zeros((1, table.shape[1])).astype(np.float32)
        self.term_2 = np.sum(table ** 2, axis=1)

        if table is None:
            table_i_c_only = np.zeros((num_entries, input_dim + context_dim), np.float32)
        else:
            i_indices = np.arange(input_dim)
            c_indices = np.arange(input_dim + output_dim, input_dim + output_dim + context_dim)
            table_i_c_only = table[:, np.concatenate((i_indices, c_indices))]

        self.table_ic_gpu = gpuarray.to_gpu(np.ascontiguousarray(np.transpose(table_i_c_only)))
        self.X_ic = np.zeros((1, table_i_c_only.shape[1])).astype(np.float32)
        self.term_2_ic = np.sum(table_i_c_only ** 2, axis=1)

    def query(self, query_input, query_output, query_context):
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

        # only two variations supported for now- only query_output can be None
        assert query_input is not None and query_context is not None

        if query_output is None:
            query_data = np.concatenate((query_input, query_context))
            X = self.X_ic
            X[0, :] = query_data[:]
            i_d_t_gpu = self.table_ic_gpu
            X_gpu = gpuarray.to_gpu(X)
            term_1 = linalg.dot(X_gpu, i_d_t_gpu).get()
            term_1 = -2 * term_1
            term_2 = self.term_2_ic
            term_3 = np.sum(X ** 2, axis=1)[:, np.newaxis]
            dists = term_1 + term_2 + term_3
        else:
            query_data = np.concatenate((query_input, query_output, query_context))
            X = self.X
            X[0, :] = query_data[:]
            i_d_t_gpu = self.table_gpu
            X_gpu = gpuarray.to_gpu(X)
            term_1 = linalg.dot(X_gpu, i_d_t_gpu).get()
            term_1 = -2 * term_1
            term_2 = self.term_2
            term_3 = np.sum(X ** 2, axis=1)[:, np.newaxis]
            dists = term_1 + term_2 + term_3

        return dists[0]

    def set_matrix_row(self, row_index, row_input, row_output, row_context):
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
        rows = self.input_dim + self.output_dim + self.context_dim
        rows_ic = self.input_dim + self.context_dim
        cols = self.num_entries

        assert row_input.shape[0] + row_output.shape[0] + row_context.shape[0] == rows
        assert row_input.shape[0] + row_context.shape[0] == rows_ic

        row_data_all = np.concatenate((row_input, row_output, row_context))
        arr_gpu = gpuarray.to_gpu(row_data_all)
        misc.set_by_index(dest_gpu=self.table_gpu, ind=col + cols * np.arange(rows), src_gpu=arr_gpu, ind_which='dest')
        self.term_2[row_index] = np.sum(row_data_all ** 2)

        row_data_ic = np.concatenate((row_input, row_context))
        arr_gpu_ic = gpuarray.to_gpu(row_data_ic)
        misc.set_by_index(dest_gpu=self.table_ic_gpu, ind=col + cols * np.arange(rows_ic), src_gpu=arr_gpu_ic, ind_which='dest')
        self.term_2_ic[row_index] = np.sum(row_data_ic ** 2)

    def get_matrix_row(self, row_index):
        col = row_index
        rows = self.input_dim + self.output_dim + self.context_dim
        cols = self.num_entries

        full_row = misc.get_by_index(src_gpu=self.table_gpu, ind=col + cols * np.arange(rows)).get()
        row_input = full_row[0:self.input_dim]
        row_output = full_row[self.input_dim:self.input_dim + self.output_dim]
        row_context = full_row[self.input_dim+self.output_dim:self.input_dim+self.output_dim+self.context_dim]

        return row_input, row_output, row_context

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

        new_row_input = keep * current_row_input + rate * row_input
        new_row_output = keep * current_row_output + rate * row_output
        new_row_context = keep * current_row_context + rate * row_context

        self.set_matrix_row(row_index=row_index,
                            row_input=new_row_input,
                            row_output=new_row_output,
                            row_context=new_row_context)


class CudaQuery(object):
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